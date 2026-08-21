#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN-VMC Adam+MARCH 版: 自旋 1/2 反铁磁 Heisenberg 链 (PBC)。
=========================================================
从 rnn_heisenberg.py 生成 (2026-08-16). 完全自包含 (仅 torch/numpy), 自动检测核数.
优化: GRUCell 融合算子 + addmm + 磁化配额每步; torch.compile 自动回退.
训练: 阶段1 VMC-Adam (lr=2.5e-4) → 阶段2 MARCH 精修.
用法: python rnn_adam_march.py --L 30 --mode mc --warm_det 0 --lr 2.5e-4 --ns 500 --steps 8000 --march_epochs 200
"""

import torch
import math
import random
import argparse
import time
import os
import itertools
import numpy as np

DT = torch.float64


def auto_threads(reserve=2):
    """自动检测 CPU 核数并分配 torch 线程 (预留 reserve 核给系统/其他任务)."""
    ncpu = os.cpu_count() or 1
    nt = max(1, ncpu - reserve)
    torch.set_num_threads(nt)
    print(f'[线程] CPU {ncpu} 核 → 分配 {nt} 线程 (预留 {reserve})', flush=True)
    return ncpu, nt


# ---------------------------------------------------------------- Heisenberg ED
def heisenberg_ed(L):
    """精确对角化自旋 1/2 Heisenberg 链 (PBC), 返回 (E0, E0/site)。"""
    N = 2 ** L
    diag = np.zeros(N)
    off = []
    for i in range(N):
        s = np.array([(i >> k) & 1 for k in range(L)], dtype=int)
        sig = 1 - 2 * s                      # bit=1 -> σ=-1 (down)
        for j in range(L):
            j1 = (j + 1) % L
            diag[i] += sig[j] * sig[j1] / 4.0
            if sig[j] != sig[j1]:
                t = s.copy(); t[j] ^= 1; t[j1] ^= 1
                k = 0
                for b in range(L):
                    k |= t[b] << b
                off.append((i, k, 0.5))
    H = np.diag(diag)
    for (i, k, v) in off:
        H[i, k] += v
    w = np.linalg.eigvalsh(H)
    return w[0], w[0] / L


def enumerate_sz0(L):
    """枚举 Sz=0 扇区全部构型 (自旋 ±1 列表的迭代器)。"""
    for ones in itertools.combinations(range(L), L // 2):
        spins = [-1.0] * L
        for i in ones:
            spins[i] = 1.0
        yield spins


def enumerate_sector(L, M):
    """枚举磁化 M=Σσ 扇区全部构型 (M=2: (L+M)/2 个 up, (L-M)/2 个 down)。"""
    n_up = (L + M) // 2
    n_down = (L - M) // 2
    for ups in itertools.combinations(range(L), n_up):
        spins = [-1.0] * L
        for i in ups:
            spins[i] = 1.0
        yield spins


# ---------------------------------------------------------------- RNN 波函数
class RNNHeisenberg(torch.nn.Module):
    """cRNN 波函数 (GRU + Softmax 幅度 + Softsign 相位). 批量 API (GCNN 范式)。"""

    def __init__(self, L, dh=32, seed=None, sz0=True, device='cpu',
                 dtype=torch.float32, ratio_clamp=20.0, eloc_clip=50.0,
                 magnetization=0, twist=0.0):
        super().__init__()
        self.L, self.dh, self.dv = L, dh, 2
        self.sz0 = sz0
        self.device = torch.device(device)
        self.dtype = dtype
        self.ratio_clamp = ratio_clamp   # E_loc 比值实部钳制 (防 exp 溢出, GCNN 同款)
        self.eloc_clip = eloc_clip       # 学习损失中 E_loc 值钳制 (稳健优化)
        # 磁化扇区: Σσ = magnetization. 偶数链基态 M=0; 奇数链基态 M=±1
        self.magnetization = magnetization
        self.n_up_t = (L + magnetization) // 2     # up 上限 (8 for L=15,M=+1)
        self.n_down_t = (L - magnetization) // 2   # down 上限 (7 for L=15,M=+1)
        self.twist = twist                         # 扭曲边界相位 θ (动量 k=θ, 激发态色散)
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)

        def init(*shape):
            return torch.nn.init.uniform_(
                torch.empty(*shape, dtype=self.dtype, device=self.device),
                -1.0, 1.0, generator=g)

        # GRU (torch.nn.GRUCell 融合 C++ 算子, 比手动 addmm 快 ~7x)
        # 门序 [r; z; n], 等价于原 Wu/Wr/Wc1/Wc2/bu/br/bc1/bc2 布局
        self.gru = torch.nn.GRUCell(2, dh)
        with torch.no_grad():
            self.gru.weight_ih.copy_(init(3 * dh, 2))
            self.gru.weight_hh.copy_(init(3 * dh, dh))
            self.gru.bias_ih.copy_(init(3 * dh))
            self.gru.bias_hh.copy_(init(3 * dh))
        # 单一复数输出头 (GCNN 范式): w = (U_re@h+c_re) + i(U_im@h+c_im)
        # 幅度 p=softmax(2·Re w), 相位 φ=Im w —— 振幅相位从同一个 w 出, 共用 GRU 隐态同时学习
        self.U_re = torch.nn.Parameter(init(2, dh))   # 复数输出实部
        self.c_re = torch.nn.Parameter(init(2))
        self.U_im = torch.nn.Parameter(init(2, dh))   # 复数输出虚部
        self.c_im = torch.nn.Parameter(init(2))
        self.h0 = torch.nn.Parameter(init(dh))        # 初始隐态
        # one-hot: σ=+1 -> [1,0], σ=-1 -> [0,1]
        self.register_buffer('onehot',
                             torch.tensor([[1., 0.], [0., 1.]],
                                          dtype=self.dtype, device=self.device))
        self.param_count = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _softsign(x):
        return x / (1 + torch.abs(x))

    # ------------------------------------------------------- 批量前向: σ → lnψ
    def ln_psi(self, states):
        """states (B, L) ±1 → lnψ (B,) 复数 (带计算图). 批量 RNN 前向.
        自回归顺序: 预测 σ_i 于 h(编码 σ_<i), 再更新 h (保证 ΣP=1).
        sz0: 论文零磁化配额 (i≥L/2 起, 每构型独立计数)."""
        B, L = states.shape
        idx = (states < 0).to(torch.long)            # (B,L) 0:+1(up), 1:-1(down)
        onehot = self.onehot[idx]                    # (B,L,2)
        h = self.h0.expand(B, -1).clone()            # (B,dh)
        n_up = torch.zeros(B, dtype=self.dtype, device=self.device)
        n_down = torch.zeros(B, dtype=self.dtype, device=self.device)
        half = L // 2
        logP = torch.zeros(B, dtype=self.dtype, device=self.device)
        phi = torch.zeros(B, dtype=self.dtype, device=self.device)
        ar = torch.arange(B, device=self.device)
        for i in range(L):
            x = onehot[:, i, :]                      # (B,2)
            w_re = torch.addmm(self.c_re, h, self.U_re.t())   # (B,2)
            w_im = torch.addmm(self.c_im, h, self.U_im.t())   # (B,2)
            y1 = torch.softmax(2.0 * w_re, dim=1)    # 幅度条件 p=softmax(2Re w)
            q = self._quota(y1, n_up, n_down)        # 磁化配额 (每步施加, 论文式)
            logP = logP + torch.log(q[ar, idx[:, i]])
            phi = phi + w_im[ar, idx[:, i]]          # 相位 φ=Im w
            h = self._gru(h, x)                      # 更新 h 摄入 σ_i (addmm 融合)
            n_up = n_up + (idx[:, i] == 0).to(self.dtype)
            n_down = n_down + (idx[:, i] == 1).to(self.dtype)
        return 0.5 * logP + 1j * phi

    def _quota(self, p, n_up, n_down):
        """磁化配额 (论文每个 site 都施加): 超配额分支清零并重归一化."""
        if self.sz0:
            p = p.clone()
            p[n_down >= self.n_down_t, 1] = 0.0      # down 配额满, 禁 down
            p[n_up >= self.n_up_t, 0] = 0.0          # up 配额满, 禁 up
            p = p / p.sum(1, keepdim=True)
        return p                 # ψ = exp(0.5logP + iφ)

    def wavefunction(self, states):
        return torch.exp(self.ln_psi(states))

    # ------------------------------------------------------- 批量局域能量 (GCNN)
    @torch.no_grad()
    def E_loc(self, states, ln_psi_old):
        """states (B,L) ±1, ln_psi_old (B,) 复数 → E_loc (B,) 复数.
        GCNN 范式: 逐 bond 用 torch.where 找翻转构型, 批量算 ln_new, index_add 累加."""
        B, L = states.shape
        E = torch.zeros(B, dtype=torch.complex128 if self.dtype == torch.float64
                        else torch.complex64, device=self.device)
        # 对角项 (1/4)Σ σ_i σ_{i+1}
        E = E + 0.25 * torch.sum(states * torch.roll(states, 1, dims=1), dim=1)
        # 全部 bond (i, i+1) 一次批量: (B, L-1) 中 σ_i≠σ_{i+1} 的非 wrap bond 对
        si = states[:, :-1]
        sj = states[:, 1:]
        mask = (si != sj)                            # (B, L-1)
        if mask.any():
            cidx, bidx = torch.where(mask)           # config c (行), bond i (列)
            ns = states[cidx].clone()                # (n, L)
            r = torch.arange(len(cidx))
            ns[r, bidx] = -ns[r, bidx]
            ns[r, bidx + 1] = -ns[r, bidx + 1]
            ln_new = self.ln_psi(ns)                 # 一次批量 (n,) 复数
            d = ln_new - ln_psi_old[cidx]
            ratio = torch.exp(torch.complex(
                torch.clamp(d.real, max=self.ratio_clamp), d.imag))
            E.index_add_(0, cidx, 0.5 * ratio)
        # wrap bond (L-1, 0): 需翻转首尾, 单独处理
        w = torch.where(states[:, L - 1] != states[:, 0])[0]
        if len(w) > 0:
            ns = states[w].clone()
            ns[:, L - 1] = -ns[:, L - 1]
            ns[:, 0] = -ns[:, 0]
            ln_new = self.ln_psi(ns)
            d = ln_new - ln_psi_old[w]
            ratio = torch.exp(torch.complex(
                torch.clamp(d.real, max=self.ratio_clamp), d.imag))
            if abs(self.twist) > 1e-12:
                # 扭曲边界: wrap bond 系数 0.5·e^{∓iθ}. σ_{L-1}=+1(|↑↓⟩) -> e^{-iθ}, -1 -> e^{+iθ}
                phase = torch.exp(-1j * self.twist * states[w, L - 1])
                E.index_add_(0, w, 0.5 * phase * ratio)
            else:
                E.index_add_(0, w, 0.5 * ratio)
        return E

    # ------------------------------------------------------- 批量自回归采样
    def _gru(self, h, x):
        """一步 GRU (torch.nn.GRUCell 融合 C++ 算子, ~7x 快).
        不带 no_grad: ln_psi 需要梯度; sample 整体已 @torch.no_grad()."""
        return self.gru(x, h)

    @torch.no_grad()
    def sample(self, batch, sz0=True):
        """批量自回归采样 states (B, L) ±1 (与 ln_psi 同序同条件)。
        优化: 前半无配额快路径 + addmm 融合 + 常量预分配."""
        B, L = batch, self.L
        states = torch.zeros(B, L, dtype=self.dtype, device=self.device)
        h = self.h0.expand(B, -1).clone()
        n_up = torch.zeros(B, dtype=self.dtype, device=self.device)
        n_down = torch.zeros(B, dtype=self.dtype, device=self.device)
        one = torch.ones(B, dtype=self.dtype, device=self.device)
        neg = -one
        for i in range(L):
            p = torch.softmax(2.0 * torch.addmm(self.c_re, h, self.U_re.t()),
                              dim=1)
            p = self._quota(p, n_up, n_down)         # 磁化配额 (每步, 论文式)
            up = torch.rand(B, device=self.device) < p[:, 0]
            states[:, i] = torch.where(up, one, neg)
            n_up = n_up + up.to(self.dtype)
            n_down = n_down + (~up).to(self.dtype)
            h = self._gru(h, self.onehot[(~up).to(torch.long)])
        return states

    # ------------------------------------------------------- 确定性 (枚举 + ψ†Hψ)
    def _build_H(self):
        """Heisenberg H 在磁化扇区构型基下 (M,M) 矩阵, 缓存."""
        if getattr(self, '_H', None) is not None:
            return self._H
        cfgs = list(enumerate_sector(self.L, self.magnetization))
        idx_of = {tuple(c): a for a, c in enumerate(cfgs)}
        M = len(cfgs)
        H = torch.zeros((M, M), dtype=torch.complex128, device=self.device)
        for a, c in enumerate(cfgs):
            s = torch.tensor(c, dtype=self.dtype, device=self.device)
            H[a, a] = 0.25 * torch.dot(s, torch.roll(s, 1))
            for i in range(self.L):
                j = (i + 1) % self.L
                if c[i] != c[j]:
                    cp = list(c); cp[i] = -cp[i]; cp[j] = -cp[j]
                    coef = 0.5
                    if i == self.L - 1 and abs(self.twist) > 1e-12:
                        coef = 0.5 * np.exp(-1j * self.twist * c[i])
                    H[idx_of[tuple(cp)], a] += coef
        self._H = H
        return H

    def energy_det(self, grad=False):
        """确定性: 枚举磁化扇区全部构型 → ψ 向量 → E = ψ†Hψ/ψ†ψ (精确).
        NQS 范式 (类比 FeI2 的 b†H_kb/‖b‖²)。grad=True 时反向累积梯度。"""
        cfgs = torch.tensor(list(enumerate_sector(self.L, self.magnetization)),
                            dtype=self.dtype, device=self.device)
        psi = self.wavefunction(cfgs)                # (M,) 复数, 带图
        Hc = self._build_H().to(psi.dtype)
        E = (psi.conj() @ Hc @ psi) / (psi.conj() @ psi)
        E = E.detach() if not grad else E
        if grad:
            E.real.backward()
        return E

    # ------------------------------------------------------- MARCH (SR 自然梯度)
    def _get_flat(self):
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def _set_flat(self, flat):
        idx = 0
        for p in self.parameters():
            p.data.copy_(flat[idx:idx + p.numel()].reshape(p.shape))
            idx += p.numel()

    def march_step(self, sps, parameter, nu, tau=0.05, mu=0.95, lam=0.001,
                   beta0=0.995):
        """一步 MARCH (GCNN 公式, 移植自 MLP修复版.py):
        O=concat[∂Re,∂Im] 中心化; M=OOᵀ+λI; δθ=OᵀM⁻¹(E_aug−O·μ·p);
        nu=β·nu+(δθ−δθ_prev)². 返回 (parameter_k, nu)。"""
        ns = sps.shape[0]
        params = list(self.parameters())
        nparam = parameter.numel()
        # 逐样本 Jacobian: O_re/O_im (ns, nparam)  (需计算图, 不能 no_grad)
        O_re = torch.zeros((ns, nparam), dtype=self.dtype, device=self.device)
        O_im = torch.zeros((ns, nparam), dtype=self.dtype, device=self.device)
        for s in range(ns):
            ln = self.ln_psi(sps[s:s + 1])[0]
            gr = torch.autograd.grad(ln.real, params, retain_graph=True)
            gi = torch.autograd.grad(ln.imag, params)
            O_re[s] = torch.cat([g.flatten() for g in gr])
            O_im[s] = torch.cat([g.flatten() for g in gi])
        O = torch.cat([O_re, O_im], dim=0)                 # (2ns, nparam)
        O = (O - O.mean(dim=0)) / math.sqrt(ns)
        # 能量 (no_grad)
        with torch.no_grad():
            ln0 = self.ln_psi(sps)
            E = self.E_loc(sps, ln0)
            E_mean = E.mean()
            E_c = -tau * (E - E_mean)
            E_aug = torch.cat([E_c.real, E_c.imag])         # (2ns,)
        M = O @ O.T + lam * torch.eye(2 * ns, dtype=self.dtype, device=self.device)
        rhs = E_aug - O @ (mu * parameter)
        x = None
        for extra in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                L = torch.linalg.cholesky(
                    M + extra * torch.eye(2 * ns, dtype=self.dtype,
                                          device=self.device))
                x = torch.cholesky_solve(rhs.reshape(2 * ns, 1), L).reshape(2 * ns)
                break
            except RuntimeError:
                continue
        if x is None:
            return parameter, nu
        temp = torch.mv(O.T, x)                             # (nparam,)
        parameter_k = temp / torch.sqrt(nu) + mu * parameter
        nu = (beta0 * nu + (parameter_k - parameter) ** 2).clamp(min=1e-6, max=1e6)
        return parameter_k.detach(), nu

    def _maybe_compile(self):
        """尝试 torch.compile 加速采样 (自动回退, 加速 ~10-50x)."""
        try:
            self.sample = torch.compile(self.sample)
            return True
        except Exception:
            return False

    # ------------------------------------------------------- MC VMC-Adam 学习
    def learning(self, epochs, lr=0.002, ns=600, grad_clip=1.0, warm_det=500,
                 march_epochs=0, verbose=True):
        """MC VMC-Adam (GCNN 范式): 采样 → E_loc → loss=cov → autograd → grad clip → Adam.
        warm_det>0: 先确定性热身 (精确梯度把状态逼近本征态, 使 E_loc 方差骤降),
        再 MC 微调——绕开随机初始态 E_loc 重尾导致的梯度噪声 (FeI2 §81.35 同款物理).
        返回 (E 历史列表, 末态)."""
        params = list(self.parameters())
        # ---- 阶段0: torch.compile 加速采样 ----
        if verbose:
            print(f'[优化] torch.compile 采样: '
                  f'{"✓" if self._maybe_compile() else "跳过(不可用)"}', flush=True)
        # ---- 阶段1: 确定性热身 (精确梯度) ----
        if warm_det > 0:
            opt_d = torch.optim.Adam(params, lr=0.02)
            for _ in range(warm_det):
                opt_d.zero_grad()
                self.energy_det(grad=True)
                opt_d.step()
            if verbose:
                with torch.no_grad():
                    E_w = self.energy_det(grad=False)
                print(f'[det 热身] E/site = {E_w.real/self.L:+.8f}', flush=True)
        # ---- 阶段2: MC VMC-Adam ----
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for it in range(epochs):
            states = self.sample(ns)
            with torch.no_grad():
                ln0 = self.ln_psi(states)
                E_loc = self.E_loc(states, ln0)
                a = E_loc.real.detach().clamp(-self.eloc_clip, self.eloc_clip)
                b = E_loc.imag.detach().clamp(-self.eloc_clip, self.eloc_clip)
            ln = self.ln_psi(states)                 # 带图
            loss = ((a - a.mean()) * ln.real).sum() + ((b - b.mean()) * ln.imag).sum()
            opt.zero_grad()
            try:
                grads = torch.autograd.grad(loss, params)
            except RuntimeError:
                continue
            tot = math.sqrt(sum(g.detach().square().sum() for g in grads))
            sc = min(1.0, grad_clip / (tot + 1e-8))
            for g, p in zip(grads, params):
                p.grad = (2.0 / ns) * g * sc
            opt.step()
            with torch.no_grad():
                for p in params:
                    p.clamp_(-10, 10)
            hist.append(float(a.mean() / self.L))
            if verbose and (it % 100 == 0 or it == 0 or it == epochs - 1):
                print(f'it {it:5d}  E/site = {a.mean()/self.L:+.6f}  '
                      f'std={E_loc.real.std()/self.L:.4f}', flush=True)
        # ---- 阶段3: MARCH 精修 (前期 Adam + 后期 MARCH, GCNN 最佳记录 §27) ----
        if march_epochs > 0:
            if verbose:
                with torch.no_grad():
                    ln0 = self.ln_psi(states)
                    E_a = self.E_loc(states, ln0)
                print(f'[Adam 完成] E/site = {E_a.real.mean()/self.L:+.6f}', flush=True)
            parameter = torch.zeros_like(self._get_flat())   # GCNN: 动量状态从 0 起
            nu = torch.ones_like(parameter)
            mns = min(ns, 200)                        # MARCH Jacobian 重, 减样本
            for it in range(march_epochs):
                mstates = self.sample(mns)
                parameter_k, nu = self.march_step(mstates, parameter, nu)
                lr_m = 0.01 / (1.0 + max(it - 8000, 0) / 8000)
                # GCNN 公式: 模块参数 p += lr·p_k; 状态 parameter ← p_k
                delta = lr_m * parameter_k
                self._set_flat(self._get_flat() + delta)
                parameter = parameter_k
                if verbose and (it % 50 == 0 or it == march_epochs - 1):
                    with torch.no_grad():
                        ln0 = self.ln_psi(mstates)
                        E_m = self.E_loc(mstates, ln0)
                    e = E_m.real.mean() / self.L
                    hist.append(float(e))
                    print(f'march {it:4d}  E/site = {e:+.6f}  '
                          f'std={E_m.real.std()/self.L:.4f}', flush=True)
        return hist, states

    # ------------------------------------------------------- 自检归一化
    @torch.no_grad()
    def check_normalization(self):
        """Σ_{全构型} P = 1 (自回归归一化自检). L 小才有意义."""
        cfgs = torch.tensor(
            [[1.0 if (i >> k) & 1 else -1.0 for k in range(self.L)]
             for i in range(2 ** self.L)],
            dtype=self.dtype, device=self.device)
        logP = self.ln_psi(cfgs).real * 2            # ΣP = Σexp(logP)
        return torch.exp(logP).sum().item()


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='RNN-VMC Heisenberg 链')
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--mode', choices=['det', 'mc'], default='det')
    ap.add_argument('--ns', type=int, default=600, help='MC 每步样本数')
    ap.add_argument('--warm_det', type=int, default=500,
                    help='MC 前确定性热身步数 (绕开初始 E_loc 重尾); 大 L 自动禁用')
    ap.add_argument('--march_epochs', type=int, default=0,
                    help='Adam 后期 MARCH 精修步数 (GCNN 最佳: 前期Adam+后期March)')
    ap.add_argument('--threads', type=int, default=0,
                    help='torch CPU 线程数 (0=自动检测核数-预留)')
    ap.add_argument('--load', type=str, default='',
                    help='加载已训练 state_dict (如小 L 训练的参数迁移到大 L)')
    ap.add_argument('--save', type=str, default='',
                    help='训练后保存 state_dict (.pt)')
    ap.add_argument('--sz0', action='store_true', default=True)
    ap.add_argument('--no-sz0', dest='sz0', action='store_false')
    ap.add_argument('--mag', type=int, default=0,
                    help='磁化扇区 Σσ (偶数链 0, 奇数链 ±1; 奇数链自动设 +1)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', action='store_true', help='用 CUDA float32 (仅大 L 有用)')
    ap.add_argument('--out', type=str, default='rnn_h', help='输出前缀 (data/)')
    args = ap.parse_args()

    if args.L % 2 == 1 and args.mag == 0:
        args.mag = 1                     # 奇数链基态在 M=+1 扇区
    if args.L % 2 == 1:
        args.sz0 = True                  # 奇数链强制磁化投影
        args.mode = 'mc'                 # det 仅偶数链 Sz0 枚举可行
        print(f'⚠️  L={args.L} 奇数链: 强制 sz0+mag={args.mag}, 用 MC')

    t0 = time.time()
    # 基准: 偶数小 L 用 ED; 否则用 Bethe ansatz (热力学极限, 不用 ED)
    if args.L <= 14 and args.L % 2 == 0:
        E0, E0site = heisenberg_ed(args.L)
        print(f'ED   L={args.L}: E0 = {E0:.8f}, E0/site = {E0site:.8f}')
    else:
        E0 = E0site = 0.25 - math.log(2.0)           # Bethe: E/N = 1/4 - ln2 ≈ -0.443147
        print(f'Bethe L={args.L}: E0/site = {E0site:.8f} (热力学极限, 有限 L 修正 ~1/L)')

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    dtype = torch.float32            # float32 足够 (GCNN 同款); --cuda 时用 GPU
    if args.threads > 0:
        torch.set_num_threads(args.threads)            # 手动指定
        print(f'[线程] 手动指定 {args.threads} 线程')
    else:
        auto_threads()                                 # 自动检测核数
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed, sz0=args.sz0,
                          device=device, dtype=dtype, magnetization=args.mag)
    if args.load:
        sd = torch.load(args.load, map_location=device)
        model.load_state_dict(sd)
        print(f'已加载热身参数: {args.load}')
    print(f'RNN cRNN dh={args.dh} 参数量={model.param_count}  设备={device}')
    if args.L <= 12:
        print(f'自检 ΣP(σ) = {model.check_normalization():.6f} (应 ≈1)')
    if args.L > 14 and args.warm_det > 0:
        print(f'⚠️  L={args.L} 无法确定性枚举, 自动禁用 warm_det (用 --load 热身)')
        args.warm_det = 0

    os.makedirs('data', exist_ok=True)
    hist = []
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.mode == 'det':
        for step in range(1, args.steps + 1):
            opt.zero_grad()
            E = model.energy_det(grad=True)
            opt.step()
            hist.append(float(E.real / args.L))
            if step % 50 == 0 or step == 1:
                print(f'step {step:5d}  E/site = {E.real/args.L:+.8f}  '
                      f'(ED {E0site:+.6f})  [{time.time()-t0:6.1f}s]')
        E_f = hist[-1]
    else:
        hist, states = model.learning(args.steps, lr=args.lr, ns=args.ns,
                                      warm_det=args.warm_det,
                                      march_epochs=args.march_epochs)
        with torch.no_grad():
            ln0 = model.ln_psi(states)
            E_fin = model.E_loc(states, ln0)
        E_f = float(E_fin.real.mean() / args.L)

    print('\n==== 结果 ====')
    print(f'RNN  E/site = {E_f:+.8f}')
    print(f'ED   E/site = {E0site:+.8f}')
    print(f'相对误差      = {abs(E_f-E0site)/abs(E0site)*100:.4f}%')
    np.savez(os.path.join('data', args.out + '.npz'),
             hist=np.array(hist), E0=E0, E0site=E0site,
             L=args.L, dh=args.dh, mode=args.mode)
    if args.save:
        torch.save(model.state_dict(), os.path.join('data', args.save))
        print(f'已保存参数: data/{args.save}')


if __name__ == '__main__':
    main()
