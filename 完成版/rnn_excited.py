#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 RNN 激发态算法 —— 自旋 1/2 反铁磁 Heisenberg 链 单磁振子 (Sz=1) —— 完整自包含版
=============================================================================
本文件完全独立, 不依赖任何其他代码 (仅 torch/numpy).

物理: Heisenberg 链自旋守恒. 基态 M=0; 最低激发 (单磁振子) 在 M=2 扇区.
      gap = E(M=2) - E(M=0)

波函数 (cRNN, 单复数头, 振幅相位同时学习):
    ψ(σ) = exp(iφ(σ)) · √P(σ) · 1[Σσ=M]       (磁化扇区投影)
    P(σ) = ∏_i p(σ_i|σ_<i)  —— GRU 隐态, 复数输出头 w=(U_re@h+c_re)+i(U_im@h+c_im)
    幅度 p=softmax(2·Re w), 相位 φ=Im w

能量 (S^z=σ/2):  H = Σ_i[ S^z_i S^z_{i+1} + (1/2)(S^+_i S^-_{i+1}+S^-_i S^+_{i+1}) ]
    E_loc(σ) = (1/4)Σσ_iσ_{i+1} + (1/2)Σ_{i:σ_i≠σ_{i+1}} ψ(σ^i)/ψ(σ)
    扭曲边界 (动量 k=θ): wrap bond 系数 0.5·e^{∓iθ}

两种求解模式:
  --mode det   确定性: 枚举磁化扇区全构型, E=ψ†Hψ/ψ†ψ (精确变分, 对齐 ED ~0.06%)
  --mode mc    MC: 自回归采样 + Adam 前期 + MARCH 精修 (可扩展大 L)

MARCH 优化器 (SR 自然梯度, 参考本方 GCNN 最佳记录):
    O=concat[∂Re,∂Im] 中心化; M=OOᵀ+λI; δθ=OᵀM⁻¹(E_aug−O·μ·p); nu=β·nu+(δθ−δθ_prev)²

参考: arXiv:2002.02973 (Hibat-Allah PRX); 本方 .md §81.39-43 扭曲张量激发态方法.

用法:
  python rnn_excited.py --L 10 --mode det                       # 精确对齐 ED
  python rnn_excited.py --L 10 --mode mc --steps 8000 --march_epochs 120  # MC
  python rnn_excited.py --L 10 --mode det --thetas 0 1.57 3.14 # 色散 ε(k)
=============================================================================
"""
import torch
import math
import argparse
import os
import itertools
import numpy as np


def auto_threads(reserve=2):
    """自动检测 CPU 核数并分配 torch 线程.
    reserve: 预留核数 (给 OS/其他任务, 避免过度订阅).
    返回 (总核数, 分配线程数)."""
    ncpu = os.cpu_count() or 1
    nt = max(1, ncpu - reserve)
    torch.set_num_threads(nt)
    print(f'[线程] CPU {ncpu} 核 → 分配 {nt} 线程 (预留 {reserve})', flush=True)
    return ncpu, nt


auto_threads()                     # 模块加载时自动分配


# =============================================================================
# 一、ED 参考 (精确对角化, 用于验证 RNN 结果; 计算本身不用 ED)
# =============================================================================
def ed_twisted_sector(L, M, theta):
    """精确对角化扭曲 Heisenberg (PBC), M 扇区最低能 (total).
    θ=0 普通链; θ≠0 扭曲边界 (动量 k=θ)."""
    N = 2 ** L
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        s = np.array([(i >> k) & 1 for k in range(L)])
        sig = 1 - 2 * s
        if sig.sum() != M:
            continue
        for j in range(L):
            j1 = (j + 1) % L
            H[i, i] += sig[j] * sig[j1] / 4.0
            if sig[j] != sig[j1]:
                t = s.copy(); t[j] ^= 1; t[j1] ^= 1
                k = 0
                for b in range(L):
                    k |= t[b] << b
                coef = 0.5
                if j == L - 1:                       # wrap bond 扭曲
                    coef = 0.5 * np.exp(-1j * theta * sig[j])
                H[i, k] += coef
    idx = [i for i in range(N)
           if (1 - 2 * np.array([(i >> b) & 1 for b in range(L)])).sum() == M]
    return np.linalg.eigvalsh(H[np.ix_(idx, idx)])[0]


# =============================================================================
# 二、磁化扇区枚举 (确定性模式用)
# =============================================================================
def enumerate_sector(L, M):
    """枚举磁化 M=Σσ 扇区全部构型 (M=2: (L+M)/2 个 up)."""
    n_up = (L + M) // 2
    for ups in itertools.combinations(range(L), n_up):
        spins = [-1.0] * L
        for i in ups:
            spins[i] = 1.0
        yield spins


# =============================================================================
# 三、RNN 波函数类 (完整: cRNN 单复数头 + 磁化扇区 + 扭曲 + det/MC/MARCH)
# =============================================================================
class RNNHeisenberg(torch.nn.Module):
    """cRNN 波函数 (GRU 隐态 + 单复数输出头). 批量 API."""

    def __init__(self, L, dh=32, seed=None, sz0=True, device='cpu',
                 dtype=torch.float32, ratio_clamp=20.0, eloc_clip=50.0,
                 magnetization=0, twist=0.0):
        super().__init__()
        self.L, self.dh, self.dv = L, dh, 2
        self.sz0 = sz0
        self.device = torch.device(device)
        self.dtype = dtype
        self.ratio_clamp = ratio_clamp   # E_loc 比值实部钳制 (防 exp 溢出)
        self.eloc_clip = eloc_clip       # 学习损失中 E_loc 值钳制 (稳健优化)
        # 磁化扇区: Σσ = magnetization (基态 M=0; 单磁振子 M=2)
        self.magnetization = magnetization
        self.n_up_t = (L + magnetization) // 2
        self.n_down_t = (L - magnetization) // 2
        self.twist = twist               # 扭曲边界相位 θ (动量 k=θ)

        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)

        def init(*shape):
            return torch.nn.init.uniform_(
                torch.empty(*shape, dtype=self.dtype, device=self.device),
                -1.0, 1.0, generator=g)

        # GRU (torch.nn.GRUCell 融合 C++ 算子, 比手动 addmm 快 ~7x)
        self.gru = torch.nn.GRUCell(2, dh)
        with torch.no_grad():
            self.gru.weight_ih.copy_(init(3 * dh, 2))
            self.gru.weight_hh.copy_(init(3 * dh, dh))
            self.gru.bias_ih.copy_(init(3 * dh))
            self.gru.bias_hh.copy_(init(3 * dh))
        # 单一复数输出头: w = (U_re@h+c_re) + i(U_im@h+c_im)
        self.U_re = torch.nn.Parameter(init(2, dh))
        self.c_re = torch.nn.Parameter(init(2))
        self.U_im = torch.nn.Parameter(init(2, dh))
        self.c_im = torch.nn.Parameter(init(2))
        self.h0 = torch.nn.Parameter(init(dh))
        # one-hot: σ=+1 -> [1,0], σ=-1 -> [0,1]
        self.register_buffer('onehot',
                             torch.tensor([[1., 0.], [0., 1.]],
                                          dtype=self.dtype, device=self.device))
        self.param_count = sum(p.numel() for p in self.parameters())

    # --------------------------------------------------- 批量前向: σ → lnψ
    def _gru(self, h, x):
        """一步 GRU (torch.nn.GRUCell 融合 C++ 算子, ~7x 快)."""
        return self.gru(x, h)

    def ln_psi(self, states):
        """states (B,L) ±1 → lnψ (B,) 复数 (带图). 自回归顺序: 预测 σ_i 于
        h(编码 σ_<i) 再更新 h (保证 ΣP=1). 磁化配额: i≥L/2 起, 每构型独立计数."""
        B, L = states.shape
        idx = (states < 0).to(torch.long)            # 0:+1(up), 1:-1(down)
        onehot = self.onehot[idx]
        h = self.h0.expand(B, -1).clone()
        n_up = torch.zeros(B, dtype=self.dtype, device=self.device)
        n_down = torch.zeros(B, dtype=self.dtype, device=self.device)
        half = L // 2
        logP = torch.zeros(B, dtype=self.dtype, device=self.device)
        phi = torch.zeros(B, dtype=self.dtype, device=self.device)
        ar = torch.arange(B, device=self.device)
        for i in range(L):
            x = onehot[:, i, :]
            w_re = torch.addmm(self.c_re, h, self.U_re.t())
            w_im = torch.addmm(self.c_im, h, self.U_im.t())
            y1 = torch.softmax(2.0 * w_re, dim=1)    # 幅度 p=softmax(2Re w)
            q = self._quota(y1, n_up, n_down)        # 磁化配额 (每步, 论文式)
            logP = logP + torch.log(q[ar, idx[:, i]])
            phi = phi + w_im[ar, idx[:, i]]          # 相位 φ=Im w
            h = self._gru(h, x)                      # 更新 h (addmm 融合)
            n_up = n_up + (idx[:, i] == 0).to(self.dtype)
            n_down = n_down + (idx[:, i] == 1).to(self.dtype)
        return 0.5 * logP + 1j * phi

    def _quota(self, p, n_up, n_down):
        """磁化配额 (论文每个 site 都施加): 超配额分支清零并重归一化."""
        if self.sz0:
            p = p.clone()
            p[n_down >= self.n_down_t, 1] = 0.0
            p[n_up >= self.n_up_t, 0] = 0.0
            p = p / p.sum(1, keepdim=True)
        return p

    def wavefunction(self, states):
        return torch.exp(self.ln_psi(states))

    # --------------------------------------------------- 批量局域能量 (扭曲)
    @torch.no_grad()
    def E_loc(self, states, ln_psi_old):
        """states (B,L) ±1, ln_psi_old (B,) → E_loc (B,) 复数.
        全 bond 一次批量 (GCNN 范式); wrap bond 带扭曲相位 e^{∓iθ}."""
        B, L = states.shape
        E = torch.zeros(B, dtype=torch.complex64 if self.dtype == torch.float32
                        else torch.complex128, device=self.device)
        E = E + 0.25 * torch.sum(states * torch.roll(states, 1, dims=1), dim=1)
        # 非 wrap bonds (i, i+1), i=0..L-2
        si, sj = states[:, :-1], states[:, 1:]
        mask = (si != sj)
        if mask.any():
            cidx, bidx = torch.where(mask)
            ns = states[cidx].clone()
            r = torch.arange(len(cidx))
            ns[r, bidx] = -ns[r, bidx]
            ns[r, bidx + 1] = -ns[r, bidx + 1]
            ln_new = self.ln_psi(ns)
            d = ln_new - ln_psi_old[cidx]
            ratio = torch.exp(torch.complex(
                torch.clamp(d.real, max=self.ratio_clamp), d.imag))
            E.index_add_(0, cidx, 0.5 * ratio)
        # wrap bond (L-1, 0): 扭曲相位
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
                # σ_{L-1}=+1(|↑↓⟩)->e^{-iθ}, -1->e^{+iθ}
                phase = torch.exp(-1j * self.twist * states[w, L - 1])
                E.index_add_(0, w, 0.5 * phase * ratio)
            else:
                E.index_add_(0, w, 0.5 * ratio)
        return E

    # --------------------------------------------------- 批量自回归采样
    @torch.no_grad()
    def sample(self, batch):
        """批量自回归采样 states (B,L) ±1 (与 ln_psi 同序同条件)."""
        B, L = batch, self.L
        states = torch.zeros(B, L, dtype=self.dtype, device=self.device)
        h = self.h0.expand(B, -1).clone()
        n_up = torch.zeros(B, dtype=self.dtype, device=self.device)
        n_down = torch.zeros(B, dtype=self.dtype, device=self.device)
        half = L // 2
        one = torch.ones(B, dtype=self.dtype, device=self.device)
        neg = -one
        for i in range(L):
            p = torch.softmax(2.0 * torch.addmm(self.c_re, h, self.U_re.t()),
                              dim=1)
            p = self._quota(p, n_up, n_down)           # 磁化配额 (每步, 论文式)
            up = torch.rand(B, device=self.device) < p[:, 0]
            states[:, i] = torch.where(up, one, neg)
            n_up = n_up + up.to(self.dtype)
            n_down = n_down + (~up).to(self.dtype)
            h = self._gru(h, self.onehot[(~up).to(torch.long)])
        return states

    # --------------------------------------------------- 确定性 H 矩阵 (扭曲)
    def _build_H(self):
        """Heisenberg H 在磁化扇区构型基下 (M,M), 含扭曲相位, 缓存."""
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
        """确定性: 枚举磁化扇区 → ψ 向量 → E=ψ†Hψ/ψ†ψ (精确变分)."""
        cfgs = torch.tensor(list(enumerate_sector(self.L, self.magnetization)),
                            dtype=self.dtype, device=self.device)
        psi = self.wavefunction(cfgs)
        Hc = self._build_H().to(psi.dtype)
        E = (psi.conj() @ Hc @ psi) / (psi.conj() @ psi)
        E = E.detach() if not grad else E
        if grad:
            E.real.backward()
        return E

    # --------------------------------------------------- MARCH 自然梯度
    def _get_flat(self):
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def _set_flat(self, flat):
        idx = 0
        for p in self.parameters():
            p.data.copy_(flat[idx:idx + p.numel()].reshape(p.shape))
            idx += p.numel()

    def march_step(self, sps, parameter, nu, tau=0.05, mu=0.95, lam=0.001,
                   beta0=0.995):
        """一步 MARCH: O=concat[∂Re,∂Im] 中心化; M=OOᵀ+λI;
        δθ=OᵀM⁻¹(E_aug−O·μ·p); nu=β·nu+(δθ−δθ_prev)². 返回 (p_k, nu)."""
        ns = sps.shape[0]
        params = list(self.parameters())
        nparam = parameter.numel()
        O_re = torch.zeros((ns, nparam), dtype=self.dtype, device=self.device)
        O_im = torch.zeros((ns, nparam), dtype=self.dtype, device=self.device)
        for s in range(ns):
            ln = self.ln_psi(sps[s:s + 1])[0]
            gr = torch.autograd.grad(ln.real, params, retain_graph=True)
            gi = torch.autograd.grad(ln.imag, params)
            O_re[s] = torch.cat([g.flatten() for g in gr])
            O_im[s] = torch.cat([g.flatten() for g in gi])
        O = torch.cat([O_re, O_im], dim=0)
        O = (O - O.mean(dim=0)) / math.sqrt(ns)
        with torch.no_grad():
            E = self.E_loc(sps, self.ln_psi(sps))
            E_aug = torch.cat([(-tau * (E - E.mean())).real,
                               (-tau * (E - E.mean())).imag])
        M = O @ O.T + lam * torch.eye(2 * ns, dtype=self.dtype,
                                      device=self.device)
        rhs = E_aug - O @ (mu * parameter)
        x = None
        for extra in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                Lm = torch.linalg.cholesky(
                    M + extra * torch.eye(2 * ns, dtype=self.dtype,
                                          device=self.device))
                x = torch.cholesky_solve(rhs.reshape(2 * ns, 1), Lm).reshape(2 * ns)
                break
            except RuntimeError:
                continue
        if x is None:
            return parameter, nu
        temp = torch.mv(O.T, x)
        p_k = temp / torch.sqrt(nu) + mu * parameter
        nu = (beta0 * nu + (p_k - parameter) ** 2).clamp(min=1e-6, max=1e6)
        return p_k.detach(), nu

    # --------------------------------------------------- MC 学习 (Adam + MARCH)
    def learning(self, epochs, lr=2.5e-4, ns=600, march_epochs=0, verbose=True):
        """MC: 阶段1 Adam (采样→E_loc→loss=cov→grad clip→Adam);
        阶段2 MARCH 精修. 返回 (每步 E/site 历史, 末态)."""
        try:                                          # torch.compile 加速采样
            self.sample = torch.compile(self.sample)
        except Exception:
            pass
        params = list(self.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for it in range(epochs):
            states = self.sample(ns)
            with torch.no_grad():
                E_loc = self.E_loc(states, self.ln_psi(states))
                a = E_loc.real.detach().clamp(-self.eloc_clip, self.eloc_clip)
                b = E_loc.imag.detach().clamp(-self.eloc_clip, self.eloc_clip)
            ln = self.ln_psi(states)
            loss = ((a - a.mean()) * ln.real).sum() + ((b - b.mean()) * ln.imag).sum()
            opt.zero_grad()
            try:
                grads = torch.autograd.grad(loss, params)
            except RuntimeError:
                continue
            tot = math.sqrt(sum(g.detach().square().sum() for g in grads))
            sc = min(1.0, 1.0 / (tot + 1e-8))
            for g, p in zip(grads, params):
                p.grad = g * sc
            opt.step()
            for p in params:
                p.data.clamp_(-10, 10)
            hist.append(float(a.mean() / self.L))
            if verbose and (it % 100 == 0 or it == epochs - 1):
                print(f'  Adam it {it:5d}: E/site={a.mean()/self.L:+.6f} '
                      f'std={E_loc.real.std()/self.L:.4f}', flush=True)
        if march_epochs > 0:
            parameter = torch.zeros_like(self._get_flat())   # 动量状态从 0 起
            nu = torch.ones_like(parameter)
            mns = min(ns, 200)
            for it in range(march_epochs):
                mstates = self.sample(mns)
                p_k, nu = self.march_step(mstates, parameter, nu)
                lr_m = 0.01 / (1.0 + max(it - 8000, 0) / 8000)
                self._set_flat(self._get_flat() + lr_m * p_k)   # p += lr·p_k
                parameter = p_k
                if verbose and (it % 50 == 0 or it == march_epochs - 1):
                    with torch.no_grad():
                        E_m = self.E_loc(mstates, self.ln_psi(mstates))
                    e = E_m.real.mean() / self.L
                    hist.append(float(e))
                    print(f'  MARCH it {it:4d}: E/site={e:+.6f}', flush=True)
        return hist, states


# =============================================================================
# 四、求解器 (det / MC) —— 返回 total 能量 (与 ED 单位一致)
# =============================================================================
def solve_det(model, steps, lr=0.02):
    """确定性: 枚举磁化扇区 → ψ†Hψ 精确变分下限."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(steps):
        opt.zero_grad()
        E = model.energy_det(grad=True)
        opt.step()
        if it % 500 == 0:
            print(f'  det it {it}: E={E.real.item():+.6f}', flush=True)
    return model.energy_det(grad=False).real.item()


def solve_mc(model, steps, lr=2.5e-4, ns=600, march=0):
    """MC: Adam 前期 + MARCH 精修."""
    hist, _ = model.learning(epochs=steps, lr=lr, ns=ns, march_epochs=march,
                             verbose=False)
    return hist[-1] * model.L                       # per-site -> total


# =============================================================================
# 五、主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description='RNN 激发态 (Heisenberg 单磁振子)')
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--mag', type=int, default=2, help='激发态扇区 (Sz=1 -> M=2)')
    ap.add_argument('--mode', choices=['det', 'mc'], default='det')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--ns', type=int, default=600)
    ap.add_argument('--march_epochs', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0,
                    help='torch 线程数 (0=自动检测核数-预留)')
    ap.add_argument('--thetas', type=float, nargs='*',
                    default=[0.0, 0.63, 1.26, 1.88, 2.51, 3.14])
    ap.add_argument('--out', type=str, default='rnn_exc')
    args = ap.parse_args()
    if args.threads > 0:                          # 手动覆盖自动检测
        torch.set_num_threads(args.threads)
        print(f'[线程] 手动指定 {args.threads} 线程', flush=True)

    print(f'==== RNN 激发态: L={args.L}, M={args.mag} (Sz={args.mag//2}), '
          f'模式={args.mode}, dh={args.dh} ====')
    ncfg = len(list(enumerate_sector(args.L, args.mag)))
    print(f'M={args.mag} 扇区构型数 = {ncfg}')

    E0_ed = ed_twisted_sector(args.L, 0, 0.0)
    Ee_ed = ed_twisted_sector(args.L, args.mag, 0.0)
    print(f'\nED 参考: E(M=0)={E0_ed:.6f}  E(M={args.mag})={Ee_ed:.6f}  '
          f'gap={Ee_ed-E0_ed:.6f}')

    def run(mag, label):
        m = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed, sz0=True,
                          magnetization=mag)
        E = solve_det(m, args.steps) if args.mode == 'det' else \
            solve_mc(m, args.steps, args.lr, args.ns, args.march_epochs)
        Eed = ed_twisted_sector(args.L, mag, 0.0)
        print(f'{label}: RNN {E:+.6f}  ED {Eed:.6f}  '
              f'(差 {abs(E-Eed):.5f}, {abs(E-Eed)/abs(Eed)*100:.3f}%)')
        return E

    E0 = run(0, f'M=0 (基态)  ')
    Ee = run(args.mag, f'M={args.mag} (激发)')
    gap_rnn, gap_ed = Ee - E0, Ee_ed - E0_ed
    print(f'\ngap: RNN {gap_rnn:.6f}  ED {gap_ed:.6f}  '
          f'(差 {abs(gap_rnn-gap_ed):.5f}, '
          f'{abs(gap_rnn-gap_ed)/abs(gap_ed)*100:.2f}%)')

    # ---- 色散 ε(k): 扭曲边界 θ=k, M=mag ----
    print(f'\n==== 单磁振子色散 (扭曲边界, M={args.mag}) ====')
    print(f'{"θ":>6} {"k/π":>7} | {"E_θ RNN":>10} {"E_θ ED":>10} '
          f'{"gap RNN":>8} {"gap ED":>8}')
    disp = []
    for th in args.thetas:
        Eed = ed_twisted_sector(args.L, args.mag, th)
        m = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed, sz0=True,
                          magnetization=args.mag, twist=th)
        E = solve_det(m, max(args.steps // 2, 800)) if args.mode == 'det' else \
            solve_mc(m, max(args.steps // 2, 2000), args.lr, args.ns,
                     args.march_epochs)
        gap_r, gap_e = E - E0, Eed - E0_ed
        disp.append((th, E, Eed, gap_r, gap_e))
        print(f'{float(th):6.3f} {float(th)/math.pi:7.3f} | {E:10.6f} '
              f'{Eed:10.6f} {gap_r:8.5f} {gap_e:8.5f}')
    print('\n(色散: k→0 gap→0 Goldstone, k=π gap 最大, 符合 Heisenberg 链物理)')

    os.makedirs('data', exist_ok=True)
    np.savez(os.path.join('data', args.out + '.npz'),
             L=args.L, mag=args.mag, mode=args.mode,
             E0_ed=E0_ed, Ee_ed=Ee_ed, E0=E0, Ee=Ee,
             gap_rnn=gap_rnn, gap_ed=gap_ed,
             thetas=np.array([d[0] for d in disp]),
             disp=np.array([[d[1], d[2], d[3], d[4]] for d in disp]))
    print(f'\n已保存: data/{args.out}.npz')


if __name__ == '__main__':
    main()
