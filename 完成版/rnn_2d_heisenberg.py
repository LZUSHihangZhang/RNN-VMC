#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 RNN 2D Heisenberg: 正方 (J2=0) + 三角 (J2=1) 晶格 —— 自包含版
=============================================================================
参考:
  [1] Hibat-Allah et al., "RNN Wave Functions", PRX 10, 031017 (2020) [arXiv:2002.02973]
  [2] Hibat-Allah, Melko, Carrasquilla, "Supplementing RNN WF with Symmetry and
      Annealing", arXiv:2207.14314 (2D Heisenberg: 10×10 OBC 匹配 QMC 7e-3)
方法: 光栅扫描 1D RNN 铺平 2D 格点; cRNN 单复数头; 磁化配额每步; det/MC/MARCH.

波函数: ψ(σ) = exp(iφ(σ))·√P(σ), P=∏p(σ_i|σ_<i), 单复数输出头 w=(U_re@h+c_re)+i(U_im@h+c_im)
能量:  H = Σ_⟨ij⟩ S_i·S_j (S^z=σ/2)
  E_loc(σ) = (1/4)Σ_⟨ij⟩σ_iσ_j + (1/2)Σ_{⟨ij⟩:σ_i≠σ_j} ψ(σ^ij)/ψ(σ)
晶格: 正方 NN = 横+纵 (PBC); 三角 NN = 横+纵+对角 (PBC).

用法:
  python rnn_2d_heisenberg.py --L 4 --lattice square --mode det
  python rnn_2d_heisenberg.py --L 4 --lattice triangular --mode det
  python rnn_2d_heisenberg.py --L 6 --lattice square --mode mc --steps 4000
=============================================================================
"""
import torch
import math
import argparse
import os
import itertools
import numpy as np


def auto_threads(reserve=2):
    ncpu = os.cpu_count() or 1
    nt = max(1, ncpu - reserve)
    torch.set_num_threads(nt)
    print(f'[线程] CPU {ncpu} 核 → 分配 {nt} 线程 (预留 {reserve})', flush=True)
    return ncpu, nt


auto_threads()


# ----------------------------------------------------------------- 晶格 bonds
def build_bonds(L, lattice):
    """2D 格点 (L×L, PBC), 返回 bond 列表 [(i, j), ...]. i=x*L+y.
    正方: NN = 横(+1,0) + 纵(0,+1). 三角: + 对角(+1,+1)."""
    bonds = set()
    for x in range(L):
        for y in range(L):
            i = x * L + y
            jr = x * L + (y + 1) % L        # 右邻居
            jd = ((x + 1) % L) * L + y      # 下邻居
            bonds.add((min(i, jr), max(i, jr)))
            bonds.add((min(i, jd), max(i, jd)))
            if lattice == 'triangular':
                jd2 = ((x + 1) % L) * L + (y + 1) % L   # 对角邻居
                bonds.add((min(i, jd2), max(i, jd2)))
    return sorted(bonds)


# ----------------------------------------------------------------- ED 参考
def ed_2d(L, lattice, M):
    """精确对角化 2D Heisenberg (PBC), M 扇区最低能 (只建 M 扇区子矩阵, 省内存)."""
    N = L * L
    bonds = build_bonds(L, lattice)
    cfgs = list(enumerate_sector_2d(N, M))
    idx_of = {tuple(c): a for a, c in enumerate(cfgs)}
    nc = len(cfgs)
    H = np.zeros((nc, nc), dtype=complex)
    for a, c in enumerate(cfgs):
        sig = np.array(c)
        for (i, j) in bonds:
            H[a, a] += sig[i] * sig[j] / 4.0
            if sig[i] != sig[j]:
                cp = list(c); cp[i] = -cp[i]; cp[j] = -cp[j]
                H[idx_of[tuple(cp)], a] += 0.5
    return np.linalg.eigvalsh(H)[0]


def enumerate_sector_2d(N, M):
    n_up = (N + M) // 2
    for ups in itertools.combinations(range(N), n_up):
        spins = [-1.0] * N
        for i in ups:
            spins[i] = 1.0
        yield spins




def c4_perms(L):
    """C4 旋转置换 (4×4 正方): perms[g][i] = 旋转 g 后 site i 的新位置."""
    perms = []
    for g in range(4):
        perm = np.zeros(L * L, dtype=int)
        for x in range(L):
            for y in range(L):
                i = x * L + y
                if g == 0: nx, ny = x, y
                elif g == 1: nx, ny = y, L - 1 - x
                elif g == 2: nx, ny = L - 1 - x, L - 1 - y
                else: nx, ny = L - 1 - y, x
                perm[i] = nx * L + ny
        perms.append(perm)
    return perms


# ----------------------------------------------------------------- RNN 波函数
class RNN2DHeisenberg(torch.nn.Module):
    """2D Heisenberg cRNN (光栅扫描 RNN + 2D bonds)."""

    def __init__(self, L, lattice='square', dh=32, seed=None, sz0=True,
                 dtype=torch.float32, magnetization=0):
        super().__init__()
        self.L = L
        self.N = L * L
        self.lattice = lattice
        self.dh = dh
        self.sz0 = sz0
        self.dtype = dtype
        self.magnetization = magnetization
        self.n_up_t = (self.N + magnetization) // 2
        self.n_down_t = (self.N - magnetization) // 2
        self.bonds = build_bonds(L, lattice)
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        def init(*shape):
            return torch.nn.init.uniform_(
                torch.empty(*shape, dtype=self.dtype), -1.0, 1.0, generator=g)

        self.gru = torch.nn.GRUCell(2, dh)          # 融合 GRU (快 ~7x)
        with torch.no_grad():
            self.gru.weight_ih.copy_(init(3 * dh, 2))
            self.gru.weight_hh.copy_(init(3 * dh, dh))
            self.gru.bias_ih.copy_(init(3 * dh))
            self.gru.bias_hh.copy_(init(3 * dh))
        self.U_re = torch.nn.Parameter(init(2, dh))  # 复数输出头实部
        self.c_re = torch.nn.Parameter(init(2))
        self.U_im = torch.nn.Parameter(init(2, dh))  # 虚部
        self.c_im = torch.nn.Parameter(init(2))
        self.h0 = torch.nn.Parameter(init(dh))
        self.register_buffer('onehot', torch.tensor([[1., 0.], [0., 1.]],
                                                    dtype=self.dtype))

    def _quota(self, p, n_up, n_down):
        if self.sz0:
            eps = 1e-10
            q = p.clone()
            q[n_down >= self.n_down_t, 1] = 0.0   # 禁 down
            q[n_up >= self.n_up_t, 0] = 0.0       # 禁 up
            q = q + eps                            # 防 softmax 饱和到 0 → 0/0 nan
            q[n_down >= self.n_down_t, 1] = 0.0   # 禁的分支保持 0
            q[n_up >= self.n_up_t, 0] = 0.0
            return q / q.sum(1, keepdim=True)
        return p

    def ln_psi(self, states):
        """states (B,N) ±1 → lnψ (B,) 复数. 光栅扫描."""
        B, N = states.shape
        idx = (states < 0).to(torch.long)
        onehot = self.onehot[idx]
        h = self.h0.expand(B, -1).clone()
        n_up = torch.zeros(B, dtype=self.dtype)
        n_down = torch.zeros(B, dtype=self.dtype)
        logP = torch.zeros(B, dtype=self.dtype)
        phi = torch.zeros(B, dtype=self.dtype)
        ar = torch.arange(B)
        for i in range(N):
            x = onehot[:, i, :]
            w_re = torch.addmm(self.c_re, h, self.U_re.t())
            w_im = torch.addmm(self.c_im, h, self.U_im.t())
            q = self._quota(torch.softmax(2.0 * w_re, 1), n_up, n_down)
            col = idx[:, i]
            logP = logP + torch.log(q[ar, col])
            phi = phi + w_im[ar, col]
            h = self.gru(x, h)
            n_up = n_up + (col == 0).to(self.dtype)
            n_down = n_down + (col == 1).to(self.dtype)
        return 0.5 * logP + 1j * phi

    def wavefunction(self, states):
        return torch.exp(self.ln_psi(states))

    @torch.no_grad()
    def sample(self, batch):
        """光栅扫描自回归采样 states (B,N) ±1."""
        B, N = batch, self.N
        states = torch.zeros(B, N, dtype=self.dtype)
        h = self.h0.expand(B, -1).clone()
        n_up = torch.zeros(B, dtype=self.dtype)
        n_down = torch.zeros(B, dtype=self.dtype)
        one = torch.ones(B, dtype=self.dtype)
        neg = -one
        for i in range(N):
            w_re = torch.addmm(self.c_re, h, self.U_re.t())
            p = self._quota(torch.softmax(2.0 * w_re, 1), n_up, n_down)
            up = torch.rand(B) < p[:, 0]
            states[:, i] = torch.where(up, one, neg)
            n_up = n_up + up.to(self.dtype)
            n_down = n_down + (~up).to(self.dtype)
            h = self.gru(self.onehot[(~up).to(torch.long)], h)
        return states

    @torch.no_grad()
    def E_loc(self, states, ln_psi_old):
        """2D E_loc: 对每个 bond 找翻转构型批量算 ψ'/ψ (GCNN 范式)."""
        B, N = states.shape
        E = torch.zeros(B, dtype=torch.complex64 if self.dtype == torch.float32
                        else torch.complex128)
        # 对角: 用 bond 列表累加 (1/4)Σ_⟨ij⟩ σ_iσ_j
        diag = torch.zeros(B, dtype=self.dtype)
        for (i, j) in self.bonds:
            diag = diag + states[:, i] * states[:, j]
        E = E + 0.25 * diag
        # 非对角: 所有 bond 一次批量
        bond_i = torch.tensor([b[0] for b in self.bonds])
        bond_j = torch.tensor([b[1] for b in self.bonds])
        mask = states[:, bond_i] != states[:, bond_j]     # (B, nbonds)
        if mask.any():
            cidx, bidx = torch.where(mask)
            ns = states[cidx].clone()
            r = torch.arange(len(cidx))
            ns[r, bond_i[bidx]] = -ns[r, bond_i[bidx]]
            ns[r, bond_j[bidx]] = -ns[r, bond_j[bidx]]
            ln_new = self.ln_psi(ns)
            d = ln_new - ln_psi_old[cidx]
            ratio = torch.exp(torch.complex(
                torch.clamp(d.real, max=20.0), d.imag))
            E.index_add_(0, cidx, 0.5 * ratio)
        return E

    def _build_H(self):
        """Heisenberg H 在 M 扇区构型基 (复杂, 含 bonds)."""
        if getattr(self, '_H', None) is not None:
            return self._H
        cfgs = list(enumerate_sector_2d(self.N, self.magnetization))
        idx_of = {tuple(c): a for a, c in enumerate(cfgs)}
        M = len(cfgs)
        H = torch.zeros((M, M), dtype=torch.complex128)
        for a, c in enumerate(cfgs):
            s = torch.tensor(c, dtype=torch.float64)
            diag = 0.0
            for (i, j) in self.bonds:
                diag += s[i] * s[j] / 4.0
            H[a, a] += diag
            for (i, j) in self.bonds:
                if c[i] != c[j]:
                    cp = list(c); cp[i] = -cp[i]; cp[j] = -cp[j]
                    H[idx_of[tuple(cp)], a] += 0.5
        self._H = H
        return H

    def energy_det(self, grad=False):
        """确定性: 枚举 M 扇区 → ψ†Hψ."""
        cfgs = torch.tensor(list(enumerate_sector_2d(self.N, self.magnetization)),
                            dtype=self.dtype)
        psi = self.wavefunction(cfgs)
        Hc = self._build_H().to(psi.dtype)
        E = (psi.conj() @ Hc @ psi) / (psi.conj() @ psi)
        E = E.detach() if not grad else E
        if grad:
            E.real.backward()
        return E



# ----------------------------------------------------------------- 2D 之字形 RNN
class RNN2DZigzag(RNN2DHeisenberg):
    """2D 之字形 RNN (论文 MDTensorizedGRUcell 的 torch 版):
    每 site 条件于上/左 (偶数行) 或上/右 (奇数行) 邻居的隐态+自旋.
    cell: inputstate_mul = h_cat ⊗ x_cat (外积耦合), 张量化 GRU."""

    def __init__(self, L, lattice='square', dh=32, seed=None, sz0=True,
                 dtype=torch.float32, magnetization=0):
        super().__init__(L, lattice, dh, seed, sz0, dtype, magnetization)
        # 2D 张量化 GRU cell (论文 MDTensorizedGRUcell)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        def init(*shape):
            return torch.nn.init.uniform_(
                torch.empty(*shape, dtype=self.dtype), -1.0, 1.0, generator=g)
        self.W2 = torch.nn.Parameter(init(dh, 2 * dh, 4))    # 主耦合
        self.b2 = torch.nn.Parameter(init(dh))
        self.Wg = torch.nn.Parameter(init(dh, 2 * dh, 4))    # 门耦合
        self.bg = torch.nn.Parameter(init(dh))
        self.Wmerge = torch.nn.Parameter(init(2 * dh, dh))   # 隐态合并
        del self.gru                                   # 光栅版 GRU 不用了

    def _cell2(self, x_side, x_up, h_side, h_up):
        """张量化 2D GRU 一步. x_side,x_up: (B,2); h_side,h_up: (B,dh)."""
        h_cat = torch.cat([h_side, h_up], 1)           # (B, 2dh)
        x_cat = torch.cat([x_side, x_up], 1)           # (B, 4)
        # inputstate_mul = h_cat ⊗ x_cat (B, 2dh, 4)
        ism = h_cat.unsqueeze(2) * x_cat.unsqueeze(1)      # (B, 2dh, 4)
        ism_f = ism.reshape(h_cat.shape[0], -1)            # (B, 2dh*4)
        state_mul = ism_f @ self.W2.reshape(self.dh, -1).T
        state_mulg = ism_f @ self.Wg.reshape(self.dh, -1).T
        u = torch.sigmoid(state_mulg + self.bg)
        h_tilde = torch.tanh(state_mul + self.b2)
        return u * h_tilde + (1 - u) * (h_cat @ self.Wmerge)

    def _neighbors(self, x, y):
        """之字形邻居: 偶数行 (左,上), 奇数行 (右,上). 返回 (side_xy, up_xy, ok)."""
        if y % 2 == 0:
            side = (x - 1, y) if x > 0 else None
        else:
            side = (x + 1, y) if x < self.L - 1 else None
        up = (x, y - 1) if y > 0 else None
        return side, up

    def _scan(self, spins, sample_mode):
        B, N = spins.shape
        L = self.L
        h_grid = torch.zeros(B, L, L, self.dh, dtype=self.dtype)
        n_up = torch.zeros(B, dtype=self.dtype)
        n_down = torch.zeros(B, dtype=self.dtype)
        logP = torch.zeros(B, dtype=self.dtype)
        phi = torch.zeros(B, dtype=self.dtype)
        ar = torch.arange(B)
        z = torch.zeros(B, 2, dtype=self.dtype)
        zh = self.h0.expand(B, -1)                    # 缺失邻居的"真空态" (h0 进图)

        def oh(xx, yy):
            return self.onehot[(spins[:, yy * L + xx] < 0).to(torch.long)]

        for y in range(L):
            xs = range(L) if y % 2 == 0 else range(L - 1, -1, -1)
            for x in xs:
                side, up = self._neighbors(x, y)
                x_side = oh(*side) if side else z
                x_up = oh(*up) if up else z
                h_side = h_grid[:, side[0], side[1]] if side else zh
                h_up = h_grid[:, up[0], up[1]] if up else zh
                h_new = self._cell2(x_side, x_up, h_side, h_up)
                w_re = torch.addmm(self.c_re, h_new, self.U_re.t())
                w_im = torch.addmm(self.c_im, h_new, self.U_im.t())
                q = self._quota(torch.softmax(2.0 * w_re, 1), n_up, n_down)
                if sample_mode:
                    up_bool = torch.rand(B) < q[:, 0]
                    sp = torch.where(up_bool, torch.ones(B, dtype=self.dtype),
                                     -torch.ones(B, dtype=self.dtype))
                    spins[:, y * L + x] = sp
                    n_up = n_up + up_bool.to(self.dtype)
                    n_down = n_down + (~up_bool).to(self.dtype)
                else:
                    col = (spins[:, y * L + x] < 0).to(torch.long)
                    logP = logP + torch.log(q[ar, col])
                    phi = phi + w_im[ar, col]
                    n_up = n_up + (col == 0).to(self.dtype)
                    n_down = n_down + (col == 1).to(self.dtype)
                h_grid[:, x, y] = h_new
        return spins if sample_mode else 0.5 * logP + 1j * phi

    def ln_psi_sym(self, states):
        """C4 对称投影 (论文 log_amplitude_c4vsym, A1 irrep):
        |ψ_sym|²=(1/4)Σ_g|ψ(σ^g)|², 相位=Σ_g e^{iφ_g}. states (B,N)."""
        perms = [torch.tensor(p, dtype=torch.long) for p in c4_perms(self.L)]
        # 4 个旋转构型 (B,4,N)
        rots = torch.stack([states[:, p] for p in perms], 1)
        ln = self.ln_psi(rots.reshape(-1, self.N)).reshape(-1, 4)  # (B,4) 复数
        # 幅度: (1/4)Σ|ψ_g|² = (1/4)Σ exp(2 Re ln)
        log_amp2 = torch.logsumexp(2.0 * ln.real, dim=1) - math.log(4)
        # 相位: angle(Σ e^{iφ_g})
        ph = torch.angle(torch.sum(torch.exp(1j * ln.imag), dim=1))
        return 0.5 * log_amp2 + 1j * ph

    def wavefunction_sym(self, states):
        return torch.exp(self.ln_psi_sym(states))

    @torch.no_grad()
    def E_loc_sym(self, states, ln_sym_old):
        """对称波函数的 E_loc: 翻转比值用 ψ_sym' / ψ_sym."""
        B, N = states.shape
        E = torch.zeros(B, dtype=torch.complex64 if self.dtype == torch.float32
                        else torch.complex128)
        for (i, j) in self.bonds:
            E = E + 0.25 * states[:, i] * states[:, j]
        bond_i = torch.tensor([b[0] for b in self.bonds])
        bond_j = torch.tensor([b[1] for b in self.bonds])
        mask = states[:, bond_i] != states[:, bond_j]
        if mask.any():
            cidx, bidx = torch.where(mask)
            ns = states[cidx].clone()
            r = torch.arange(len(cidx))
            ns[r, bond_i[bidx]] = -ns[r, bond_i[bidx]]
            ns[r, bond_j[bidx]] = -ns[r, bond_j[bidx]]
            ln_new = self.ln_psi_sym(ns)
            d = ln_new - ln_sym_old[cidx]
            ratio = torch.exp(torch.complex(
                torch.clamp(d.real, max=20.0), d.imag))
            E.index_add_(0, cidx, 0.5 * ratio)
        return E

    def learning_sym(self, epochs, lr=2.5e-4, ns=200, T0=2.0, verbose=True):
        """C4 对称 + 退火训练 (论文): cost=cov + 4·T·Var(Re ln_ψ_sym), T 随步数降."""
        params = list(self.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for it in range(epochs):
            T = T0 * (1.0 - it / epochs)                 # 退火温度线性降到 0
            states = self.sample(ns)
            ln_sym = self.ln_psi_sym(states)             # 对称对数振幅 (带图)
            with torch.no_grad():
                El = self.E_loc_sym(states, ln_sym.detach())
                a = El.real.detach().clamp(-50, 50)
                b = El.imag.detach().clamp(-50, 50)
            var_lr = torch.var(ln_sym.real, unbiased=False)
            loss = ((a - a.mean()) * ln_sym.real).sum() + ((b - b.mean()) * ln_sym.imag).sum() \
                   + 4.0 * T * ns * var_lr          # 退火项 (论文 cost 公式)
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
            hist.append(float(a.mean() / self.N))
            if verbose and (it % 200 == 0 or it == epochs - 1):
                print(f'  it {it}: E/site={a.mean()/self.N:+.6f} T={T:.3f}', flush=True)
        return hist, states

    def sample(self, batch):
        spins = torch.zeros(batch, self.N, dtype=self.dtype)
        return self._scan(spins, True)

    def ln_psi(self, states):
        return self._scan(states.clone(), False)


    def learning(self, epochs, lr=2.5e-4, ns=600, march_epochs=0, verbose=True):
        params = list(self.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        hist = []
        for it in range(epochs):
            states = self.sample(ns)
            with torch.no_grad():
                E_loc = self.E_loc(states, self.ln_psi(states))
                a = E_loc.real.detach().clamp(-50, 50)
                b = E_loc.imag.detach().clamp(-50, 50)
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
            hist.append(float(a.mean() / self.N))
            if verbose and (it % 200 == 0 or it == epochs - 1):
                print(f'  it {it}: E/site={a.mean()/self.N:+.6f}', flush=True)
        return hist, states


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='RNN 2D Heisenberg (正方/三角)')
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--lattice', choices=['square', 'triangular'], default='square')
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--arch', choices=['raster', 'zigzag'], default='raster',
                    help='RNN 架构: raster=光栅扫描 1D, zigzag=2D 之字形 (论文式)')
    ap.add_argument('--mode', choices=['det', 'mc'], default='det')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--ns', type=int, default=600)
    ap.add_argument('--march_epochs', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ed0', type=float, default=None,
                    help='已知 ED 基态能 (跳过慢的对角化)')
    args = ap.parse_args()

    N = args.L * args.L
    M = 0 if N % 2 == 0 else 1          # 偶数格点 M=0; 奇数格点 M=±1 (基态扇区)
    print(f'==== RNN 2D Heisenberg: {args.L}×{args.L} {args.lattice} '
          f'({N} sites, M={M}), 模式={args.mode}, dh={args.dh} ====')
    print(f'bonds 数: {len(build_bonds(args.L, args.lattice))}')

    if args.ed0 is not None:
        E_ed = args.ed0
        print(f'ED 参考 (已知): E0={E_ed:.6f}  E/site={E_ed/N:.6f}')
    elif args.L <= 5:
        E_ed = ed_2d(args.L, args.lattice, M)
        print(f'ED 参考: E0={E_ed:.6f}  E/site={E_ed/N:.6f}')
    else:
        E_ed = None
        print('L>5 ED 不可行 (2^(L²) 太大), 用 MC')

    cls = RNN2DZigzag if args.arch == 'zigzag' else RNN2DHeisenberg
    m = cls(args.L, args.lattice, args.dh, seed=args.seed, magnetization=M)
    if args.mode == 'det':
        opt = torch.optim.Adam(m.parameters(), lr=0.02)
        for it in range(args.steps):
            opt.zero_grad()
            E = m.energy_det(grad=True)
            opt.step()
            if it % 400 == 0:
                print(f'det it {it}: E={E.real.item():.6f}', flush=True)
        E = m.energy_det(grad=False).real.item()
    else:
        hist, _ = m.learning(args.steps, ns=args.ns, march_epochs=args.march_epochs)
        E = hist[-1] * N

    print(f'\nRNN E0={E:.6f}  E/site={E/N:.6f}')
    if E_ed is not None:
        print(f'ED   E0={E_ed:.6f}  E/site={E_ed/N:.6f}  '
              f'(差 {abs(E-E_ed):.5f}, {abs(E-E_ed)/abs(E_ed)*100:.3f}%)')
    os.makedirs('data', exist_ok=True)
    np.savez(f'data/rnn_2d_{args.lattice}_L{args.L}.npz', E=E, E_ed=E_ed, L=args.L)


if __name__ == '__main__':
    main()
