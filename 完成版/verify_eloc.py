#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小验证脚本 (NQS 范式: 先验证再训练)。

1) RNN 自回归归一化 ΣP = 1 (全构型)
2) 批量 E_loc 正确性: E_loc(batch) vs 暴力逐构型 E_loc
3) 确定性 E_det = ψ†Hψ/ψ†ψ vs 显式 ψ 向量 + H 矩阵
4) 梯度正确性: energy_det 分析梯度 vs 有限差分
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_adam_march import RNNHeisenberg, enumerate_sz0

L = 10
torch.manual_seed(1)
model = RNNHeisenberg(L, dh=16, seed=1)
DT = model.dtype                      # 用模型 dtype (默认 float32)

# ---------- 1) 归一化 ----------
norm = model.check_normalization()
print(f'1) Σ_{{all}} P(σ) = {norm:.8f}   (应 ≈ 1)')
assert abs(norm - 1.0) < 1e-5, 'normalization failed'

# ---------- 2) 批量 E_loc vs 暴力逐构型 ----------
def eloc_brute(states):
    """逐构型 E_loc (暴力, 前缀不复用)。states: (B,L) ±1"""
    B, L = states.shape
    out = []
    for b in range(B):
        sp = states[b]
        ln0 = model.ln_psi(sp.unsqueeze(0))[0]         # 复数 0.5logP + iφ
        E = 0.25 * torch.dot(sp, torch.roll(sp, 1))
        for i in range(L):
            j = (i + 1) % L
            if sp[i] != sp[j]:
                ns = sp.clone(); ns[i] = -ns[i]; ns[j] = -ns[j]
                ln1 = model.ln_psi(ns.unsqueeze(0))[0]
                E = E + 0.5 * torch.exp(ln1 - ln0)
        out.append(E)
    return torch.stack(out)

with torch.no_grad():
    sps = torch.tensor(list(enumerate_sz0(L))[:40], dtype=DT)
    ln0 = model.ln_psi(sps)
    E_batch = model.E_loc(sps, ln0)
    E_brute = eloc_brute(sps)
d = (E_batch - E_brute).abs().max().item()
print(f'2) E_loc 批量 vs 暴力: maxdiff = {d:.2e}')
assert d < (1e-4 if DT == torch.float32 else 1e-9), 'E_loc batch mismatch'

# ---------- 3) E_det (ψ†Hψ) vs 显式 H 矩阵 ----------
cfgs = list(enumerate_sz0(L))
idx_of = {tuple(c): a for a, c in enumerate(cfgs)}
M = len(cfgs)
with torch.no_grad():
    psi = model.wavefunction(torch.tensor(cfgs, dtype=DT)).numpy()
H = np.zeros((M, M), dtype=complex)
for a, c in enumerate(cfgs):
    s = np.array(c)
    H[a, a] += 0.25 * np.dot(s, np.roll(s, 1))
    for i in range(L):
        j = (i + 1) % L
        if s[i] != s[j]:
            cp = list(c); cp[i] = -cp[i]; cp[j] = -cp[j]
            H[idx_of[tuple(cp)], a] += 0.5
E_matrix = (psi.conj() @ H @ psi) / (psi.conj() @ psi)
E_det = model.energy_det(grad=False)
dd = abs(E_matrix.real / L - E_det.real / L)
print(f'3) E_matrix/site = {E_matrix.real/L:+.8f}   E_det/site = {E_det.real/L:+.8f}   diff = {dd:.2e}')
assert dd < 1e-5, 'E_det mismatch'

# ---------- 4) 梯度 vs 有限差分 ----------
model.zero_grad()
E = model.energy_det(grad=True)
g_ana = {n: p.grad.clone() for n, p in model.named_parameters()}
eps = 1e-3 if DT == torch.float32 else 1e-6   # float32 有限差分步长需更大 (FD 噪声 ~1e-3)
print('4) 梯度对比 (前几个元素)')
for name, p in list(model.named_parameters())[:4]:
    flat = p.data.view(-1)
    g_fd = torch.zeros_like(flat)
    for k in range(min(6, flat.numel())):
        p0 = flat[k].item()
        flat[k] = p0 + eps
        Ep = model.energy_det(grad=False).real
        flat[k] = p0 - eps
        Em = model.energy_det(grad=False).real
        flat[k] = p0
        g_fd[k] = (Ep - Em) / (2 * eps)
    ga = g_ana[name].view(-1)[:6]
    print(f'   {name:6s} ana={ga.detach().numpy()} fd={g_fd[:6].numpy()}')
    err = (g_ana[name].view(-1)[:6] - g_fd[:6]).abs().max().item()
    assert err < (5e-3 if DT == torch.float32 else 1e-5), \
        f'grad mismatch for {name}: {err}'

print('\n✅ 全部验证通过: 归一化 / E_loc / E_det / 梯度')
