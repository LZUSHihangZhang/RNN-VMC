#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN-VMC: minSR (论文 arXiv:2608.18065) vs Adam 对比 (自旋 1/2 Heisenberg 链 PBC)。
=================================================================================
复现: Attar, Aboussalah, Hibat-Allah, "QGT Preconditioning for Stable Training
      of RNN-QS", arXiv:2608.18065 (2026). 官方 repo (werro40/RNN-Natural-Gradients)
      标注 "Under construction!" 且不可直接运行 (缺 Models/Hamiltonians/init-params,
      get_minSR_gradients 的 lmbda 未定义, 2D 类 jax_dtype 未定义) —— 故在此按论文
      Eq.(minsrR) 用 torch 干净重写, 挂在已有的 RNNHeisenberg 波函数上。

方法 (论文 sample×sample minSR), 复数 cRNN 用实表示:
  O_ki   = (1/√Ns)·∂_θk logψ   (中心化 Ō)
  Ō'     = (Re Ō ; Im Ō)      (Np × 2Ns)
  ε̄_i   = (1/√Ns)(ε_i − E)^* ;  ε̄' = 2·(Re ε̄ ; −Im ε̄)   (2Ns,)
  δθ     = η·Ō'·(T + λI)^{-1}·ε̄',   T = Ō'ᵀŌ'           (2Ns×2Ns, 样本空间)
  动量    m_{t+1} = μ·m_t + (1−μ)·δθ_t ;  θ_{t+1} = θ_t − m_{t+1}

对比基准 (同一 RNNHeisenberg, 同一初始化):
  [Adam] 你的 MC VMC-Adam (rnn_heisenberg.learning, warm_det=0)
  [minSR] 本文写的最小侵入实现
两者都不做"精确能量 det 热身", 直接随机初始 → 直接检验论文"minSR 比 Adam 稳定"的主张。

用法:
  python3 rnn_minsr.py --L 10 --dh 32 --ns 200 --steps 400
"""
import torch, math, argparse, time
import torch.autograd
from torch import autograd
from rnn_heisenberg import RNNHeisenberg, heisenberg_ed, auto_threads

DT = torch.float64


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_flat(params, flat):
    i = 0
    for p in params:
        p.data.copy_(flat[i:i + p.numel()].reshape(p.shape))
        i += p.numel()


def per_sample_jacobian(model, samples):
    """复数 log-derivative O = (∂Re, ∂Im), (Ns, Np) 各一张. (复用 march_step 思路)"""
    ns = samples.shape[0]
    params = list(model.parameters())
    nparam = sum(p.numel() for p in params)
    ln = model.ln_psi(samples)                       # (Ns,) 复数, 带图
    O_re = torch.zeros((ns, nparam), dtype=DT)
    O_im = torch.zeros((ns, nparam), dtype=DT)
    for s in range(ns):
        gr = autograd.grad(ln[s].real, params, retain_graph=True, allow_unused=True)
        gi = autograd.grad(ln[s].imag, params, retain_graph=True, allow_unused=True)
        O_re[s] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gr, params)])
        O_im[s] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gi, params)])
    return O_re, O_im   # (Ns,Np)


def minsr_step(model, samples, lr, lam, momentum, m_buf, trust_region=False):
    """论文 minSR 一步, 返回 (互参平坦向量 δθ, E_loc 均值, E_loc 向量)。"""
    ns = samples.shape[0]
    params = list(model.parameters())
    O_re, O_im = per_sample_jacobian(model, samples)   # (Ns,Np)
    # 中心化 over samples
    Oc_re = O_re - O_re.mean(0, keepdim=True)
    Oc_im = O_im - O_im.mean(0, keepdim=True)
    # 能量 & 中心化 ε̄
    with torch.no_grad():
        ln0 = model.ln_psi(samples)
        E_loc = model.E_loc(samples, ln0)              # (Ns,) 复数
        E = E_loc.mean()
        e_c = (E_loc - E).conj() / math.sqrt(ns)       # (1/√Ns)·(ε−E)*
    ebar = 2.0 * torch.cat([e_c.real, -e_c.imag]).to(DT)  # (2Ns,)
    # 实表示样本空间矩阵 T = [Oc_re; Oc_im] @ [Oc_re; Oc_im]^T  (2Ns×2Ns)
    A = torch.cat([Oc_re, Oc_im], dim=0)                # (2Ns, Np)
    T = A @ A.T                                         # (2Ns,2Ns)
    # 论文 1D 用 lr 带 inverse_schedule; 这里固定 lr. 大 lr 会放大。
    delta = lr * (A.T @ torch.linalg.solve(T + lam * torch.eye(2 * ns).to(T), ebar))
    # trust-region lr (论文 Eq. trust-region-lr): η_star = η/‖τ‖
    if trust_region:
        with torch.no_grad():
            tau = (torch.eye(2 * ns).to(T) - lam * torch.linalg.inv(T + lam * torch.eye(2 * ns).to(T))) @ ebar
            eta_star = lr / (torch.norm(tau) + 1e-10)
        delta = delta * (eta_star / lr)
    # 动量
    m_buf = momentum * m_buf + (1 - momentum) * delta
    return m_buf, E.real, E_loc.real


def run_adam(model, epochs, lr=0.02, ns=200, grad_clip=1.0):
    """你的 MC VMC-Adam (直采, 无 det 热身), 返回 E/site 历史。"""
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for it in range(epochs):
        states = model.sample(ns)
        with torch.no_grad():
            ln0 = model.ln_psi(states)
            E_loc = model.E_loc(states, ln0)
            a = E_loc.real.detach().clamp(-50, 50)
            b = E_loc.imag.detach().clamp(-50, 50)
        ln = model.ln_psi(states)
        loss = ((a - a.mean()) * ln.real).sum() + ((b - b.mean()) * ln.imag).sum()
        opt.zero_grad()
        try:
            grads = torch.autograd.grad(loss, params)
        except RuntimeError:
            hist.append(float('nan')); continue
        tot = math.sqrt(sum(g.detach().square().sum() for g in grads))
        sc = min(1.0, grad_clip / (tot + 1e-8))
        for g, p in zip(grads, params):
            p.grad = (2.0 / ns) * g * sc
        opt.step()
        with torch.no_grad():
            for p in params:
                p.clamp_(-10, 10)
        hist.append(float(a.mean() / model.L))
    return hist


def run_minsr(model, epochs, lr=0.05, lam=1e-3, mu=0.7, ns=200, trust_region=False):
    params = list(model.parameters())
    m_buf = torch.zeros(flat_params(params).shape[0], dtype=DT)
    hist = []
    for it in range(epochs):
        states = model.sample(ns)
        m_buf, E, _ = minsr_step(model, states, lr=lr, lam=lam, momentum=mu,
                                 m_buf=m_buf, trust_region=trust_region)
        set_flat(params, flat_params(params) - m_buf.to(
            next(model.parameters()).dtype))
        with torch.no_grad():
            for p in params:
                p.clamp_(-10, 10)
        if math.isnan(E):
            hist.append(float('nan'))
        else:
            hist.append(float(E / model.L))
    return hist


def eval_e(model, ns=2000):
    with torch.no_grad():
        st = model.sample(ns)
        ln0 = model.ln_psi(st)
        El = model.E_loc(st, ln0)
    return float(El.real.mean() / model.L), float(El.real.std() / model.L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--ns', type=int, default=200)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr_adam', type=float, default=0.02)
    ap.add_argument('--lr_minsr', type=float, default=0.05)
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--mu', type=float, default=0.7)
    ap.add_argument('--trust_region', action='store_true')
    ap.add_argument('--softsign_phase', action='store_true', help='有界相头 φ=π·Softsign (论文 Eq.145)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    else:
        auto_threads()
    torch.manual_seed(args.seed)
    E0, E0site = heisenberg_ed(args.L)
    print(f'== 基准 == ED L={args.L} E0/site={E0site:.8f}')
    print(f'== 配置 == dh={args.dh} ns={args.ns} steps={args.steps} '
          f'lr_adam={args.lr_adam} lr_minsr={args.lr_minsr} lam={args.lam} mu={args.mu} '
          f'trust_region={args.trust_region}')

    # 同一初始化 → 公平对比优化器
    m_adam = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed, softsign_phase=args.softsign_phase)
    m_minsr = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed, softsign_phase=args.softsign_phase)
    print(f'参数量={m_adam.param_count}')
    print(f'自检 ΣP = {m_adam.check_normalization():.5f} (应≈1)')

    print('\n[Adam] 直采 VMC-Adam (无 det 热身)...')
    t0 = time.time()
    h_a = run_adam(m_adam, args.steps, lr=args.lr_adam, ns=args.ns)
    e_a, s_a = eval_e(m_adam)
    print(f'  Adam  用时 {time.time()-t0:.0f}s  final E/site={e_a:+.6f}±{s_a:.4f} '
          f'(ED {E0site:+.6f}, 差 {(e_a-E0site)/abs(E0site)*100:.2f}%)  '
          f'min={min(h_a):+.4f} max={max(h_a):+.4f}')

    print(f'[minSR] 论文 sample×sample ('f'λ={args.lam}, μ={args.mu}, '
          f'trust={args.trust_region})...')
    t0 = time.time()
    h_m = run_minsr(m_minsr, args.steps, lr=args.lr_minsr, lam=args.lam,
                    mu=args.mu, ns=args.ns, trust_region=args.trust_region)
    e_m, s_m = eval_e(m_minsr)
    print(f'  minSR 用时 {time.time()-t0:.0f}s  final E/site={e_m:+.6f}±{s_m:.4f} '
          f'(ED {E0site:+.6f}, 差 {(e_m-E0site)/abs(E0site)*100:.2f}%)  '
          f'min={min(h_m):+.4f} max={max(h_m):+.4f}')

    n_nan_a = sum(math.isnan(x) for x in h_a); n_nan_m = sum(math.isnan(x) for x in h_m)
    print(f'\n== 稳定性 == Adam nan步={n_nan_a}/{args.steps}, minSR nan步={n_nan_m}/{args.steps}')
    import numpy as np
    np.savez('data/rnn_minsr_cmp.npz', hist_adam=np.array(h_a), hist_minsr=np.array(h_m),
             E0=E0, L=args.L)


if __name__ == '__main__':
    main()