#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Adam · MC（无 det 热身）固化的验证脚本。
------------------------------------------------------------------------
论文 arXiv:2608.18065 (minSR) 对比线收束结论 (§83): 最优解 = 用户原始代码
+ 小 lr。本脚本把其中 "纯 Adam + MC + warm_det=0" 的验证配置固化, 可复现:
  对同一 RNNHeisenberg (seed), 直接自回归采样 |ψ|², Adam 随机梯度最小化
  波动能量 E_loc 协方差 (VMC-Adam), 全套无 det 热身。跑完用稀疏 ED 参考值
  对照误差。

用法 (Sequential, 勿并发):
  python3 rnn_mc_pure_adam.py --L 10 --lr 2.5e-4 --ns 600 --steps 10000

观察:
  L=10  ED E0/site=-0.451545 (M=0 扇区稀疏 ED, = heisenberg_ed)
  小 lr (2.5e-4, 官方同款) 纯 MC 可贴近基态 (<1% 量级), std≈0.025, 仍在收敛。
"""
import torch, argparse, time, os

# 复用原有实现, 不另起炉灶
from rnn_heisenberg import RNNHeisenberg, heisenberg_ed


def eval_e(model, ns=2000):
    with torch.no_grad():
        st = model.sample(ns)
        ln0 = model.ln_psi(st)
        El = model.E_loc(st, ln0)
    return float(El.real.mean() / model.L), float(El.real.std() / model.L)


def main():
    ap = argparse.ArgumentParser(description='纯 Adam · MC（无 det 热身）验证')
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2.5e-4, help='小 lr 是收敛关键 (§81.66 / 官方同款)')
    ap.add_argument('--ns', type=int, default=600, help='MC 每步样本数')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    E0, E0s = heisenberg_ed(args.L)
    m = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed)
    print(f'== 基准 == ED L={args.L} E0/site={E0s:.6f}  参数量={m.param_count}')
    print(f'== 配置 == 纯 Adam · MC · warm_det=0  lr={args.lr} ns={args.ns} '
          f'steps={args.steps} (seed={args.seed})')
    print(f'自检 ΣP = {m.check_normalization():.6f} (应≈1)')

    t0 = time.time()
    hist, _st = m.learning(args.steps, lr=args.lr, ns=args.ns,
                            grad_clip=args.grad_clip, warm_det=0,   # << 无 det 热身
                            march_epochs=0, verbose=True)           # << 纯 Adam
    e, s = eval_e(m)
    dt = time.time() - t0
    er = (e - E0s) / abs(E0s) * 100
    print(f'\n[纯 Adam · MC] 用时 {dt:.0f}s  final E/site={e:+.6f}±{s:.4f} '
          f'(ED {E0s:+.6f}, 差 {er:.2f}%)')
    print(f'  hist min={min(hist):+.6f}  last={hist[-1]:+.6f}  仍在收敛={hist[-1] > e}')

    os.makedirs('data', exist_ok=True)
    import numpy as np
    np.savez(f'data/rnn_mc_pure_adam_L{args.L}.npz',
             hist=np.array(hist), E0=E0, E0s=E0s, L=args.L,
             lr=args.lr, ns=args.ns, steps=args.steps, seed=args.seed)


if __name__ == '__main__':
    main()