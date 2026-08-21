#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Adam · MC（无 det 热身）· 实时落盘 · 接近基态即停。
----------------------------------------------------------------------
复用 RNNHeisenberg + 逐步骤 VMC-Adam（与 rnn_minsr.run_adam 同逻辑），
每 eval_every 步用高样本评估 E/site，实时写 data/mc_watch_L{}.log，
误差 < tol% 即停止。参考值 = M=0 扇区稀疏 ED（复用 §L16 ed_sector）。
用法:
  python3 rnn_mc_watch.py --L 14 --lr 2.5e-4 \
      --ns 600 --eval_every 500 --tol 1.5 --max_steps 40000
"""
import torch, argparse, time, os, math
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from rnn_heisenberg import RNNHeisenberg, enumerate_sector


def ed_sector(L, M):
    cfgs = list(enumerate_sector(L, M))
    idx = {tuple(c): i for i, c in enumerate(cfgs)}
    N = len(cfgs); rows, cols, vals = [], [], []
    for i, c in enumerate(cfgs):
        s = list(c)
        for j in range(L):
            j1 = (j + 1) % L
            vals.append(s[j] * s[j1] / 4.0); rows.append(i); cols.append(i)
            if s[j] != s[j1]:
                t = s.copy(); t[j] *= -1; t[j1] *= -1
                k = idx[tuple(t)]
                if i < k:   # 仅较小端加一次, 防 off-diag 双计 (关键修复 §81)
                    vals.append(0.5); rows.append(i); cols.append(k)
                    vals.append(0.5); rows.append(k); cols.append(i)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    w = spla.eigsh(H, k=1, which='SA', return_eigenvectors=False)
    return w[0], w[0] / L


def eval_e(model, ns=2000):
    with torch.no_grad():
        st = model.sample(ns)
        ln0 = model.ln_psi(st)
        El = model.E_loc(st, ln0)
    return float(El.real.mean() / model.L), float(El.real.std() / model.L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=14)
    ap.add_argument('--dh', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--ns', type=int, default=600)
    ap.add_argument('--eval_ns', type=int, default=2000)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--tol', type=float, default=1.5, help='误差%<tol 即停')
    ap.add_argument('--max_steps', type=int, default=40000)
    ap.add_argument('--eval_ns_hi', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    E0, E0s = ed_sector(args.L, 0)
    tag = args.tag or f'L{args.L}'
    logpath = f'data/mc_watch_{tag}.log'
    zpath = f'data/mc_watch_{tag}.npz'
    os.makedirs('data', exist_ok=True)

    m = RNNHeisenberg(args.L, dh=args.dh, seed=args.seed)
    log = open(logpath, 'w', buffering=1)
    log.write(f'# 纯Adam·MC 看门狗  L={args.L} dh={args.dh} lr={args.lr} '
              f'ns={args.ns} eval_every={args.eval_every} tol={args.tol}% '
              f'E0(M=0 ED)={E0:.6f} E0/site={E0s:.6f}\n'); log.flush()
    print(f'ED L={args.L} M=0 E0/site={E0s:.6f}', flush=True)

    params = list(m.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    hist, evals = [], []
    t0 = time.time()
    for it in range(args.max_steps):
        states = m.sample(args.ns)
        with torch.no_grad():
            ln0 = m.ln_psi(states)
            El = m.E_loc(states, ln0)
            a = El.real.detach().clamp(-50, 50)
            b = El.imag.detach().clamp(-50, 50)
        ln = m.ln_psi(states)
        loss = ((a - a.mean()) * ln.real).sum() + ((b - b.mean()) * ln.imag).sum()
        opt.zero_grad()
        grads = torch.autograd.grad(loss, params)
        tot = math.sqrt(sum(g.detach().square().sum() for g in grads))
        sc = min(1.0, 1.0 / (tot + 1e-8))
        for g, p in zip(grads, params):
            p.grad = (2.0 / args.ns) * g * sc
        opt.step()
        with torch.no_grad():
            for p in params:
                p.clamp_(-10, 10)
        hist.append(float(a.mean() / args.L))

        if (it + 1) % args.eval_every == 0:
            e, s = eval_e(m, args.eval_ns)
            er = (e - E0s) / abs(E0s) * 100
            evals.append((it + 1, e, s, er))
            ln_line = f'step {it+1:6d}  E={e:+.6f}±{s:.4f}  err={er:6.2f}%  t={time.time()-t0:5.0f}s'
            log.write(ln_line + '\n'); log.flush()
            print(ln_line, flush=True)
            if er < args.tol:
                e_hi, s_hi = eval_e(m, args.eval_ns_hi)
                er_hi = (e_hi - E0s) / abs(E0s) * 100
                done = f'>> 达标停止 @step {it+1}  复验 E={e_hi:+.6f}±{s_hi:.4f} err={er_hi:.3f}%'
                log.write(done + '\n'); log.flush()
                print(done, flush=True)
                break
    else:
        msg = '>> 达 max_steps 未达标'; log.write(msg+'\n'); log.flush(); print(msg, flush=True)

    e_f, s_f = eval_e(m, args.eval_ns_hi)
    er_f = (e_f - E0s) / abs(E0s) * 100
    fin = (f'final E={e_f:+.6f}±{s_f:.4f} err={er_f:.3f}% '
           f'hist_min={min(hist):+.5f} total_t={time.time()-t0:.0f}s')
    log.write(fin + '\n'); log.close()
    np.savez(zpath, hist=np.array(hist), evals=np.array(evals),
             E0=E0, E0s=E0s, L=args.L)
    print(fin, flush=True)
    print(f'落盘: {logpath} 与 {zpath}', flush=True)


if __name__ == '__main__':
    main()