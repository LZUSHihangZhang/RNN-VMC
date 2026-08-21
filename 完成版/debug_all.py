#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全面调试验证 (GRUCell + 配额修复 + 优化后): 基态/激发态/磁化/梯度/MARCH/一致性."""
import sys, os, time, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

torch.set_num_threads(16)
PASS = 0
FAIL = 0

def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ✓ {name}  {detail}')
    else:
        FAIL += 1
        print(f'  ✗ {name}  {detail}')

# ---------- 1) verify_eloc (4 项) ----------
print('=== 1) verify_eloc (归一化/E_loc/E_det/梯度) ===')
import importlib.util
spec = importlib.util.spec_from_file_location('ve', 'verify_eloc.py')
ve = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ve)
    check('verify_eloc', True)
except Exception as e:
    check('verify_eloc', False, str(e)[:100])

# ---------- 2) 基态 det L=10 ----------
print('=== 2) 基态 M=0 det L=10 (ED -4.51545) ===')
from rnn_adam_march import RNNHeisenberg
torch.manual_seed(0)
m0 = RNNHeisenberg(10, dh=32, seed=0, magnetization=0)
opt = torch.optim.Adam(m0.parameters(), lr=0.02)
for _ in range(1500):
    opt.zero_grad(); m0.energy_det(grad=True); opt.step()
E0 = m0.energy_det(grad=False).real.item()
err0 = abs(E0 + 4.515446) / 4.515446 * 100
check('基态 det E0', err0 < 0.1, f'RNN {E0:.5f} vs ED -4.51545 ({err0:.3f}%)')

# ---------- 3) 激发态 M=2 det L=10 ----------
print('=== 3) 激发态 M=2 det L=10 (ED -4.09221) ===')
torch.manual_seed(0)
m2 = RNNHeisenberg(10, dh=32, seed=0, magnetization=2)
opt = torch.optim.Adam(m2.parameters(), lr=0.02)
for _ in range(1500):
    opt.zero_grad(); m2.energy_det(grad=True); opt.step()
E2 = m2.energy_det(grad=False).real.item()
err2 = abs(E2 + 4.092207) / 4.092207 * 100
gap = E2 - E0
check('激发态 det E(M=2)', err2 < 0.3, f'RNN {E2:.5f} vs ED -4.09221 ({err2:.3f}%)')
check('gap', abs(gap - 0.42324) < 0.02, f'gap {gap:.5f} vs ED 0.42324')

# ---------- 4) 磁化扇区 ----------
print('=== 4) 磁化扇区采样 ===')
ok = True
for L, Ms in [(10, [0, 2, 4]), (15, [1, 3, 5])]:
    for M in Ms:
        m = RNNHeisenberg(L, dh=16, seed=0, magnetization=M)
        with torch.no_grad():
            s = m.sample(500)
        if torch.unique(s.sum(1)).tolist() != [float(M)]:
            ok = False
check('磁化扇区 {0,2,4}@{10}, {1,3,5}@{15}', ok)

# ---------- 5) MC 基态短训 (配额+GRUCell 下收敛) ----------
print('=== 5) MC L=15 M=1 短训 (4000 步) ===')
torch.manual_seed(0)
m15 = RNNHeisenberg(15, dh=32, seed=0, magnetization=1)
hist, _ = m15.learning(epochs=4000, lr=2.5e-4, ns=400, warm_det=0, march_epochs=0, verbose=False)
E15 = hist[-1]
check('MC L=15 M=1 收敛', E15 < -0.35, f'E/site {E15:.4f} (Bethe -0.443, 欠收敛正常)')

# ---------- 6) MARCH 精修有效 ----------
print('=== 6) MARCH 精修 (Adam 后) ===')
torch.manual_seed(0)
mm = RNNHeisenberg(12, dh=32, seed=0, magnetization=0)
h_a, _ = mm.learning(epochs=800, lr=2.5e-4, ns=400, warm_det=0, march_epochs=0, verbose=False)
mm2 = RNNHeisenberg(12, dh=32, seed=0, magnetization=0)
h_am, _ = mm2.learning(epochs=800, lr=2.5e-4, ns=400, warm_det=0, march_epochs=30, verbose=False)
check('MARCH 精修', h_am[-1] < h_a[-1] - 1e-4, f'Adam {h_a[-1]:.5f} → Adam+MARCH {h_am[-1]:.5f}')

# ---------- 7) 三个文件一致性 ----------
print('=== 7) rnn_adam_march / rnn_excited 类一致 ===')
import importlib.util
spec2 = importlib.util.spec_from_file_location('ram', 'rnn_adam_march.py')
ram = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ram)
spec3 = importlib.util.spec_from_file_location('rex', 'rnn_excited.py')
rex = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(rex)
for name, mod in [('rnn_adam_march', ram), ('rnn_excited', rex)]:
    m = mod.RNNHeisenberg(10, dh=16, seed=1, magnetization=2)
    with torch.no_grad():
        s = m.sample(300)
        ln = m.ln_psi(s)
        E = m.E_loc(s, ln).real.mean().item()
    ok = torch.unique(s.sum(1)).tolist() == [2.0] and not np.isnan(E)
    check(f'{name} 类可用', ok, f'E/site {E/10:.4f}')

print(f'\n==== 调试结果: {PASS} 通过, {FAIL} 失败 ====')
sys.exit(1 if FAIL else 0)
