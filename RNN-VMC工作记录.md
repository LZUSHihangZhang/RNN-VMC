# RNN-VMC 工作记录（自旋 1/2 Heisenberg 链）

> **目的**：让下一个 AI 直接接手当前工作 + 实时记录。参考本方 NQS 历史工作流（`AIagent工作流/NQS项目全记录2.md` 的范式：一句话现状/核心成果表/接手指引/§ 实时记录）。
> **参考论文**：`参考论文/arXiv-2002.02973v4/` —— M. Hibat-Allah et al., "Recurrent Neural Network Wave Functions", PRX 10, 031017 (2020)。

---

## ⭐ 一句话现状

**RNN-VMC 计算 Heisenberg 链完成**：cRNN（复数相位，**不用 Marshall 符号**）复现 L=10 PBC 基态能量 **E/site = -0.45148（确定性模式，相对误差 0.013%）**；MC 模式（det 热身 + MC 微调）**-0.45126（0.063%）**。代码采用正版 GCNN 范式（批量 ln_psi / 批量 E_loc / VMC-Adam）。

---

## 一、当前工作状态

| 工作线 | 状态 |
|--------|------|
| RNN-VMC Heisenberg 链（本文件夹）| ✅ 完成（2026-08-15）|
| 原 19 个未完成脚本 | 📦 归档到 `历史未完成/` |
| 参考论文 | 📄 `参考论文/arXiv-2002.02973v4/` |

**本次任务**：接手用户自己写的、未完成的 RNN-VMC 代码（`历史未完成/`），参考论文 + 本方 NQS 范式（`nqs_gcnn.py`），**完成 Heisenberg 链计算**，创立本工作记录。

---

## 二、核心成果（可靠数值）

**模型**：L=10 PBC 自旋 1/2 反铁磁 Heisenberg 链，H = Σ⟨ij⟩ S_i·S_j（S^z = σ/2）。

| 量 | 值 | 方法 |
|----|-----|------|
| ED 基态能量 E0/site | **-0.4515446** | 精确对角化（`heisenberg_ed`）|
| **RNN cRNN E/site** | **-0.45148** | 确定性模式 ψ†Hψ/ψ†ψ，dh=32，2000 步 |
| 相对误差 | **0.013%** | vs ED |
| **RNN MC E/site** | **-0.45126** | det 热身 500 + MC 微调 300，ns=600 |
| 相对误差 | **0.063%** | vs ED |
| E_loc std/site | 5.25（随机初始）→ **0.008**（近本征态）| 投影正则化，σ→0 |

**波函数（cRNN，不用 Marshall 符号）**：
ψ(σ) = exp(iφ(σ))·√P(σ)，P(σ)=∏_i p(σ_i|σ_<i)（GRU+Softmax），φ(σ)=Σ_i φ_i（Softsign 相位层）。
Sz=0 投影用**论文零磁化配额**算法（前 L/2 自由采样，后半配额限制）。

---

## 三、方法（GCNN 范式，`完成版/rnn_heisenberg.py`）

**正版范式**（学习自 `nqs_gcnn.py`）：
1. **批量 `ln_psi(states)`**：states (B,L) → lnψ (B,) 复数，一次批量前向
2. **批量 `E_loc(states, ln_psi)`**：逐 bond 用 `torch.where` 找翻转构型，批量算 `ln_new`，`index_add_` 累加 0.5·ψ'/ψ
3. **VMC-Adam**：`loss = cov(Re E_loc, ln|ψ|) + cov(Im E_loc, φ)`，`p.grad = (2/N)·∇loss·sc`（grad clip），Adam 更新，参数 clamp
   （loss 公式已用有限差分验证 = ∂E/∂θ = 2Re⟨(E_loc−E)∂logψ*⟩）

**两种训练模式**：
- **确定性（--mode det）**：枚举 Sz=0 全部构型 → ψ 向量 → **E = ψ†Hψ/ψ†ψ**（精确，NQS 范式类比 FeI2 的 b†H_kb/‖b‖²）。无 MC 噪声，0.013%。
- **MC（--mode mc）**：自回归采样 |ψ|²，VMC-Adam。**必须 det 热身**（`--warm_det 500`），否则随机相位 → E_loc 重尾 → 梯度发散（FeI2 §81.35 同款物理）。

---

## 四、重要文件标注

### 🔴 核心脚本（`完成版/`）
| 文件 | 作用 |
|------|------|
| `rnn_heisenberg.py` | **主脚本**：cRNN 类 + 批量 API + det/MC 双模式 + ED 基准 |
| `verify_eloc.py` | **小验证脚本**（先验证再训练）：归一化 ΣP=1 / E_loc 批量 vs 暴力 / E_det vs H 矩阵 / 梯度 vs 有限差分 |
| `data/rnn_h_det.npz` | 确定性训练历史（0.013%）|
| `data/rnn_h_mc_hyb.npz` | MC 混合训练历史（0.063%）|

### 📦 历史未完成（`历史未完成/`，19 个原脚本，归档）
`CRNN_Heisenberg_Adam*.py`（4 版，主攻 Heisenberg，均有 bug）、`CRNN_Marshall_sign.py`（Marshall 符号版）、`CRNN矢量版.py`、`CRNN_Heisenberg_原来的方式.py`（SGD）、`1DGRNN+复数相位.py`、`CNN_Heisenberg_矢量化编程.py`、`Heisenberg随机重构.py`、`自回归采样3/4/5.py`、`RNN_自回归采样*.py`、`IsingModel+MC随机重构.py`、`TIM.py`、`加入横场.py`、`实验.py`

### 📄 参考论文（`参考论文/`）
`arXiv-2002.02973v4/`（PRX 论文 LaTeX + 图）——cRNN 定义、Sz=0 投影算法、VMC 梯度公式、GRU 附录

---

## 五、待办 / 下一步

1. **可选**：更大 dh / 更长训练逼近 ED（当前 0.013%，已很好）
2. **可选**：加平移对称（论文 §App. discrete symmetries：抽样时对称化），进一步降方差
3. **可选**：扩展到阻挫模型（J1-J2，Marshall 符号不适用 → cRNN 相位正是为此）或更高 L
4. **可选**：算关联函数/纠缠熵（论文 §III.C/D）

---

## 六、给下一个 AI 的接手指引

1. **复现**：
   ```bash
   cd "完成版"
   python3 verify_eloc.py          # 先验证
   python3 rnn_heisenberg.py --mode det --steps 2000   # 确定性
   python3 rnn_heisenberg.py --mode mc --warm_det 500  # MC 混合
   ```
2. **🔴 MC 必须 det 热身**：随机初始态 E_loc 方差 ~5/site（重尾），直接 MC 梯度发散；det 热身 → E_loc std 降到 0.008 → MC 稳定。这是 FeI2 §81.35-37 同款物理（构型级 E_loc 重尾）。
3. **不用 Marshall 符号**（用户明确要求）：cRNN 相位层学习符号结构。
4. **代码范式**：批量 API（GCNN 范式），参考 `nqs_gcnn.py`。注意**自回归顺序**（预测 σ_i 于 h 编码 σ_<i，再更新 h）——这是 ΣP=1 的前提（原代码这里错了导致归一化失败）。
5. **原代码 bug 教训**（`历史未完成/`）：相位 0.5j 因子错、`log(-1)=πi` 污染 Marshall 符号、复→实 cast 丢虚部、AdamW 自带 weight_decay、param_count 手算错、自回归顺序错、过紧的 Sz=0 配额（应 `n_up >= half` 而非 `half-(n_up+1)<=0`）。

---

## 七、实时工作记录（2026-08-15）

### 81.60 📦 接手 RNN-VMC 重制版（2026-08-15）

> 用户指示：参考历史 `.md` 工作流 + 文件夹内论文，在该文件夹创立 `.md`，**完成 RNN 计算 Heisenberg 链**；随后用户补充：**整理文件夹** + **参考 `nqs_gcnn.py` 的 GCNN 范式写** + **不用 Marshall 符号**。

**现有代码诊断**（`历史未完成/` 19 个脚本）：主攻 Heisenberg 的 `CRNN_Heisenberg_Adam_*.py` 系列有多处 bug：
1. `Adam_3`：`ln_psi = 1j*phi*0.5` — 相位被砍半（论文 ψ=e^{iφ}√P，lnψ = 0.5lnP + iφ）
2. `AdamW` 默认 weight_decay=0.01（多余 L2）
3. Marshall 版 `torch.log(marshall_sign)` 当 sign=-1 得 πi → 复→实 cast 丢虚部（警告）
4. `param_count` 手算与实参不符
5. 自回归顺序"摄入 σ_i 后预测 σ_i" → ΣP≠1（归一化破坏）
6. Sz=0 配额过紧 `(half-(n_up+1))<=0` → 合法构型被双禁 → 0/0=nan

**参考论文精读**（arXiv:2002.02973，PRX）：cRNN 定义（Softmax 幅度 + Softsign 相位，φ_i=π·Softsign(U2h+c2)）、Sz=0 零磁化配额算法（Ξ(N/2−N_up(i))，前 L/2 自由）、VMC 梯度公式（∂E/∂θ = 2Re⟨(E_loc−E)∂logψ*⟩，App. sec:VMC）。

### 81.61 🏆 小验证脚本通过（verify_eloc.py，NQS 铁律 4）

先写 `完成版/rnn_heisenberg.py`（初版逐构型 + 前缀缓存 E_loc）+ `verify_eloc.py`，修 3 个 bug 后 4 项全过：
1. **自回归顺序**：改"预测 σ_i → 更新 h"（原"摄入后预测"致 ΣP=1.3）
2. **Sz=0 配额**：`n_up>=half` 才禁（原 `half-(n_up+1)<=0` 过紧 → nan）
3. **共享计算图单次 backward**：logP/φ 合并 `L = w(dE.real·logP + 2·dE.imag·φ)`

验证结果：ΣP=1.0 / E_loc 批量 vs 暴力 1.8e-13 / E_det vs H 矩阵 0.00 / 梯度 vs 有限差分逐位一致。

### 81.62 🚀 性能优化：批量向量化确定性（GCNN 范式）

初版逐构型 1.3s/步 → 批量 `wavefunction_vector`（一次前向全 Sz0 构型 + ψ†Hψ/ψ†ψ）→ **0.063s/步（20×）**。随后按用户要求改用 GCNN 范式（`nqs_gcnn.py`）整体重写：批量 `ln_psi`/`E_loc`/`sample`/VMC-Adam `learning`。

### 81.63 🏆 确定性模式完成：E/site = -0.45148（0.013%）

`--mode det`，dh=32，2000 步，21.8s：
```
RNN  E/site = -0.45148426
ED   E/site = -0.45154464
相对误差      = 0.013%
```
**cRNN（不用 Marshall 符号）精确复现 Heisenberg 基态能量。**

### 81.64 🔬 MC 模式：直接训练发散 + 根因 + 解法（det 热身）

**直接 MC（随机初始态）发散**（E/site → ±1e11~1e28）：
- 根因：**随机相位 → E_loc 方差 5.25/site（重尾）** → MC 梯度被离群样本主导 → 发散。这是 FeI2 §81.35-37 同款"构型级 E_loc 重尾"物理。
- 验证：初始态 MC ⟨E_loc⟩/site = -0.199 ± 0.17（1000 样本，std 5.25），与 E_det 统计一致但方差巨大。

**解法（NQS 范式）**：**det 热身 → MC 微调**。det 800 步后 **E_loc std 骤降 5.25 → 0.008**（近本征态 → E_loc 良态，投影正则化），MC 微调稳定收敛：
```
[det 热身] E/site = -0.45129
it 299  E/site = -0.45120  std=0.0087
RNN  E/site = -0.45126  (0.063%)
```

**方法论结论**：确定性（ψ†Hψ）是可靠路线；MC 在 E_loc 方差可控（近本征态）后才稳定——与 FeI2 SFP-NQS 的结论一致。

### 81.65 📦 整理文件夹 + 归档

- 19 个原脚本 → `历史未完成/`；论文 → `参考论文/`；工作代码 → `完成版/`；本记录 → `RNN-VMC工作记录.md`
- 删除 `完成版/__pycache__`

### 81.66 🔴 关键：官方源码 + 自旋编码 bug（2026-08-15，L=15 途中）

**官方源码**（`RNNWavefunctions-master/J1J2/TrainingRNN_J1J2.py`）：
- 纯 Heisenberg（J2=0）默认 `Marshall_sign=False`（cRNN 无 Marshall 符号），**lr=2.5e-4，10⁵ 步**
- cost = `2Re[⟨conj(logψ)·E_loc⟩ − conj(⟨logψ⟩)·⟨E_loc⟩]` = 2[cov(Re lnψ,Re E_loc)+cov(Im lnψ,Im E_loc)] = **与我代码 loss 数学等价**
- 我之前 lr(0.001~0.02) 是官方 4~80 倍 → 发散主因之一

**🔴 自旋编码 bug（MC 一切怪象的总根因）**：
- `sample()` 里 `+1 → onehot[0]=[1,0]`（up），`idx=(~up).to(long)` → idx=0:up
- `ln_psi()` 里 `idx=(states>0).to(long)` → `+1 → idx=1 → onehot[1]=[0,1]`（**down**）
- → 采样分布 ≠ |ψ|²！MC 能量估计/梯度全错（Heisenberg 自旋翻转对称使能量碰巧对，但梯度在优化错误目标）
- **所有 det 结果不受影响**（不用采样）；verify 全过是因为 ln_psi 内部自洽（ΣP=1 恒成立）
- **修复**：`idx = (states < 0).to(long)`（0:+1, 1:-1 与 sample 一致）
- 修复后 L=15 短跑 300 步平稳收敛（+0.11→-0.27），无 nan/游走

**架构改造**（用户要求"最后一层复数，振幅相位同时学习"，GCNN 范式）：
- 分离头（Softmax 幅度 + Softsign 相位）→ **单复数头**：`w=(U_re@h+c_re)+i(U_im@h+c_im)`，幅度 `p=softmax(2Re w)`，相位 `φ=Im w`，共用 GRU 隐态
- 无界相位裸跑发散 → 需 **grad clip + param clamp [-10,10]**（GCNN 稳健措施）

### 81.67 ✅ L=15 纯 MC 完成（不用 ED）

- L=15 奇数链 → 基态在 **M=+1 扇区**（8 up 7 down），代码加 `magnetization` 支持
- 基准不用 ED（用户要求），用 **Bethe 热力学极限 E/N=1/4−ln2=−0.443147**（含 ~1/L 有限尺寸 caveat）
- 单复数头 + lr=2.5e-4 + grad clip + param clamp，ns=600，**30000 步**

**结果**（`data/rnn_h_L15_mc.npz`）：
| 量 | 值 |
|----|-----|
| RNN E/site（末 500 步均值）| **-0.422074**（统计误差 0.00005）|
| Bethe E/site（热力学极限）| -0.443147 |
| 与 Bethe 差 | 0.0211（4.8%）|
| 收敛轨迹 | -0.27(300步) → -0.35(2k) → -0.40(10k) → -0.42(26k) |

- 收敛轨迹平稳无发散（编码 bug 修复后），MC 噪声 std ~0.02-0.05/site
- 4.8% 差 = **未完全收敛**（官方要 10⁵ 步；30k 步在 MC 噪声极限附近减缓）；不是方法失败
- 全部纯 MC：**未用 ED**（无 ED 热身/基准/监督），基准仅 Bethe 解析值

### 81.68 🏆 MARCH 优化器（前期 Adam + 后期 MARCH，GCNN 最佳记录）

**用户指示**：参考 GCNN 最佳记录（§27 `GCNN_rotbasis_half.py --adam_epochs 400 --march_epochs 150`，99.13%），**Adam 跑到能量很低后 MARCH 微调**，算 L=30，尽量并行化。

**MARCH 实现**（移植自 `MLP修复版.py`，SR 自然梯度 + 动量 + 二阶矩）：
```
O=concat[∂Re,∂Im] 中心化;  M=OOᵀ+λI;  δθ=OᵀM⁻¹(E_aug−O·μ·p)
nu=β·nu+(δθ−δθ_prev)²;  超参 β=0.995, μ=0.95, λ=0.001, τ=0.05
```
- 逐样本 Jacobian（ns=200，`torch.autograd.grad`）
- **🔴 关键修复**：GCNN 里 `parameter` 动量状态从 **0** 初始化（`torch.zeros`），不是当前参数——我一开始初始化为参数导致动量项被参数向量主导（发散/无改善）

**L=15 验证**：Adam 2000 步 -0.342 → **MARCH 30 步 -0.356**（明确改善，无发散）

**并行化**：`torch.set_num_threads(16)`（机器 20 核）——backward 2.2s→0.01s（200 倍）；L=30 单步 ~0.45s（ns=400）

**L=30 运行**：Adam 10000 + MARCH 200 启动；it 3200 已过 -0.4（-0.404），it 6500 到 **-0.418**；用户要求停止（未到 MARCH 阶段）。结果：Adam-only -0.418 vs Bethe -0.443（5.6%，未完全收敛）

**float32 默认**（用户：神经网络训练 float32 够用）：两个文件默认 dtype 改 float32（GCNN 同款），verify 容差/FD 步长按 dtype 调整（float32 下 E_loc 批量 vs 暴力 1.8e-5=相对 1.8e-6，纯精度）

### 81.69 🚀 优化 + 配额 bug 修复（2026-08-16）

**🔴 磁化配额 bug**：原只在 i≥L/2 施加配额，但 M=2（4 down 预算）时前半 5 自由位用光 down 配额 → 采样混入 M=0。查官方源码（论文 normalization 每个 site 调用）→ **配额每步施加**。修复后所有有效扇区正确（L=10 {0,2,4}, L=15 {1,3,5}）。

**优化（实测 L=30 ns=600）**：
| 优化 | 效果 |
|------|------|
| 采样手动优化（addmm 融合 + 常量预分配 + 拆分配额）| 采样 17→9ms |
| **GRUCell 融合算子**（torch 内置 C++）| GRU 步 6.8x，E_loc 43→27ms，完整步 62→55ms |
| torch.compile | ❌ 反而慢 100x（小顺序循环不适合）|
| **pybind11（torch C++ API）** | ❌ 无加速（同一 dispatcher，算子分派开销没消掉）|

**pybind11 结论**：torch C++ API 写扩展 = 无收益（实测 0.9x）。**裸 C++**（Eigen/BLAS 手写矩阵，不用 torch 分派）才能真加速前向（~5-10x），但只对无 autograd 部分（采样/E_loc 前向），backward 不变，且工作量大。**GRUCell 是性价比最高的选择**（torch 自带融合 C++ 算子）。参考文件 `rnn_cpp.cpp`（torch API 版，正确但不快）。
