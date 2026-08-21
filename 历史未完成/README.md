# 历史未完成脚本（归档，勿直接用）

这些是 2025-12 ~ 2026-02 期间自己写的 RNN-VMC 尝试，**均未完成/有 bug**。
已完成的正版实现见 `../完成版/rnn_heisenberg.py`（2026-08-15，GCNN 范式）。
工作全过程记录见 `../RNN-VMC工作记录.md`。

## 分类

### Heisenberg 链主攻（复杂相位 RNN，均有 bug）
| 文件 | 说明 | 主要 bug |
|------|------|---------|
| `CRNN_Heisenberg_Adam.py` | 首版 Adam 版 | 梯度/参数收集手写繁琐 |
| `CRNN_Heisenberg_Adam_2.py` | 修正 GRU 门 + 多线程采样 | 多线程共享参数不安全 |
| `CRNN_Heisenberg_Adam_3.py` | 最新版 | **相位 0.5j 因子错**、AdamW weight_decay |
| `CRNN_Marshall_sign.py` | Marshall 符号版 | `log(-1)=πi` 污染、复→实 cast 丢虚部 |
| `CRNN矢量版.py` | 批量版 | 同上 |
| `CRNN_Heisenberg_原来的方式.py` | SGD 手动更新 | 收敛慢 |
| `1DGRNN+复数相位.py` | 1D GRU 复数相位 | 相位用 softmax 非 Softsign |

### 采样/其他模型实验
| 文件 | 说明 |
|------|------|
| `RNN_自回归采样*.py`、`自回归采样3/4/5.py` | RNN 自回归采样练习 |
| `IsingModel+MC随机重构.py`、`TIM.py`、`加入横场.py` | Ising/TIM 模型 MC |
| `CNN_Heisenberg_矢量化编程.py` | CNN 矢量化尝试 |
| `Heisenberg随机重构.py` | 随机重构法 |
| `实验.py` | 极小的 dtype 测试 |

## 关键教训（已在完成版修复）
1. 自回归顺序：**先预测 σ_i 再更新 h**（摄入 σ_i 后再预测 σ_i 会破坏 ΣP=1）
2. Sz=0 配额：`n_up >= L/2` 才禁止（过紧条件 → 合法构型被双禁 → nan）
3. 相位：lnψ = 0.5·lnP + i·φ（无 0.5 因子）
4. 梯度：`∂E/∂θ = 2Re⟨(E_loc−E)∂logψ*⟩`，用 `ln_psi.conj()` 反向传播
5. 确定性优先（ψ†Hψ/ψ†ψ），MC 需 det 热身（初始 E_loc 重尾）
