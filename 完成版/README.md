# RNN-VMC 完成版 (可移植)

自旋 1/2 反铁磁 Heisenberg 链的 RNN 变分蒙特卡洛 (VMC)，完整独立，可复制到任意装有 Python 的电脑直接运行。

## 文件（发送给别人只需以下 6 个）
| 文件 | 用途 |
|------|------|
| `rnn_adam_march.py` | 基态计算：cRNN（单复数头）+ 前期 Adam + 后期 MARCH 精修（自包含）|
| `rnn_excited.py` | 激发态计算：磁化扇区定位 + 扭曲边界色散 + ED 参考（自包含）|
| `verify_eloc.py` | 验证脚本（归一化 / E_loc / 梯度 vs 有限差分）|
| `debug_all.py` | 全面调试验证（基态/激发态/磁化/MARCH/一致性）|
| `requirements.txt` | 依赖（torch, numpy）|
| `README.md` | 本说明 |
| `data/` | 运行结果（自动创建）|

> 可选（非必需）：`rnn_heisenberg.py`（主开发版，与 rnn_adam_march 等价）、`rnn_cpp.cpp`（pybind11 参考，torch API 版实测无加速）

## 安装 (任意电脑)
```bash
pip install -r requirements.txt
# 或: pip install torch numpy
```
无需 GPU（CPU 多核自动利用）；有 GPU 也兼容。

## 运行
```bash
# 基态 (L=10 确定性, 精确变分, 对齐 ED ~0.01%)
python rnn_adam_march.py --L 10 --mode det --steps 2000

# 基态 MC (大 L, Adam→MARCH)
python rnn_adam_march.py --L 30 --mode mc --warm_det 0 --lr 2.5e-4 \
    --ns 500 --steps 8000 --march_epochs 200

# 激发态 (磁振子, det 对齐 ED ~0.06%)
python rnn_excited.py --L 10 --mode det

# 激发态色散 ε(k) (扭曲边界)
python rnn_excited.py --L 10 --mode det --thetas 0 1.57 3.14

# 验证
python verify_eloc.py
```

## 说明
- **自动线程**：默认自动检测 CPU 核数并分配（预留 2 核给系统）；`--threads N` 可手动覆盖。
- **float32**：默认，神经网络训练足够（GCNN 同款）。
- 代码**无外部依赖**（除 torch/numpy）、**无硬编码路径**、结果自动存 `data/`。
- 方法细节：参考 `../RNN-VMC工作记录.md`（§81.60-68 全记录）。

## 参考
M. Hibat-Allah et al., "Recurrent Neural Network Wave Functions", PRX 10, 031017 (2020) [arXiv:2002.02973].
