import torch
import math
import random
from concurrent.futures import ThreadPoolExecutor
import threading

dtype1 = torch.float64
dtype2 = torch.complex128
device1 = torch.device('cpu')
device2 = torch.device('cuda')


class Complex_RNN:
    def __init__(self, L):
        self.L = L
        self.sigma1 = torch.tensor([1, 0], dtype=dtype1, device=device1)
        self.sigma0 = torch.tensor([0, 1], dtype=dtype1, device=device1)
        self.sigma = torch.stack([self.sigma1, self.sigma0], dim=0)
        self.dh = 5
        self.dv = 2
        self.Wu = torch.rand((self.dh, self.dh + self.dv), device=device1, dtype=dtype1)
        self.Wr = torch.rand((self.dh, self.dh + self.dv), device=device1, dtype=dtype1)
        self.bu = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.br = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.Wc1 = torch.rand((self.dh, self.dh), device=device1, dtype=dtype1)
        self.Wc2 = torch.rand((self.dh, self.dv), device=device1, dtype=dtype1)
        self.bc1 = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.bc2 = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.U1 = torch.rand((self.dv, self.dh), device=device1, dtype=dtype1)
        self.U2 = torch.rand((self.dv, self.dh), device=device1, dtype=dtype1)
        self.c1 = torch.rand(self.dv, device=device1, dtype=dtype1)
        self.c2 = torch.rand(self.dv, device=device1, dtype=dtype1)
        self.state0 = torch.rand(2, device=device1, dtype=dtype1)
        self.h0 = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.step = 500
        self.alpha = 0.1
        self.param_count = 3 * self.dh ** 2 + 5 * self.dh * self.dv + 4 * self.dh + 2 * self.dv

        # 添加Adam优化器
        import torch.optim as optim
        self.optimizer = optim.AdamW([
            self.Wu, self.Wr, self.bu, self.br,
            self.Wc1, self.Wc2, self.bc1, self.bc2,
            self.U1, self.U2, self.c1, self.c2
        ], lr=self.alpha)

    def Softsign(self, x):
        return x / (1 + torch.abs(x))


    '''Gate单位,减小对平移不变形的破坏'''
    def Gate(self, state, h):
        """
        修正后的GRU门控单元
        按照论文标准GRU结构实现
        """
        # 1. 输入拼接 - 您的这部分正确
        concat_input = torch.cat((h, state), dim=0)  # 维度: (dh + dv)

        # 2. 门控计算 - 需要修正参数维度
        u_n = torch.sigmoid(torch.mv(self.Wu, concat_input) + self.bu)  # 更新门
        r_n = torch.sigmoid(torch.mv(self.Wr, concat_input) + self.br)  # 重置门

        # 3. 候选隐藏态计算 - 关键修正部分
        # 标准GRU: h' = tanh(Wc2 * state + r_n ⊙ (Wc1 * h) + bc)
        Wc1_h = torch.mv(self.Wc1, h) + self.bc1  # 线性变换
        reset_gate_effect = r_n * Wc1_h  # 重置门作用
        input_contribution = torch.mv(self.Wc2, state) + self.bc2  # 输入贡献
        h_tilde = torch.tanh(input_contribution + reset_gate_effect)  # 候选状态

        # 4. 最终隐藏状态更新
        h_n = (1 - u_n) * h + u_n * h_tilde
        return h_n

    '''计算与state相关的局部能量'''
    def E_local(self, state, ln_psi):
        E_loc = 0 + 0j
        '''计算局域能量'''
        for i in range(self.L):
            if state[i] != state[i-1]:
                state1 = state.clone()
                state1[i] = -state1[i]
                state1[i - 1] = -state1[i - 1]
                h_next1 = self.h0
                lnP_relate1 = 0
                phi1 = 0
                # 计算右边格点自旋翻转后构型的概率
                for j in range(self.L):
                    h_next1 = self.Gate(self.sigma[int(state1[j].item())], h_next1)
                    y1R = torch.softmax(torch.mv(self.U1, h_next1) + self.c1, dim=0)
                    y1C = math.pi * self.Softsign(torch.mv(self.U2, h_next1) + self.c2)

                    a = int(state1[j].item() != 1)
                    lnP_relate1 += torch.log(y1R[a])
                    phi1 += torch.dot(y1C, self.sigma[a])
                ln_psi1 = 1j * phi1 + 0.5 * lnP_relate1
                E_loc += torch.exp(ln_psi1 - ln_psi) * 0.5

        state_rolled_right = torch.roll(state, shifts=1, dims=0)
        E_loc += torch.dot(state, state_rolled_right)/4
        return E_loc

    def CRNN(self):
        state = torch.zeros(self.L, device=device1, dtype=dtype1)
        h_next = self.h0
        state_initial = self.state0.clone()
        ln_P = 0
        phi = 0
        for i in range(self.L):
            h_next = self.Gate(state_initial, h_next)
            yn1 = torch.softmax(torch.mv(self.U1, h_next) + self.c1, dim=0)
            yn2 = math.pi * self.Softsign(torch.mv(self.U2, h_next) + self.c2)
            P1 = yn1[0]
            P2 = yn1[1]
            R = random.random()
            if R < P1:
                state[i] = 1
                state_initial = self.sigma1
                ln_P += torch.log(P1)
                phi += torch.dot(yn2, self.sigma1)
            else:
                state[i] = -1
                state_initial = self.sigma0
                ln_P += torch.log(P2)
                phi += torch.dot(yn2, self.sigma0)

        ln_psi = 1j * phi + 0.5 * ln_P  # 对于这个构型展开系数的对数值
        E_loc = self.E_local(state, ln_psi)  # 定义局域能量
        return E_loc, ln_psi, state


    '''多线程计算'''
    def independent_CRNN_sampling(self):
        """针对独立样本的简化多线程抽样"""
        num_samples = self.step

        # 不需要锁！因为样本独立，写入位置不同
        all_E = torch.zeros(num_samples, dtype=dtype2, device=device1)
        all_ln_psi = torch.zeros(num_samples, dtype=dtype2, device=device1)
        all_states = [None]*num_samples
        partial_E = torch.zeros(self.step, self.param_count, dtype=dtype2, device=device1)

        def independent_worker(sample_idx):
            """完全独立的抽样工作函数"""
            # 每个样本独立生成，无需考虑线程安全
            E_loc, ln_psi, state = self.CRNN()

            '''分实部和虚部来计算梯度'''
            '''计算实部的梯度'''
            # 清空梯度
            if self.Wu.grad is not None: self.Wu.grad.zero_()
            if self.Wr.grad is not None: self.Wr.grad.zero_()
            if self.bu.grad is not None: self.bu.grad.zero_()
            if self.br.grad is not None: self.br.grad.zero_()
            if self.Wc1.grad is not None: self.Wc1.grad.zero_()
            if self.Wc2.grad is not None: self.Wc2.grad.zero_()
            if self.bc1.grad is not None: self.bc1.grad.zero_()
            if self.bc2.grad is not None: self.bc2.grad.zero_()
            if self.U1.grad is not None: self.U1.grad.zero_()
            if self.U2.grad is not None: self.U2.grad.zero_()
            if self.c1.grad is not None: self.c1.grad.zero_()
            if self.c2.grad is not None: self.c2.grad.zero_()
            '''对实部反向传播'''
            ln_psi.conj().real.backward(retain_graph=True)
            # 收集梯度向量
            params_grad_real = torch.cat([
                self.Wu.grad.flatten() if self.Wu.grad is not None else torch.zeros_like(self.Wu.flatten()),
                self.Wr.grad.flatten() if self.Wr.grad is not None else torch.zeros_like(self.Wr.flatten()),
                self.bu.grad.flatten() if self.bu.grad is not None else torch.zeros_like(self.bu.flatten()),
                self.br.grad.flatten() if self.br.grad is not None else torch.zeros_like(self.br.flatten()),
                self.Wc1.grad.flatten() if self.Wc1.grad is not None else torch.zeros_like(self.Wc1.flatten()),
                self.Wc2.grad.flatten() if self.Wc2.grad is not None else torch.zeros_like(self.Wc2.flatten()),
                self.bc1.grad.flatten() if self.bc1.grad is not None else torch.zeros_like(self.bc1.flatten()),
                self.bc2.grad.flatten() if self.bc2.grad is not None else torch.zeros_like(self.bc2.flatten()),
                self.U1.grad.flatten() if self.U1.grad is not None else torch.zeros_like(self.U1.flatten()),
                self.U2.grad.flatten() if self.U2.grad is not None else torch.zeros_like(self.U2.flatten()),
                self.c1.grad.flatten() if self.c1.grad is not None else torch.zeros_like(self.c1.flatten()),
                self.c2.grad.flatten() if self.c2.grad is not None else torch.zeros_like(self.c2.flatten())
            ])

            '''计算虚部的梯度'''
            # 清空梯度
            if self.Wu.grad is not None: self.Wu.grad.zero_()
            if self.Wr.grad is not None: self.Wr.grad.zero_()
            if self.bu.grad is not None: self.bu.grad.zero_()
            if self.br.grad is not None: self.br.grad.zero_()
            if self.Wc1.grad is not None: self.Wc1.grad.zero_()
            if self.Wc2.grad is not None: self.Wc2.grad.zero_()
            if self.bc1.grad is not None: self.bc1.grad.zero_()
            if self.bc2.grad is not None: self.bc2.grad.zero_()
            if self.U1.grad is not None: self.U1.grad.zero_()
            if self.U2.grad is not None: self.U2.grad.zero_()
            if self.c1.grad is not None: self.c1.grad.zero_()
            if self.c2.grad is not None: self.c2.grad.zero_()
            '''对虚部反向传播'''
            ln_psi.conj().imag.backward()
            # 收集梯度向量
            params_grad_imag = torch.cat([
                self.Wu.grad.flatten() if self.Wu.grad is not None else torch.zeros_like(self.Wu.flatten()),
                self.Wr.grad.flatten() if self.Wr.grad is not None else torch.zeros_like(self.Wr.flatten()),
                self.bu.grad.flatten() if self.bu.grad is not None else torch.zeros_like(self.bu.flatten()),
                self.br.grad.flatten() if self.br.grad is not None else torch.zeros_like(self.br.flatten()),
                self.Wc1.grad.flatten() if self.Wc1.grad is not None else torch.zeros_like(self.Wc1.flatten()),
                self.Wc2.grad.flatten() if self.Wc2.grad is not None else torch.zeros_like(self.Wc2.flatten()),
                self.bc1.grad.flatten() if self.bc1.grad is not None else torch.zeros_like(self.bc1.flatten()),
                self.bc2.grad.flatten() if self.bc2.grad is not None else torch.zeros_like(self.bc2.flatten()),
                self.U1.grad.flatten() if self.U1.grad is not None else torch.zeros_like(self.U1.flatten()),
                self.U2.grad.flatten() if self.U2.grad is not None else torch.zeros_like(self.U2.flatten()),
                self.c1.grad.flatten() if self.c1.grad is not None else torch.zeros_like(self.c1.flatten()),
                self.c2.grad.flatten() if self.c2.grad is not None else torch.zeros_like(self.c2.flatten())
            ])

            '''将梯度储存,将局部能量储存'''
            partial_E[sample_idx] = params_grad_real + 1j * params_grad_imag

            # 直接写入，不会冲突（每个样本写入不同位置）
            all_E[sample_idx] = E_loc
            all_ln_psi[sample_idx] = ln_psi.conj()
            all_states[sample_idx] = state

            return sample_idx  # 返回索引用于验证

        # 最大化并行度（因为完全独立）
        max_workers = min(32, num_samples)  # 可以使用更多线程

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，无需等待顺序
            futures = [executor.submit(independent_worker, i)
                       for i in range(num_samples)]

            # 可选的完成验证
            completed_indices = [future.result() for future in futures]
            print(f"完成样本索引: {sorted(completed_indices)}")  # 验证完整性

        return all_E, all_ln_psi, all_states,partial_E

    def P_update(self):
        # 1. 采样
        E, ln_psi_conj, states,partial_E = self.independent_CRNN_sampling()
        E_mean = torch.mean(E)
        Delta_E = E - E_mean

        partial_E = 2 * torch.tensordot(partial_E, Delta_E, ([0], [0])).real / self.step
        '''参数的更新'''
        idx = 0
        params_list = [
            self.Wu, self.Wr, self.bu, self.br,
            self.Wc1, self.Wc2, self.bc1, self.bc2,
            self.U1, self.U2, self.c1, self.c2
        ]

        for param in params_list:
            size = param.numel()

            if param.grad is None:
                param.grad = torch.zeros_like(param)

            param_grad = partial_E[idx:idx + size].reshape(param.shape)
            param.grad.copy_(param_grad)

            idx += size

        # Adam更新
        self.optimizer.step()
        self.optimizer.zero_grad()
        return E_mean

    def learning(self):
        self.Wu.requires_grad_(True)
        self.Wr.requires_grad_(True)
        self.bu.requires_grad_(True)
        self.br.requires_grad_(True)
        self.Wc1.requires_grad_(True)
        self.Wc2.requires_grad_(True)
        self.bc1.requires_grad_(True)
        self.bc2.requires_grad_(True)
        self.U1.requires_grad_(True)
        self.U2.requires_grad_(True)
        self.c1.requires_grad_(True)
        self.c2.requires_grad_(True)
        for i in range(10000):
            E = self.P_update()
            print(f'第', i + 1, f'步的能每个格点能量为', E / self.L)


A = Complex_RNN(10)
A.learning()