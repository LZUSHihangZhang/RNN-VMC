import torch
import math
import random

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
        self.dh = 4
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
        self.c1 = torch.rand(self.dv, device=device1, dtype=dtype1)
        self.state0 = torch.rand(2, device=device1, dtype=dtype1)
        self.h0 = torch.rand(self.dh, device=device1, dtype=dtype1)
        self.step = 200
        self.param_count = 3 * self.dh ** 2 + 4 * self.dh * self.dv + 4 * self.dh +  self.dv+self.dh
        self.beta = 1
        '''设置虚时间步'''
        self.tau_A = 1
        self.tau_phi = 1
        self.m = self.tau_phi/self.tau_A
        self.tau = self.tau_A

    def Softsign(self, x):
        return x / (1 + torch.abs(x))

    '''计算Ising型的能量构型,即对角项'''

    def Energy_z(self, state):
        state_rolled_right = torch.roll(state, shifts=1, dims=0)
        E = torch.dot(state, state_rolled_right)
        return E / 4

    '''Gate单位,减小对平移不变形的破坏'''

    def Gate(self, state, h):
        pi1 = torch.cat((h, state), dim=0)  # 中间变量

        u_n = torch.sigmoid(torch.mv(self.Wu, pi1) + self.bu)
        r_n = torch.sigmoid(torch.mv(self.Wr, pi1) + self.br)

        h_n_1 = torch.mv(self.Wc1, h) + self.bc1
        h_n_2 = torch.tanh(torch.mv(self.Wc2, state) + r_n * h_n_1 + self.bc2)

        h_n = (1 - u_n) * h + u_n * h_n_2
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

                '''计算新构型的marshall_sign'''
                state1_even = state1[::2]
                marshall_sign = torch.prod(state1_even, dim=0).to(dtype2)


                # 计算右边格点自旋翻转后构型的概率
                for j in range(self.L):
                    h_next1 = self.Gate(self.sigma[int(state1[j].item())], h_next1)
                    y1R = torch.softmax(torch.mv(self.U1, h_next1) + self.c1, dim=0)

                    a = int(state1[j].item() != 1)
                    lnP_relate1 += torch.log(y1R[a])
                ln_psi1 = torch.log(marshall_sign) + 0.5 * lnP_relate1
                E_loc += torch.exp(ln_psi1 - ln_psi) * 0.5
        return E_loc

    def CRNN(self):
        state = torch.zeros(self.L, device=device1, dtype=dtype1)
        h_next = self.h0
        state_initial = self.state0.clone()
        ln_P = 0
        for i in range(self.L):
            h_next = self.Gate(state_initial, h_next)
            yn1 = torch.softmax(torch.mv(self.U1, h_next) + self.c1, dim=0)
            P1 = yn1[0]
            P2 = yn1[1]
            R = random.random()
            if R < P1:
                state[i] = 1
                state_initial = self.sigma1
                ln_P += torch.log(P1)
            else:
                state[i] = -1
                state_initial = self.sigma0
                ln_P += torch.log(P2)
        '''我们直接利用Marshall sign计算相位部分'''
        state_even = state[::2]
        marshall_sign = torch.prod(state_even, dim=0,dtype=dtype2)
        ln_psi = torch.log(marshall_sign) + 0.5 * ln_P  # 对于这个构型展开系数的对数值
        E_loc = self.Energy_z(state) + self.E_local(state, ln_psi)  # 定义局域能量
        return E_loc, ln_psi, state

    def learning(self):
        self.h0.requires_grad_(True)
        self.Wu.requires_grad_(True)
        self.Wr.requires_grad_(True)
        self.bu.requires_grad_(True)
        self.br.requires_grad_(True)
        self.Wc1.requires_grad_(True)
        self.Wc2.requires_grad_(True)
        self.bc1.requires_grad_(True)
        self.bc2.requires_grad_(True)
        self.U1.requires_grad_(True)
        self.c1.requires_grad_(True)


        for i in range(10000):
            '''分实部和虚部来计算梯度'''
            '''计算实部的梯度'''
            # 清空梯度
            if self.h0.grad is not None: self.h0.grad.zero_()
            if self.Wu.grad is not None: self.Wu.grad.zero_()
            if self.Wr.grad is not None: self.Wr.grad.zero_()
            if self.bu.grad is not None: self.bu.grad.zero_()
            if self.br.grad is not None: self.br.grad.zero_()
            if self.Wc1.grad is not None: self.Wc1.grad.zero_()
            if self.Wc2.grad is not None: self.Wc2.grad.zero_()
            if self.bc1.grad is not None: self.bc1.grad.zero_()
            if self.bc2.grad is not None: self.bc2.grad.zero_()
            if self.U1.grad is not None: self.U1.grad.zero_()
            if self.c1.grad is not None: self.c1.grad.zero_()
            E = torch.zeros(self.step, dtype=dtype1, device=device1)
            O = torch.zeros(self.step, self.param_count, dtype=dtype1, device=device1)
            k = 0
            while k<self.step:
                '''对实部反向传播'''
                E[k], ln_psi, state = self.CRNN()
                if torch.sum(state)!=0:
                    continue
                ln_psi.real.backward()
                # 收集梯度向量
                O[k, :] = torch.cat([
                    self.h0.grad.flatten() if self.h0.grad is not None else torch.zeros_like(self.h0.flatten()),
                    self.Wu.grad.flatten() if self.Wu.grad is not None else torch.zeros_like(self.Wu.flatten()),
                    self.Wr.grad.flatten() if self.Wr.grad is not None else torch.zeros_like(self.Wr.flatten()),
                    self.bu.grad.flatten() if self.bu.grad is not None else torch.zeros_like(self.bu.flatten()),
                    self.br.grad.flatten() if self.br.grad is not None else torch.zeros_like(self.br.flatten()),
                    self.Wc1.grad.flatten() if self.Wc1.grad is not None else torch.zeros_like(self.Wc1.flatten()),
                    self.Wc2.grad.flatten() if self.Wc2.grad is not None else torch.zeros_like(self.Wc2.flatten()),
                    self.bc1.grad.flatten() if self.bc1.grad is not None else torch.zeros_like(self.bc1.flatten()),
                    self.bc2.grad.flatten() if self.bc2.grad is not None else torch.zeros_like(self.bc2.flatten()),
                    self.U1.grad.flatten() if self.U1.grad is not None else torch.zeros_like(self.U1.flatten()),
                    self.c1.grad.flatten() if self.c1.grad is not None else torch.zeros_like(self.c1.flatten())
                ])
                k += 1
            O_mean = torch.mean(O, dim=0)
            bar_O = (O - O_mean) / math.sqrt(self.step)
            tilde_O = self.m * bar_O
            inv_matrix = torch.inverse(
                torch.mm(tilde_O, tilde_O.T.conj()) + self.beta * torch.eye(self.step, dtype=dtype1, device=device1))
            E_mean = torch.mean(E)
            epsilon = (-self.tau * (E - E_mean) / math.sqrt(self.step)).real
            gamma = torch.mv(tilde_O.T.conj(), torch.mv(inv_matrix, torch.mv(inv_matrix, torch.mv(tilde_O, torch.mv(
                bar_O.T.conj(), epsilon)))))
            with torch.no_grad():
                idx = 0
                h0_size = self.h0.numel()
                self.h0 += gamma[idx:idx + h0_size].reshape(self.h0.shape)
                idx += h0_size

                Wu_size = self.Wu.numel()
                self.Wu += gamma[idx:idx + Wu_size].reshape(self.Wu.shape)
                idx += Wu_size

                Wr_size = self.Wr.numel()
                self.Wr += gamma[idx:idx + Wr_size].reshape(self.Wr.shape)
                idx += Wr_size

                bu_size = self.bu.numel()
                self.bu += gamma[idx:idx + bu_size].reshape(self.bu.shape)
                idx += bu_size

                br_size = self.br.numel()
                self.br += gamma[idx:idx + br_size].reshape(self.br.shape)
                idx += br_size

                Wc1_size = self.Wc1.numel()
                self.Wc1 += gamma[idx:idx + Wc1_size].reshape(self.Wc1.shape)
                idx += Wc1_size

                Wc2_size = self.Wc2.numel()
                self.Wc2 += gamma[idx:idx + Wc2_size].reshape(self.Wc2.shape)
                idx +=  Wc2_size

                bc1_size = self.bc1.numel()
                self.bc1 += gamma[idx:idx + bc1_size].reshape(self.bc1.shape)
                idx += bc1_size

                bc2_size = self.bc2.numel()
                self.bc2 += gamma[idx:idx + bc2_size].reshape(self.bc2.shape)
                idx += bc2_size

                U1_size = self.U1.numel()
                self.U1 += gamma[idx:idx + U1_size].reshape(self.U1.shape)
                idx += U1_size

                c1_size = self.c1.numel()
                self.c1 += gamma[idx:idx + c1_size].reshape(self.c1.shape)
                idx += c1_size
            print('******************************************************************')
            print(f'第', i + 1, f'步的能每个格点能量为', torch.mean(E).item() / self.L)



A = Complex_RNN(4)
A.learning()