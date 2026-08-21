import torch
import math
import random

dtype1 = torch.float64
dtype2 = torch.complex128
device1 = torch.device('cpu')
device2 = torch.device('cuda')

class Complex_RNN:
    def __init__(self,L):
        self.L = L
        self.sigma1 = torch.tensor([1, 0], dtype=dtype1, device=device1)
        self.sigma0 = torch.tensor([0, 1], dtype=dtype1, device=device1)
        self.sigma = torch.stack([self.sigma1, self.sigma0], dim=0)
        self.dh = 2
        self.dv = 2
        self.Wu = torch.rand((self.dh,self.dh+self.dv), device = device1, dtype=dtype1)
        self.Wr = torch.rand((self.dh,self.dh+self.dv), device = device1, dtype=dtype1)
        self.bu = torch.rand(self.dh, device = device1, dtype=dtype1)
        self.br = torch.rand(self.dh, device = device1, dtype=dtype1)
        self.Wc1 = torch.rand((self.dh,self.dh), device = device1, dtype=dtype1)
        self.Wc2 = torch.rand((self.dh,self.dv), device = device1, dtype=dtype1)
        self.bc1 = torch.rand(self.dh, device = device1, dtype=dtype1)
        self.bc2 = torch.rand(self.dh, device = device1, dtype=dtype1)
        self.U1 = torch.rand((self.dv,self.dh), device = device1, dtype=dtype1)
        self.U2 = torch.rand((self.dv,self.dh), device = device1, dtype=dtype1)
        self.c1 = torch.rand(self.dv, device = device1, dtype=dtype1)
        self.c2 = torch.rand(self.dv, device = device1, dtype=dtype1)
        self.state0 = torch.rand(2, device=device1, dtype=dtype1)
        self.h0 = torch.ones(self.dh, device=device1, dtype=dtype1)
        self.step = 100
        self.alpha = 0.1


    def Softsign(self,x):
        return x/(1+torch.abs(x))

    '''计算Ising型的能量构型,即对角项'''
    def Energy_z(self,state):
        state_rolled_right = torch.roll(state, shifts=1, dims=0)
        E = torch.dot(state, state_rolled_right)
        return E /4

    '''Gate单位,减小对平移不变形的破坏'''
    def Gate(self,state,h):
        pi1 = torch.cat((h, state), dim=0)#中间变量

        u_n = torch.sigmoid(torch.mv(self.Wu,pi1)+self.bu)
        r_n = torch.sigmoid(torch.mv(self.Wr,pi1)+self.br)

        h_n_1 = torch.mv(self.Wc1,h)+self.bc1
        h_n_2 = torch.tanh(torch.mv(self.Wc2,state)+r_n*h_n_1+self.bc2)

        h_n = (1-u_n)*h + u_n*h_n_2
        return h_n

    '''计算与state相关的局部能量'''
    def E_local(self,state,ln_psi):
        E_loc = 0+0j
        '''计算局域能量'''
        for i in range(self.L):
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
            ln_psi1 = 1j*phi1 + 0.5*lnP_relate1
            E_loc += torch.exp(ln_psi1 - ln_psi) * 0.5
        return E_loc

    def CRNN(self):
        state = torch.zeros(self.L,device = device1, dtype=dtype1)
        h_next = self.h0
        state_initial = self.state0.clone()
        ln_P = 0
        phi = 0
        for i in range(self.L):
            h_next = self.Gate(state_initial,h_next)
            yn1 = torch.softmax(torch.mv(self.U1,h_next)+self.c1,dim=0)
            yn2 = math.pi*self.Softsign(torch.mv(self.U2,h_next)+self.c2)
            P1 = yn1[0]
            P2 = yn1[1]
            R = random.random()
            if R<P1:
                state[i] = 1
                state_initial = self.sigma1
                ln_P += torch.log(P1)
                phi += torch.dot(yn2,self.sigma1)
            else:
                state[i] = -1
                state_initial = self.sigma0
                ln_P += torch.log(P2)
                phi += torch.dot(yn2,self.sigma0)

        ln_psi = 1j*phi+0.5*ln_P#对于这个构型展开系数的对数值
        E_loc = self.Energy_z(state) + self.E_local(state,ln_psi)#定义局域能量
        return E_loc,ln_psi,state


    def P_update(self):
        param_count = 3*self.dh**2 + 5*self.dh*self.dv + 4*self.dh + 2*self.dv
        partial_E = torch.zeros(self.step,param_count,dtype=dtype2,device=device1)
        E = torch.zeros(self.step,dtype=dtype2,device=device1)
        for i in range(self.step):
            E_loc,ln_psi,state = self.CRNN()

            '''分实部和虚部来计算梯度'''
            '''计算实部的梯度'''
            # 清空梯度
            if self.Wu.grad is not None:self.Wu.grad.zero_()
            if self.Wr.grad is not None:self.Wr.grad.zero_()
            if self.bu.grad is not None:self.bu.grad.zero_()
            if self.br.grad is not None:self.br.grad.zero_()
            if self.Wc1.grad is not None:self.Wc1.grad.zero_()
            if self.Wc2.grad is not None:self.Wc2.grad.zero_()
            if self.bc1.grad is not None:self.bc1.grad.zero_()
            if self.bc2.grad is not None:self.bc2.grad.zero_()
            if self.U1.grad is not None:self.U1.grad.zero_()
            if self.U2.grad is not None:self.U2.grad.zero_()
            if self.c1.grad is not None:self.c1.grad.zero_()
            if self.c2.grad is not None:self.c2.grad.zero_()
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
            if self.Wu.grad is not None:self.Wu.grad.zero_()
            if self.Wr.grad is not None:self.Wr.grad.zero_()
            if self.bu.grad is not None:self.bu.grad.zero_()
            if self.br.grad is not None:self.br.grad.zero_()
            if self.Wc1.grad is not None:self.Wc1.grad.zero_()
            if self.Wc2.grad is not None:self.Wc2.grad.zero_()
            if self.bc1.grad is not None:self.bc1.grad.zero_()
            if self.bc2.grad is not None:self.bc2.grad.zero_()
            if self.U1.grad is not None:self.U1.grad.zero_()
            if self.U2.grad is not None:self.U2.grad.zero_()
            if self.c1.grad is not None:self.c1.grad.zero_()
            if self.c2.grad is not None:self.c2.grad.zero_()
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
            partial_E[i] = params_grad_real+1j*params_grad_imag
            E[i] = E_loc
        E_mean = torch.mean(E)

        partial_E = self.alpha*2*torch.tensordot(partial_E,(E-E_mean),([0],[0])).real/self.step

        '''参数的更新'''
        idx = 0
        '''self.Wu的更新'''
        wu_size = self.Wu.numel()
        self.Wu.data -= partial_E[idx:idx+wu_size].reshape(self.dh,self.dh+self.dv)
        idx += wu_size

        '''self.Wr的更新'''
        wr_size = self.Wr.numel()
        self.Wr.data -= partial_E[idx:idx + wr_size].reshape(self.dh,self.dh+self.dv)
        idx += wr_size

        '''self.bu的更新'''
        bu_size = self.bu.numel()
        self.bu.data -= partial_E[idx:idx + bu_size].reshape(self.dh)
        idx += bu_size

        '''self.br的更新'''
        br_size = self.br.numel()
        self.br.data -= partial_E[idx:idx + br_size].reshape(self.dh)
        idx += br_size

        '''self.Wc1的更新'''
        wc1_size = self.Wc1.numel()
        self.Wc1.data -= partial_E[idx:idx + wc1_size].reshape(self.dh,self.dh)
        idx += wc1_size

        '''self.Wc2的更新'''
        wc2_size = self.Wc2.numel()
        self.Wc2.data -= partial_E[idx:idx + wc2_size].reshape(self.dh,self.dv)
        idx += wc2_size

        '''self.bc1的更新'''
        bc1_size = self.bc1.numel()
        self.bc1.data -= partial_E[idx:idx + bc1_size].reshape(self.dh)
        idx += bc1_size

        '''self.bc2的更新'''
        bc2_size = self.bc2.numel()
        self.bc2.data -= partial_E[idx:idx + bc2_size].reshape(self.dh)
        idx += bc2_size

        '''self.U1的更新'''
        U1_size = self.U1.numel()
        self.U1.data -= partial_E[idx:idx + U1_size].reshape(self.dv,self.dh)
        idx += U1_size

        '''self.U2的更新'''
        U2_size = self.U2.numel()
        self.U2.data -= partial_E[idx:idx + U2_size].reshape(self.dv, self.dh)
        idx += U2_size

        '''self.c1的更新'''
        c1_size = self.c1.numel()
        self.c1.data -= partial_E[idx:idx + c1_size].reshape(self.dv)
        idx += c1_size

        '''self.c2的更新'''
        c2_size = self.c2.numel()
        self.c2.data -= partial_E[idx:idx + c2_size].reshape(self.dv)
        idx += c2_size
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
        for i in range(1000):
            E = self.P_update()
            print(f'第',i+1,f'步的能每个格点能量为',E/self.L)


A = Complex_RNN(20)
A.learning()
