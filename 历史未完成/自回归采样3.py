import torch
import random
import math

device = torch.device('cpu')
type = torch.float64
class VMC:
    def __init__(self,N):
        self.sigma1 = torch.tensor([1,0],dtype=type,device=device)
        self.sigma0 = torch.tensor([0,1],dtype=type,device=device)
        self.dv = 2
        self.dh = 5
        self.W = torch.rand(self.dh, self.dh + self.dv, device=device, dtype=type)
        self.U = torch.rand(self.dv, self.dh, device=device, dtype=type)
        self.c = torch.zeros(self.dv, device=device, dtype=type)  # 让初始P1=P2=0.5
        self.b = torch.zeros(self.dh, device=device, dtype=type)
        self.N = N
        self.P = torch.tensor([[0,1],[1,0]],device=device,dtype=type)
        self.W_size = self.dh * (self.dh + self.dv)
        self.b_size = self.dh
        self.U_size = self.dv*self.dh
        self.c_size = self.dv
        self.alpha_size = self.W_size+self.b_size+self.U_size+self.c_size
        self.mu = 1
        self.step = 100
        self.h = 1
        self.beta = 1

    def Ising(self,state):
        state_rolled_right = torch.roll(state, shifts=1, dims=0)
        state_rolled_left = torch.roll(state, shifts=-1, dims=0)
        E = torch.dot(state,state_rolled_right)+torch.dot(state,state_rolled_left)
        return E/2

    def RNN(self):
        # W=d_h*(d_h+d_v)
        # b=d_h*1
        # h_n = f(W[h_{n-1};\sigma_{n-1}]+b)
        # f = x,x>0;exp(x)-1,x<0
        # P_n = y_n*\sigma_n
        # S(v_n) = exp(v_n)/\sum_i exp(v_i)
        # y_n = S(Uh_n+c)
        # P(\sigma) = \Pi_{n-1}^N y_n*\sigma_n
        # U=d_v*d_h
        # c=d_v
        # 使用列表来存储隐藏状态，避免inplace操作
        h_next = torch.ones(self.dh, device=device, dtype=type)
        state = torch.zeros(self.N, device=device, dtype=type)
        state0 = torch.rand(2,device=device, dtype=type)
        log_P = 0
        for i in range(self.N):
            pi = torch.cat((h_next, state0), dim=0)
            h_next = torch.nn.functional.elu(torch.mv(self.W, pi) + self.b)
            y = torch.softmax(torch.mv(self.U, h_next) + self.c, dim=0)
            P1 = y[0]
            P2 = y[1]
            R = random.random()
            if R < P1:#如果R小于随机数,选择1
                state[i] = 1
                state0 = self.sigma1
                log_P += torch.log(P1)
            else:#如果R大于随机数,选择2
                state[i] = -1
                state0 = self.sigma0
                log_P += torch.log(P2)

        E = self.Ising(state)
        return log_P,state,E

    def StochasticConfiguration(self):
        E = []
        all_grads = []  # 储存梯度
        for i in range(self.step):
            # 现在记录一个样本（经过L步后）
            # 清空梯度
            if self.W.grad is not None: self.W.grad.zero_()
            if self.b.grad is not None: self.b.grad.zero_()
            if self.U.grad is not None: self.U.grad.zero_()
            if self.c.grad is not None: self.c.grad.zero_()

            # 计算当前状态的梯度和概率
            log_P, state, Energy = self.RNN()
            ln_sqrt_psi = 0.5*log_P
            ln_sqrt_psi.backward()

            # 收集梯度
            grad_vec = torch.cat([
                self.W.grad.flatten() if self.W.grad is not None else torch.zeros_like(self.W.flatten()),
                self.b.grad.flatten() if self.b.grad is not None else torch.zeros_like(self.b.flatten()),
                self.U.grad.flatten() if self.U.grad is not None else torch.zeros_like(self.U.flatten()),
                self.c.grad.flatten() if self.c.grad is not None else torch.zeros_like(self.c.flatten())
            ])

            all_grads.append(grad_vec)
            E.append(Energy)

        O = torch.stack(all_grads)
        E = torch.stack(E)

        print(f'所有抽样的构型的能量', E)
        M = len(E)  # M = 100

        # 计算平均值和涨落
        mean_O = O.mean(dim=0)
        mean_E = E.mean()
        print("能量", (mean_E/self.N).item())
        delta_O = O - mean_O
        delta_E = E - mean_E

        '''检验程序是否终止'''
        for grad_vec in all_grads:
            if torch.all(grad_vec == torch.zeros_like(grad_vec)):
                print("梯度全为0，终止程序")
                import sys
                sys.exit(1)

        f = -2 * torch.tensordot(delta_E, delta_O, dims=([0], [0])) / M
        S = torch.tensordot(delta_O, delta_O, dims=([0], [0])) / M
        G = torch.zeros((self.alpha_size, self.alpha_size), device=device, dtype=type)
        for i in range(M):
            G += delta_E[i] * torch.outer(delta_O[i,:], delta_O[i,:])
        G = 2 * G / M
        gamma = torch.zeros(self.alpha_size, device=device, dtype=type)
        # 尝试不同的mu值
        for mu_try in [0.1,1.0, 10.0, 100.0, 1000.0,10000.0]:
            try:
                A = (self.beta+1)*G + mu_try * S
                gamma = torch.linalg.solve(A, f)
                print("*" * 100)  # 输出50个星号
                print(f"使用 mu = {mu_try} 成功求解")
                self.mu = mu_try  # 更新mu值
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            print("所有mu值都失败，使用伪逆")
            A = (self.beta+1)*G + 10000.0 * S  # 使用最大的mu
            gamma = torch.linalg.pinv(A) @ f

        with torch.no_grad():
            idx = 0
            w_size = self.W.numel()
            self.W += gamma[idx:idx + w_size].reshape(self.W.shape)
            idx += w_size
            b_size = self.b.numel()
            self.b += gamma[idx:idx + b_size].reshape(self.b.shape)
            idx += b_size
            u_size = self.U.numel()
            self.U += gamma[idx:idx + u_size].reshape(self.U.shape)
            idx += u_size
            c_size = self.c.numel()
            self.c += gamma[idx:idx + c_size].reshape(self.c.shape)
        return gamma

    def learning(self):
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        self.U.requires_grad_(True)
        self.c.requires_grad_(True)
        for i in range(1000):
            gamma = self.StochasticConfiguration()
            gamma0 = torch.linalg.norm(gamma)
            if self.W.grad is not None: self.W.grad.zero_()
            if self.b.grad is not None: self.b.grad.zero_()
            if self.U.grad is not None: self.U.grad.zero_()
            if self.c.grad is not None: self.c.grad.zero_()
            '''这时候梯度过大,重置程序'''
            if gamma0 > 1e5 or math.isnan(gamma0):  # 重置参数
                self.W = torch.rand(self.dh, self.dh + self.dv, device=device, dtype=type)
                self.b = torch.rand(self.dh, device=device, dtype=type)
                self.U = torch.rand(self.dv, self.dh, device=device, dtype=type)
                self.c = torch.rand(self.dv, device=device, dtype=type)
                self.W.requires_grad_(True)
                self.b.requires_grad_(True)
                self.U.requires_grad_(True)
                self.c.requires_grad_(True)

A = VMC(10)
A.learning()
