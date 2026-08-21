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
        self.dh = 8
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
        self.tau = 1

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
            if R < P1:#如果R小于P1,选择1
                state[i] = 1
                state0 = self.sigma1
                log_P += torch.log(P1)
            else:#如果R大于P2,选择2
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
            ln_sqrt_psi = -0.5*log_P
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
        # O 的形状应为 (M, p)， E 的形状应为 (M,)
        M = self.step  # 样本数
        p = self.alpha_size  # 参数个数

        # 计算均值
        mean_O = O.mean(dim=0)  # 形状 (p,)
        mean_E = E.mean()  # 标量

        # *** 关键修正1：按照论文定义构造 Ō 和 ε̄ ***
        # 注意：这里使用论文中的 τ（self.tau）作为 δτ
        delta_tau = self.tau

        # Ō: 每个元素减去均值后，再除以 sqrt(M)
        Ō = (O - mean_O) / math.sqrt(M)  # 形状 (M, p)

        # ε̄: 根据方程(7)和上下文定义
        ε̄ = -delta_tau * (E - mean_E) / math.sqrt(M)  # 形状 (M,)

        # *** 关键修正2：直接构造并求解论文中的线性系统 ***
        # 方程对应: (Ō^T Ō + β I) δθ = Ō^T ε̄
        # 左边矩阵 A = Ō^T Ō + β I, 形状 (p, p)
        # 右边向量 b = Ō^T ε̄, 形状 (p,)

        Ō_T = Ō.T  # 转置，形状 (p, M)
        A = torch.mm(Ō_T, Ō)  # Ō^T Ō, 形状 (p, p)
        b = torch.mv(Ō_T, ε̄)  # Ō^T ε̄, 形状 (p,)

        I = torch.eye(p, device=device, dtype=type)
        gamma = None

        # 尝试不同的正则化参数 β
        for beta_try in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
            try:
                A_reg = A + beta_try * I
                gamma = torch.linalg.solve(A_reg, b)  # 求解 δθ
                print(f"使用 β = {beta_try} 成功求解")
                print(f"更新步长 |γ| = {torch.norm(gamma).item():.6f}")
                break
            except torch.linalg.LinAlgError:
                continue

        if gamma is None:
            print("所有β值都失败，使用伪逆")
            gamma = torch.linalg.pinv(A + self.beta * I) @ b

        # *** 关键修正3：检查更新方向（可选但推荐）***
        # 计算梯度向量 f = Ō^T ε̄ （即 b）
        # 计算 f·γ，在梯度下降中通常期望它为负值（表示能量下降方向）
        dot_product = torch.dot(b, gamma).item()
        print(f"梯度与更新方向的点积 f·γ = {dot_product:.6f}")
        if dot_product > 0:
            print("警告：点积为正，本次更新可能不会降低能量。")

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

            if gamma0 < 1e-10:
                print('梯度过小,停止优化')
                break

A = VMC(10)
A.learning()
