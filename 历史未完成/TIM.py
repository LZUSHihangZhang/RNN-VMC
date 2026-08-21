import torch
import random
import itertools
device = torch.device('cpu')
type = torch.float64
class VMC:
    def __init__(self,N):
        self.sigma1 = torch.tensor([1,0],dtype=type)
        self.sigma0 = torch.tensor([0,1],dtype=type)
        self.N = N
        self.Jx = 1
        dh = 2
        dv = 2
        self.W = torch.rand(dh, dh + dv, device=device, dtype=type)
        self.b = torch.rand(dh, device=device, dtype=type)
        self.U = torch.rand(dv, dh, device=device, dtype=type)
        self.c = torch.rand(dv, device=device, dtype=type)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        self.U.requires_grad_(True)
        self.c.requires_grad_(True)


    def generate_binary_numbers(self):
        # 生成 2^n 个长度为 n 的向量组成的二维数组
        return torch.tensor([list(bits) for bits in itertools.product([0, 1], repeat=self.N)], dtype=type,device=device)

    def second_number_compare(self, a, b):
        for i in range(self.N):
            if a[i] > b[i]:
                return 1
            elif a[i] < b[i]:
                return 0

            elif i == self.N - 1 and a[i] == b[i]:
                return 2

            elif a[i] == b[i]:
                continue

    def find_state(self, S, all_state):
        b_min = 0
        b_max = self.N - 1
        while b_max != b_min:
            b = int((b_min + b_max) / 2)
            k = self.second_number_compare(all_state[b], S)
            if k == 1:
                b_max = b - 1
            elif k == 0:
                b_min = b + 1
            elif k == 2:
                b_max = b
                break
        return b_max


    def randstate(self):
        state = torch.zeros((2,self.N),dtype=type,device=device)
        for i in range(self.N):
            a = random.random()
            if a < 0.5:
                state[:,i] = self.sigma0
            else:
                state[:,i] = self.sigma1
        return state

    def Ising_z(self,all_state,P):
        E = 0
        for i,a in enumerate(all_state):
            state1 = (a-0.5)*2
            state_rolled = torch.roll(state1, shifts=1, dims=0)
            E += torch.dot(state1, state_rolled)*P[i]
        return E

    def Ising_x(self,all_state,P):
        E = 0
        for i,a in enumerate(all_state):
            state1 = a
            for j in range(self.N):
                state2 = a
                state2[j] = (1+(-1)**state1[j])/2
                b = self.find_state(state2,all_state)
                E += torch.sqrt(P[i]*P[b])*self.Jx
        return E


    def Ising(self,all_state,P):
        E1 = self.Ising_z(all_state,P)
        E2 = self.Ising_x(all_state,P)
        return E1+E2



    def RNN(self, state):
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
        dh = 2
        # 使用列表来存储隐藏状态，避免inplace操作
        h_0 = torch.ones(dh, device=device, dtype=type)
        P = 1

        for i in range(self.N):
            pi = torch.cat((h_0, state[i,:]), dim=0)
            h_next = torch.nn.functional.elu(torch.mv(self.W, pi) + self.b)
            y = torch.softmax(torch.mv(self.U, h_next) + self.c, dim=0)
            P *= torch.dot(y, state[i,:].to(type))
            h_0 = h_next

        return P

    def compute_all_state(self,all_state):
        n = all_state.shape[0]
        one_hot_tensors = torch.nn.functional.one_hot(all_state.long(), num_classes=2)
        P_tensors = torch.zeros(n,device=device,dtype=type)
        for i in range(all_state.shape[0]):
            P_tensors[i] = self.RNN(one_hot_tensors[i])
        return P_tensors



    def learning(self):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam([self.W, self.b, self.U, self.c], lr=0.01)
        num_epochs = 10000

        all_state = self.generate_binary_numbers()


        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # 计算所有构型的概率 P(σ)
            probabilities = self.compute_all_state(all_state)
            # 检查概率是否已经归一化
            prob_sum = torch.sum(probabilities)
            probabilities = probabilities/prob_sum
            # 计算能量期望值: ⟨E⟩ = Σ P(σ) E(σ)
            Loss = self.Ising(all_state,probabilities)
            Loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Energy: {Loss.item()/self.N:.6f}, Prob Sum: {torch.sum(probabilities):.6f}")
                print(probabilities)

A = VMC(5)
A.learning()

