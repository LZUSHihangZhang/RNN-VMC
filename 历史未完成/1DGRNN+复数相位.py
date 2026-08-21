import torch

class RNN_Cell:
    def __init__(self,N):
        self.device = torch.device('cuda')
        self.dtype = torch.complex64
        self.N = N
        self.sigma0 = torch.tensor([1,0],dtype=self.dtype,device=self.device)
        self.sigma1 = torch.tensor([0,1],dtype=self.dtype,device=self.device)
        self.dh = 2
        self.dv = 2
        self.W_size = self.dh * (self.dh + self.dv)
        self.b_size = self.dh
        self.U_size = self.dv * self.dh
        self.c_size = self.dv
        self.Wu = torch.rand((self.dh,self.dh*self.dv),device=self.device,dtype=self.dtype)
        self.Wr = torch.rand((self.dh,self.dh*self.dv),device=self.device,dtype=self.dtype)
        self.Wc1 = torch.rand(self.dh,self.dv,device=self.device,dtype=self.dtype)
        self.Wc2 = torch.rand(self.dh,self.dv,device=self.device,dtype=self.dtype)
        self.bc1 = torch.rand(self.dh)
        self.bc2 = torch.rand(self.dh)
        self.bu = torch.rand(self.dh,device=self.device,dtype=self.dtype)
        self.br = torch.rand(self.dh,device=self.device,dtype=self.dtype)
        self.U1 = torch.rand((self.dv,self.dh),device=self.device,dtype=self.dtype)
        self.U2 = torch.rand((self.dv,self.dh),device=self.device,dtype=self.dtype)
        self.c1 = torch.rand(self.dv,dtype=self.dtype,device=self.device)
        self.c2 = torch.rand(self.dv,dtype=self.dtype,device=self.device)
        self.h0 = torch.tensor((1,0),dtype=self.dtype,device=self.device)


    def RNN(self,state):
        h_0 = self.h0
        P = 1
        phi = 0
        for i in range(self.N):
            pi = torch.cat((h_0, state[:, i]), dim=0)
            un = torch.sigmoid(torch.mv(self.Wu,pi)+self.bu)
            rn = torch.sigmoid(torch.mv(self.Wr,pi)+self.br)
            h_n1 = torch.mv(self.Wc1,self.h0)+self.bc1
            h_n_tilde = torch.tanh(torch.mv(self.Wc2,state[:,i])+rn*h_n1+self.bc2)
            h_n = (1-un)*h_0 + un*h_n_tilde
            y1 = torch.softmax(torch.mv(self.U1,h_n)+self.c1)
            y2 = torch.pi*torch.softmax(torch.mv(self.U2,h_n)+self.c2)
            P *= torch.dot(y1,state[:,i])
            phi += torch.dot(y2,state[:,i])
            h_0 = h_n
        return P,phi



