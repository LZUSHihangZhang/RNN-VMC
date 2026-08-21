import torch


a = torch.tensor([1,2,4,2],dtype=torch.float64)
print(1-a)
b = a*1j
c = b+a
print(c.dtype)