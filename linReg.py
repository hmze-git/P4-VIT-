import torch
from torch import nn

class LinearReg(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(1,dtype=torch.float,requires_grad=True))
        self.bias=nn.Parameter(torch.randn(1,dtype=torch.float,requires_grad=True))

        #specify datatype for readability using :torch.Tensor
        #specify return type with ->torch.tensor
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.weights*x+self.bias