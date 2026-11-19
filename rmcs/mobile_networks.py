import torch
from torch import nn

class TinyAngleCNN(nn.Module):
    def __init__(self, angles=41):
        super().__init__()
        layers=[]
        for a,b in zip([1,8,16,24],[8,16,24,32]):
            layers += [nn.Conv2d(a,b,3,2,1,bias=False),nn.BatchNorm2d(b),nn.ReLU()]
        self.body=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,angles))

    def forward(self,x):
        return self.head(self.body(x))
