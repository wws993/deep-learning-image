import torch.nn as nn
import torch


class Lmodel(nn.Module):
        def __init__(self,input_dims):
            super(Lmodel,self).__init__()
            self.layers=nn.Sequential(
                  nn.Linear(input_dims,16),
                  nn.ReLU(),
                  nn.Linear(16,8),
                  nn.ReLU(),
                  nn.Linear(8,1)
            )
        def forward(self,x):
            x=self.layers(x)
            x=x.squeeze(1)
            return x