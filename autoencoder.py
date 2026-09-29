from torch import nn
class Autoencoder(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,8),nn.ReLU(),nn.Linear(8,32),nn.ReLU(),nn.Linear(32,n))
 def forward(self,x): return self.net(x)
