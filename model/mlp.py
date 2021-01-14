import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPNetwork(nn.Module):
    def __init__(self, in_dim):
        super(MLPNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 64
        self.fc1 = nn.Sequential(
                        nn.Linear(self.in_dim, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
        self.fc3_1 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
        self.fc3_2 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
        self.mlp = MLP(in_dim=self.hidden_size)

    def forward(self, x_shot, x_query):
        x_all = torch.cat([x_shot, x_query], dim=0)
        x_all = self.fc1(x_all)
        feature_all = self.fc2(x_all)
        feature_shot, feature_query = feature_all[:len(x_shot)], feature_all[-len(x_query):]

        proto = feature_shot.reshape(5, 5, -1).mean(dim=1)
        mu = self.fc3_1(proto)
        logvar = self.fc3_2(proto)
        z = self.reparametrization(mu, logvar)
        return z, feature_shot, feature_query
    
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()).cuda() * std + mu
        z1 = torch.randn(std.size()).cuda() * std + mu
        z2 = torch.randn(std.size()).cuda() * std + mu
        z3 = torch.randn(std.size()).cuda() * std + mu
        z4 = torch.randn(std.size()).cuda() * std + mu
        z5 = torch.randn(std.size()).cuda() * std + mu
        z6 = torch.randn(std.size()).cuda() * std + mu
        z7 = torch.randn(std.size()).cuda() * std + mu
        z8 = torch.randn(std.size()).cuda() * std + mu
        z9 = torch.randn(std.size()).cuda() * std + mu
        z10 = torch.randn(std.size()).cuda() * std + mu
        z11 = torch.randn(std.size()).cuda() * std + mu
        z12 = torch.randn(std.size()).cuda() * std + mu
        z13 = torch.randn(std.size()).cuda() * std + mu
        z14 = torch.randn(std.size()).cuda() * std + mu
        z15 = torch.randn(std.size()).cuda() * std + mu
        z16 = torch.randn(std.size()).cuda() * std + mu
        z17 = torch.randn(std.size()).cuda() * std + mu
        z18 = torch.randn(std.size()).cuda() * std + mu
        z19 = torch.randn(std.size()).cuda() * std + mu
        z20 = torch.randn(std.size()).cuda() * std + mu
        z21 = torch.randn(std.size()).cuda() * std + mu
        z22 = torch.randn(std.size()).cuda() * std + mu
        z23 = torch.randn(std.size()).cuda() * std + mu
        z24 = torch.randn(std.size()).cuda() * std + mu
        z25 = torch.randn(std.size()).cuda() * std + mu
        z26 = torch.randn(std.size()).cuda() * std + mu
        z27 = torch.randn(std.size()).cuda() * std + mu
        z28 = torch.randn(std.size()).cuda() * std + mu
        z29 = torch.randn(std.size()).cuda() * std + mu
        return (z+z1+z2+z3+z4+z5+z6+z7+z8+z9 + z10+z11+z12+z13+z14+z15+z16+z17+z18+z19 + z20+z21+z22+z23+z24+z25+z26+z27+z28+z29) / 30.0
        
    
class MLP(nn.Module):
    
    def __init__(self, in_dim):
        super(MLP,self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 64
        self.fc1 = nn.Sequential(
                        nn.Linear(self.in_dim, self.hidden_size*2),
                        nn.BatchNorm1d(self.hidden_size*2),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size*4),
                        nn.BatchNorm1d(self.hidden_size*4),
                        nn.ReLU())
        self.fc3 = nn.Sequential(
                        nn.Linear(self.hidden_size*4, self.hidden_size*8),
                        nn.BatchNorm1d(self.hidden_size*8),
                        nn.ReLU())
    def forward(self, x):
       
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
        