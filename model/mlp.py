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

    def forward(self, x):
        out = self.fc1(x)
        feature = self.fc2(out)
        proto = feature.reshape(5, 5, -1).mean(dim=1)
        mu = self.fc3_1(proto)
        logvar = self.fc3_2(proto)
        z = self.reparametrization(mu, logvar)
        return z, feature
    
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()).cuda() * std + mu
        return z
    

class MLP(nn.Module):
    
    def __init__(self, in_dim):
        super(MLP,self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 64
        self.fc1 = nn.Sequential(
                        nn.Linear(self.in_dim, self.hidden_size*4),
                        nn.BatchNorm1d(self.hidden_size*4),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(self.hidden_size*4, self.hidden_size*2),
                        nn.BatchNorm1d(self.hidden_size*2),
                        nn.ReLU())
        self.fc3 = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
    def forward(self, x):
       
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
        