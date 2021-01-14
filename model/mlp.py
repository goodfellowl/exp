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
        out = self.fc2(out)
        mu = self.fc3_1(out)
        logvar = self.fc3_2(out)
        return self.reparametrization(mu, logvar), self.kld_loss(mu, logvar)
    
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()).cuda() * std + mu
        return z
    
    def kld_loss(self, mu, logvar):
        """
        docstring
        """
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kldloss = torch.sum(kld).mul_(-0.5)
        return kldloss