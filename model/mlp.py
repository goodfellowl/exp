import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self, in_dim):
        super(RelationNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 64
        self.fc1 = nn.Sequential(
                        nn.Linear(self.in_dim, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU())
        self.fc2 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm2d(self.hidden_size),
                        nn.ReLU())
        self.fc3_1 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm2d(self.hidden_size),
                        nn.ReLU())
        self.fc3_2 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm2d(self.hidden_size),
                        nn.ReLU())
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out_mu = self.fc3_1(out)
        out_sigma = self.fc3_2(out)
        return out_mu,out_sigma