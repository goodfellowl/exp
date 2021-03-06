import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPNetwork(nn.Module):
    def __init__(self, in_dim):
        super(MLPNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 64

        self.layer1 = nn.Sequential(
                            nn.Conv2d(self.in_dim, self.hidden_size*4, 1, bias=False),
                            nn.BatchNorm2d(self.hidden_size*4),
                            nn.LeakyReLU(0.1))
        self.layer2 = nn.Sequential(
                            nn.Conv2d(self.hidden_size*4, self.hidden_size*2, 1, bias=False),
                            nn.BatchNorm2d(self.hidden_size*2),
                            nn.LeakyReLU(0.1))
        # self.fc1 = nn.Sequential(
        #                 nn.Linear(self.in_dim, self.hidden_size*2),
        #                 nn.BatchNorm1d(self.hidden_size*2),
        #                 nn.LeakyReLU(0.1))
        # self.fc2 = nn.Sequential(
        #                 nn.Linear(self.hidden_size*2, self.hidden_size),
        #                 nn.BatchNorm1d(self.hidden_size),
        #                 nn.LeakyReLU(0.1))
        self.fc3_1 = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.1))
        self.fc3_2 = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.1))
        self.decoder = Decoder(in_dim=self.hidden_size)

    def forward(self, x_shot, x_query):
        x_all = torch.cat([x_shot, x_query], dim=0)

        # x_all = self.fc1(x_all)
        # feature_all = self.fc2(x_all)

        x_all = self.layer1(x_all)
        feature_all = self.layer2(x_all)
        feature_all = feature_all.view(feature_all.shape[0], feature_all.shape[1], -1).mean(dim=2)

        feature_shot, feature_query = feature_all[:len(x_shot)], feature_all[-len(x_query):]
        proto = feature_shot.reshape(5, 5, -1).mean(dim=1)
        mu = self.fc3_1(proto)
        logvar = self.fc3_2(proto)
        z = self.reparametrization(mu, logvar)
        return self.decoder(z)
    
    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z0 = torch.randn(std.size()).cuda() * std + mu
        z1 = torch.randn(std.size()).cuda() * std + mu
        z2 = torch.randn(std.size()).cuda() * std + mu
        z3 = torch.randn(std.size()).cuda() * std + mu
        z4 = torch.randn(std.size()).cuda() * std + mu
        z5 = torch.randn(std.size()).cuda() * std + mu
        z6 = torch.randn(std.size()).cuda() * std + mu
        z7 = torch.randn(std.size()).cuda() * std + mu
        z8 = torch.randn(std.size()).cuda() * std + mu
        z9 = torch.randn(std.size()).cuda() * std + mu
        return (z0+z1+z2+z3+z4+z5+z6+z7+z8+z9 ) / 10.0
    
class Decoder(nn.Module):
    def __init__(self, in_dim):
        super(Decoder,self).__init__()
        self.in_dim = in_dim
        self.hidden_size = 128
        self.fc1 = nn.Sequential(
                        nn.Linear(self.in_dim, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size*2),
                        nn.BatchNorm1d(self.hidden_size*2),
                        nn.LeakyReLU(0.1))
        self.fc3 = nn.Sequential(
                        nn.Linear(self.hidden_size*2, self.hidden_size*4),
                        nn.BatchNorm1d(self.hidden_size*4),
                        nn.LeakyReLU(0.1))
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
        