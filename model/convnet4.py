import torch
import numpy as np
import torch.nn.functional as F

class Conv4(torch.nn.Module):
    def __init__(self, avg_pool=True):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.avg_pool = avg_pool
        if self.avg_pool == True:
            self.out_dim = 64
        else:
            self.out_dim = 1600
    
    def forward(self, x):
        batch, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.conv0(x) # 84
        x = F.relu(self.bn0(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 84 -> 42
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 42 -> 21
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 10
        x = self.conv3(x)
        x = F.relu(self.bn3(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 5
        if self.avg_pool == True:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        return x.view(batch, self.out_dim)
