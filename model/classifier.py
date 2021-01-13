import math
import torch
import torch.nn as nn
import utils


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


class CosClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, metric='cos', temper=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temper is None:
            if metric == 'cos':
                temper = nn.Parameter(torch.tensor(10.))
            else:
                temper = 1.0
        self.metric = metric
        self.temper = temper

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temper)
