import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, n_classes, in_dim):
        super().__init__()
        self.vars = nn.ParameterList()
        self.n_classes = n_classes
        self.in_dim = in_dim
        
        self.fc1_w = nn.Parameter(torch.ones([self.n_classes, self.in_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.n_classes))
        self.vars.append(self.fc1_b)

    def forward(self, inp, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        logits = F.linear(F.normalize(inp, dim=-1), F.normalize(fc1_w, dim=-1), fc1_b)
        return logits

    def parameters(self):
        return self.vars

    def setfc1_w(self, proto):
        self.fc1_w = nn.Parameter(proto)

if __name__ == "__main__":

    # self.inner_learner.setfc1_w(proto)
    # logits = self.inner_learner(x_shot)
    # loss = F.cross_entropy(logits, y_shot)
    # grad = torch.autograd.grad(loss, self.inner_learner.parameters()) 
    # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.inner_learner.parameters())))
    
    # for _ in range(1, 100):
    #     logits = self.inner_learner(x_shot, fast_weights)
    #     loss = F.cross_entropy(logits, y_shot)
    #     grad = torch.autograd.grad(loss, self.inner_learner.parameters())
    #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.inner_learner.parameters())))

    # logits_q = self.inner_learner(x_query, fast_weights)        
    
    # query_proto = []
    # for c in range(n_way):
    #     ind = []
    #     for i, y in enumerate(pesudo_q_label):
    #         if int(y) == c:
    #             ind.append(i)
    #     query_proto.append(x_query[torch.LongTensor(ind)].mean(dim=0))
    # query_proto = torch.stack(query_proto)