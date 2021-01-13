import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropy(nn.Module):
    """
    Select the hard example through the loss.
    L = -(1-y_hat).pow(gamma)log(y_hat), y_hat : the pred followed softmax
    """
    def __init__(self, gamma=2):
        """
        gamma: exp of the weight
        """
        super().__init__()
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, target):
        """
        docstring
        """
        softmax_inp_hat = self.softmax(inp)
        log_softmax_inp_hat = self.log_softmax(inp)
        weight = (1-softmax_inp_hat).pow(self.gamma)
        weighted_inp = weight * log_softmax_inp_hat
        loss = self.nll_loss(weighted_inp, target)
        return loss

class CenterLoss(nn.Module):
    """
    docstring
    """
    def __init__(self, m1=0.25, m2=0.9):
        """
        docstring
        """
        super(CenterLoss,self).__init__()
        self.m1 = m1
        self.m2 = m2
    
    def forward(self, proto, x_shot, y_shot):
        """
        proto : nway * ndim
        x_shot ： nway*nshot*ndim
        """
        n_way = x_shot.shape[0] 
        n_shot = x_shot.shape[1]
        loss1 = torch.sum(F.relu(((F.normalize(x_shot, dim=-1) - F.normalize(proto.unsqueeze(1), dim=-1))**2).sum(-1) - self.m1)) / (n_way * n_shot * 1.0)
        proto_1 = F.normalize(torch.unsqueeze(proto,1), dim=-1)  # N*1*d
        proto_2 = F.normalize(torch.unsqueeze(proto,0), dim=-1)  # 1*N*d
        center_pair_dist = ((proto_1 - proto_2)**2).sum(-1)
        ones = torch.triu(torch.ones(n_way, n_way)).t().cuda()
        center_pair_dist = torch.triu(center_pair_dist) + ones
        loss2 = torch.sum(F.relu(self.m2 - center_pair_dist)) / 10.0

        x_shot_flat = x_shot.reshape(n_way*n_shot, -1)  # shape:(n_way*n_shot, -1)
        batch = []
        for c in range(n_way):
            ind = []
            for i, y in enumerate(y_shot):
                if int(y) != c:
                    ind.append(i)
            batch.append(x_shot_flat[torch.LongTensor(ind)])
        batch = torch.stack(batch).reshape(n_way, n_shot * (n_way-1), -1)
        loss3 = torch.sum(F.relu(0.4 - ((F.normalize(batch, dim=-1) - F.normalize(proto.unsqueeze(1), dim=-1))**2).sum(-1))) / (n_way * (n_way - 1) * n_shot * 1.0)
        return loss1 + loss2 + loss3