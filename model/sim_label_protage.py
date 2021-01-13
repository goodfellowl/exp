import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimLabel(nn.Module):
    """
    docstring
    """
    def __init__(self, topk=5):
        """
        docstring
        """
        super(SimLabel,self).__init__()
        self.topk = topk
    
    def forward(self, support, query):
        """
        docstring
        """
        n_way = 5
        n_shot = int(support.shape[0] / n_way)
        n_query = int(query.shape[0] / n_way)

        cos_dist = torch.mm(F.normalize(query, dim=-1), F.normalize(support, dim=-1).t())
        # topk, indices = torch.topk(cos_dist, self.topk)
        # mask = torch.zeros_like(cos_dist)
        # mask = mask.scatter(1, indices, topk)
        # mask_cos_dist = mask
        c1, c2, c3, c4, c5 = torch.split(cos_dist, [n_shot, n_shot, n_shot, n_shot, n_shot], 1)
        c1 = c1.sum(-1);c2 = c2.sum(-1);c3 = c3.sum(-1);c4 = c4.sum(-1);c5 = c5.sum(-1)
        logits = torch.stack([c1, c2, c3, c4, c5], 1)
        
        return logits
        


