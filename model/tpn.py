import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, in_dim, topk=10):
        super(LabelPropagation, self).__init__()
        self.topK = topk
        self.in_dim = in_dim

        self.conv_q = nn.Conv2d(self.in_dim, self.in_dim, 1, bias=False)
        self.conv_v = nn.Conv2d(self.in_dim, self.in_dim, 1, bias=False)
       

    def forward(self, support, s_labels, query, q_labels):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x d
            query:      (N_way*N_query)x d
            s_labels:   (N_way*N_shot)
            q_labels:   (N_way*N_query)
        """
        n_way = 5
        n_shot = int(support.shape[0] / n_way)
        n_query = int(query.shape[0] / n_way)

        emb_all = torch.cat([support, query], dim=0)

        emb_q = self.conv_q(emb_all)
        emb_q = emb_q.view(emb_q.shape[0], emb_q.shape[1], -1).mean(-1)
       
        emb_v = self.conv_v(emb_all)
        emb_v = emb_v.view(emb_v.shape[0], emb_v.shape[1], -1).mean(-1)
        
        # Step1: Graph Construction，generate weights(N*N)
        eps = np.finfo(float).eps
        
        emb1 = torch.unsqueeze(emb_q,1) # N*1*d
        emb2 = torch.unsqueeze(emb_q,0) # 1*N*d
        weights = ((emb1-emb2)**2).sum(-1)   # N*N*d -> N*N
        mask = weights != 0
        weights = weights / weights[mask].std()
        weights = torch.exp(-weights/2)
            
        ## keep top-k values
        if self.topK > 0:
            topk, indices = torch.topk(weights, self.topK)
            mask = torch.zeros_like(weights)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph
            weights = weights*mask
            
        ## normalize
        N = weights.shape[0]
        D = weights.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2 = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S = D1*weights*D2
        return torch.mm(S, emb_v)
