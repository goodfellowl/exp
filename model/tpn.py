import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, in_dim, learned_alpha=False, learned_sigma=False, alpha=0.99, topk=10):
        super(LabelPropagation, self).__init__()
        self.topK = topk
        self.in_dim = in_dim
        self.learned_sigma = learned_sigma
        if self.learned_sigma:
            self.relation = RelationNetwork()

        if learned_alpha == False:   # learned sigma, fixed alpha
            self.alpha = torch.tensor(alpha, requires_grad=False).cuda(0)
        else:                        # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor(alpha).cuda(0), requires_grad=True)
        
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
        ## sigmma
        if self.learned_sigma:
            self.sigma = self.relation(emb_q)
            emb_q = emb_q / (self.sigma+eps) # N*d
            emb1 = torch.unsqueeze(emb_q,1) # N*1*d
            emb2 = torch.unsqueeze(emb_q,0) # 1*N*d
            weights = ((emb1-emb2)**2).sum(-1)   # N*N*d -> N*N
            weights = torch.exp(-weights/2)
        else:
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
        
        # Step2: Label Propagation, F = (I-\alpha S)^{-1}Y
        # s_one_hot_labels = torch.zeros(n_way*n_shot, n_way).scatter_(1, s_labels.unsqueeze(1).cpu(), 1).cuda()
        # q_one_hot_labels = torch.zeros(n_way*n_query, n_way).cuda()
        # all_one_hot_labels = torch.cat([s_one_hot_labels,q_one_hot_labels], dim=0)
        # pred_all =  torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), all_one_hot_labels)
        # pred_q = pred_all[n_way*n_shot:, :]
        
        return torch.mm(S, emb_v)


class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2*2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)            # max-pool without padding 
        self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu

        out = out.view(out.size(0),-1) # bs*1

        return out
