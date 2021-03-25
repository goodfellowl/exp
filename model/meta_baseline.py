import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
from model.resnet12 import resnet12
from model.tpn import LabelPropagation


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def graphbuild(input):

    eps = np.finfo(float).eps
    emb1 = torch.unsqueeze(input,1) # N*1*d
    emb2 = torch.unsqueeze(input,0) # 1*N*d
    weights = ((emb1-emb2)**2).sum(-1)   # N*N*d -> N*N
    mask = weights != 0
    weights = weights / weights[mask].std()
    weights = torch.exp(-weights/2)
        
    ## keep top-k values
    # if self.topK > 0:
    topk, indices = torch.topk(weights, 10)
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
    return S

class MetaBaseline(nn.Module):
    def __init__(self, config, method='cos', temper=10., temper_learnable=True):
        super().__init__()
        if config['model_args']['encoder'] == 'wrn':    
            self.encoder = WideResNet()
        elif config['model_args']['encoder'] == 'resnet12': 
            self.encoder = resnet12()
        self.method = method
        in_dim = self.encoder.out_dim

        if temper_learnable:
            self.temper = nn.Parameter(torch.tensor(temper))
        else:
            self.temper = temper
            
        # method_1
        # self.lp = LabelPropagation(in_dim=in_dim)

        self.conv = nn.Sequential(
                            nn.Conv2d(in_dim,in_dim, 1, bias=False),
                            nn.BatchNorm2d(in_dim),
                            nn.LeakyReLU(0.1))
        self.gc1 = GraphConvolution(512, 512)
        
    def forward(self, x_shot, y_shot, x_query, y_query):
        '''
            x_shot:(n_way, n_shot, *img_shape)
            y_shot: (n_way * n_shot,)

            x_query:(n_way, n_query, *img_shape)
            y_query: (n_way * n_query,)
        '''
        n_way = x_shot.shape[0]
        n_shot = x_shot.shape[1]
        n_query = x_query.shape[1]
        image_shape = x_shot.shape[2:]
        
        x_shot = x_shot.reshape(n_way * n_shot, *image_shape)
        x_query = x_query.reshape(n_way * n_query, *image_shape)
        
        x_all = torch.cat([x_shot, x_query], dim=0)
        x_all = self.encoder(x_all)           # (batch_size, 512, 5,5)
        x_all = self.conv(x_all)
        x_all = x_all.view(x_all.shape[0], x_all.shape[1], -1).mean(dim=2)
        
        adj1 = graphbuild(x_all)
        x_all = self.gc1(x_all, adj1)

        x_shot, x_query = x_all[:len(x_shot)], x_all[-len(x_query):]
        proto = x_shot.reshape(n_way, n_shot, -1).mean(dim=1)

        # method_1
        # att_emb = self.lp(x_shot, y_shot, x_query, y_query)
        # x_shot, x_query = att_emb[:len(x_shot)], att_emb[-len(x_query):]   
        # proto = x_shot.reshape(n_way, n_shot, -1).mean(dim=1)

        logits = utils.compute_logits(x_query, proto, metric=self.method, temp=self.temper).view(-1, n_way)
        loss = F.cross_entropy(logits, y_query) 
        acc = utils.compute_acc(logits, y_query)
        return loss, acc
