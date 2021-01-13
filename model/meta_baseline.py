import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.resnet12 import resnet12
from model.WRN28 import WideResNet
from model.convnet4 import Conv4
import utils
from model.tpn import LabelPropagation
from model.sim_label_protage import SimLabel


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

        self.lp = LabelPropagation(in_dim=in_dim)
        
        
    def forward(self, x_shot, y_shot, x_query, y_query):
        '''
            x_shot:(n_way, n_shot, *img_shape)
            x_query: (n_way * n_shot,)

            y_shot:(n_way, n_query, *img_shape)
            y_query: (n_way * n_query,)
        '''
        n_way = x_shot.shape[0]
        n_shot = x_shot.shape[1]
        n_query = x_query.shape[1]
        image_shape = x_shot.shape[2:]
        
        x_shot = x_shot.reshape(n_way * n_shot, *image_shape)
        x_query = x_query.reshape(n_way * n_query, *image_shape)
        
        x_all = torch.cat([x_shot, x_query], dim=0)
        x_all = self.encoder(x_all)           # (batch_size, 512)
        x_shot, x_query = x_all[:len(x_shot)], x_all[-len(x_query):]
        
        att_emb = self.lp(x_shot, y_shot, x_query, y_query)
        x_shot, x_query = att_emb[:len(x_shot)], att_emb[-len(x_query):]
        
       
        proto = x_shot.reshape(n_way, n_shot, -1).mean(dim=1)
        logits = utils.compute_logits(x_query, proto, metric=self.method, temp=self.temper)
        return logits, proto, x_shot.reshape(n_way, n_shot, -1)
