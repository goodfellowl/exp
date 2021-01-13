import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR


_log_path = None


def split_shot_query(data, way, shot, query):
    '''
        x_shot:(n_way, n_shot, *img_shape)
        y_shot:(n_way, n_query, *img_shape)
        
    '''
    img_shape = data.shape[1:]
    data = data.view(way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=1)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous()
    return x_shot, x_query


def make_nk_label(way, shot, query):
    '''
        x_query: (n_way * n_shot,)
        y_query: (n_way * n_query,)
    '''
    label = torch.arange(way).unsqueeze(1).expand(way, shot + query)
    y_shot, y_query = label.split([shot, query], dim=1)
    y_shot = y_shot.reshape(-1)
    y_query = y_query.reshape(-1)
    return y_shot, y_query
    
def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None, freeze_encoder=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        if freeze_encoder:
            optimizer = SGD(filter(lambda p: p.requires_grad, params), lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        if freeze_encoder:
            optimizer = Adam(filter(lambda p: p.requires_grad, params), lr, weight_decay=weight_decay)
        else:
            optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    