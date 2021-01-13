import argparse
import os
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.samplers import CategoriesSampler
from datasets.dataset import make
import utils
from model.meta_baseline import MetaBaseline
from model.center_loss import CenterLoss,WeightedCrossEntropy

def main(config):
    svname = args.name
    if svname is None:
        svname = '{}-{}-way-{}-shot'.format(config['train_dataset'], config['n_way'], config['n_shot'])
        svname += '_' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # Dataset
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    
    # train
    train_dataset = make(config['train_dataset'], 'train', 'resize')
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    train_sampler = CategoriesSampler(train_dataset.label, config['train_batches'], n_train_way, n_train_shot + n_query)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    # val
    val_dataset = make(config['val_dataset'], 'test', None)
    utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(val_dataset.label, 200, n_way, n_shot + n_query)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    # ### Model and optimizer ####
    if config.get('load'):
        checkpoint = torch.load(config['load'])
        model = MetaBaseline(config)
        model.load_state_dict(checkpoint['model_sd'])
    else:
        model = MetaBaseline(config)
        if config.get('load_encoder'):
            model_dict = model.state_dict()
            
            checkpoint = torch.load(config['load_encoder'])
            pretrained_dict = checkpoint['model_sd']
            pretrained_encoder_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_encoder_dict)
            model.load_state_dict(model_dict)
    
    model = model.cuda()
    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    # optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    base_params = list(map(id, model.encoder.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr":0.001},
        {"params": model.encoder.parameters(), "lr": 0.0002}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=5.e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # lr_scheduler = None
    #####################loss####################
    centerloss = CenterLoss()
    weighted_ce = WeightedCrossEntropy()
    ######## train ######################
 
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['t_loss', 't_acc', 'v_loss', 'v_acc']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    accumulation_steps = 4
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
        if config.get('freeze_encoder'):
            utils.freeze_encoder(model)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, _ in tqdm(train_loader, desc='train', leave=False):     
            x_shot, x_query = utils.split_shot_query(data.cuda(), n_train_way, n_train_shot, n_query)
            y_shot, y_query = utils.make_nk_label(n_train_way, n_train_shot, n_query)    
            y_shot = y_shot.cuda();y_query = y_query.cuda()
            
            logits, center, feat = model(x_shot, y_shot, x_query, y_query)
            logits = logits.view(-1, n_train_way)
            closs = centerloss(center, feat, y_shot)
            loss = F.cross_entropy(logits, y_query) + closs
            acc = utils.compute_acc(logits, y_query)

            # loss = loss / accumulation_steps
            # loss.backward()
            # if epoch % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['t_loss'].add(loss.item())
            aves['t_acc'].add(acc)

        ############################## eval #######################
        model.eval()
        for name, loader, name_loss, name_acc in [('val', val_loader, 'v_loss', 'v_acc')]:
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = utils.split_shot_query(data.cuda(), n_way, n_shot, n_query)
                y_shot, y_query = utils.make_nk_label(n_way, n_shot, n_query)
                y_shot = y_shot.cuda();y_query = y_query.cuda()

                with torch.no_grad():
                    logits, _, _ = model(x_shot, y_shot, x_query, y_query)
                    logits = logits.view(-1, n_way)
                    loss = F.cross_entropy(logits, y_query)
                    acc = utils.compute_acc(logits, y_query)
                    
                    aves[name_loss].add(loss.item())
                    aves[name_acc].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train:loss {:.4f}|acc {:.4f}, val:loss {:.4f}|acc {:.4f}, {} {}/{}'.format(
            epoch, aves['t_loss'], aves['t_acc'], aves['v_loss'], aves['v_acc'], t_epoch, t_used, t_estimate))

        writer.add_scalars('loss', {
            'train': aves['t_loss'],
            'val': aves['v_loss'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['t_acc'],
            'val': aves['v_acc'],
        }, epoch)
        # save model
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'model_sd': model_.state_dict(),
            'training': training,
        }
        if epoch == max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch_last.pth'))
            torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(save_path, 'epoch_{}.pth'.format(epoch)))

        if aves['v_acc'] > max_va:
            max_va = aves['v_acc']
            torch.save(save_obj, os.path.join(save_path, 'max_v_acc.pth'))
        writer.flush()

if __name__ == '__main__':
    seed = 42
    random.seed(seed)                       # random产生的随机数
    np.random.seed(seed)                # numpy产生的随机数
    torch.manual_seed(seed)                                                        #为当前CPU 设置随机种子
    torch.cuda.manual_seed(seed)                                            # 为当前的GPU 设置随机种子
    torch.cuda.manual_seed_all(seed)                                    #当使用多块GPU 时，均设置随机种子
    torch.backends.cudnn.deterministic = True                   # pytorch 使用CUDANN 加速，即使用GPU加速
    torch.backends.cudnn.benchmark = False                     # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
    torch.backends.cudnn.enabled = True   

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_few_shot_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
