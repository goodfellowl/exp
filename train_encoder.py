import argparse
import os
import yaml
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
from datasets.samplers import CategoriesSampler
from datasets.dataset import make
from model.pre_baseline import PreBaseline



def main(config):
    svname = args.name
    if svname is None:
        svname = 'encoder_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        svname += '_' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #  ### Dataset ####
    # train dataset
    train_dataset = make(config['train_dataset'], 'train', 'resize', True)
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))
    # if config.get('visualize_datasets'):
    #     utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    print('#####################################################################################') 
    # val dataset
    if config.get('val_dataset'):
        eval_val = True
        n_way, n_shot, n_query = 5, 5, 15
        val_dataset = make(config['val_dataset'], 'test', None)
        utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
        val_sampler = CategoriesSampler(val_dataset.label, 600, n_way, n_shot + n_query)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    else:
        eval_val = False

    #  ### Model and Optimizer ####
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = PreBaseline()
        model.load_state_dict(model_sv['model_sd'])
    else:
        model = PreBaseline(config)
    model = model.cuda()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    ######## train model ################
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_v_acc = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    
    for epoch in range(1, max_epoch + 1):    
        timer_epoch.s()
        aves_keys = ['t_loss', 't_acc', 'v_loss', 'v_acc']
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        for data, label,rot in tqdm(train_loader, desc='train', leave=False):
            data = data.reshape(config['batch_size'] * 4, 3, 84,84)
            label = label.reshape(-1)
            rot = rot.reshape(-1)
           
            data, label, rot = data.cuda(), label.cuda(), rot.cuda()
            logits, logits_rot = model(data)

            loss = F.cross_entropy(logits, label)
            loss_rot = F.cross_entropy(logits_rot, rot)
            loss_total = loss + loss_rot
            acc = utils.compute_acc(logits, label)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            aves['t_loss'].add(loss.item())
            aves['t_acc'].add(acc)

            logits = None;loss = None

        # eval
        if eval_val:
            model.eval()            
            for data, _ in tqdm(val_loader, desc='val', leave=False):
                x_shot, x_query = utils.split_shot_query(data.cuda(), n_way, n_shot, n_query)
                y_shot, y_query = utils.make_nk_label(n_way, n_shot, n_query)
                y_shot = y_shot.cuda();y_query = y_query.cuda()

                with torch.no_grad():
                    image_shape = x_shot.shape[2:]
                    x_shot = x_shot.reshape(n_way * n_shot, *image_shape)
                    x_query = x_query.reshape(n_way * n_query, *image_shape)
        
                    x_all = torch.cat([x_shot, x_query], dim=0)
                    x_all = model.module.encoder(x_all)           # (batch_size, 512)
                    x_shot, x_query = x_all[:len(x_shot)], x_all[-len(x_query):]

                    proto = x_shot.reshape(n_way, n_shot, -1).mean(dim=1)
                    logits = utils.compute_logits(x_query, proto, metric='cos', temp=model.module.classifier.temper)
                    logits = logits.view(-1, n_way)
                    loss = F.cross_entropy(logits, y_query)
                    acc = utils.compute_acc(logits, y_query)
                
                    aves['v_loss'].add(loss.item())
                    aves['v_acc'].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        
        log_str = 'epoch {}, train:loss {:.4f}|acc {:.4f}'.format(epoch_str, aves['t_loss'], aves['t_acc'])
        writer.add_scalars('loss', {'train': aves['t_loss']}, epoch)
        writer.add_scalars('acc', {'train': aves['t_acc']}, epoch)

        if eval_val:
            log_str += ', val:loss {:.4f}|acc {:.4f}'.format(aves['v_loss'], aves['v_acc'])
            writer.add_scalars('loss', {'val': aves['v_loss']}, epoch)
            writer.add_scalars('acc', {'val': aves['v_acc']}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        
        utils.log(log_str)

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
            
        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(save_path, 'epoch_{}.pth'.format(epoch)))

        if aves['v_acc'] > max_v_acc:
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
    parser.add_argument('--config', default='./configs/train_encoder_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='fs_test_256_rot_dropout')
    parser.add_argument('--gpu', default='1,2,3')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
