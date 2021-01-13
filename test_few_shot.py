import argparse
import yaml
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.dataset import make
from datasets.samplers import CategoriesSampler
import utils
from model.meta_baseline import MetaBaseline

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = make(config['dataset'], 'test', None)
    utils.log('dataset: {} (x{}), {}'.format(dataset[0][0].shape, len(dataset), dataset.n_classes))

    n_way = 5
    n_shot, n_query = args.shot, 15
    n_batch = 600
    test_epochs = args.test_epochs
    batch_sampler = CategoriesSampler(dataset.label, n_batch, n_way, n_shot + n_query)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)

    # model
    model = MetaBaseline()
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model.load_state_dict(model_sv['model_sd'])
    else:
        print('no saved model')

    if config.get('load_encoder') is not None:
        encoder = model.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    model = model.cuda()
    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['test_loss', 'test_acc']
    aves = {k: utils.Averager() for k in aves_keys}

    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = utils.split_shot_query(data.cuda(), n_way, n_shot, n_query)
            y_shot, y_query = utils.make_nk_label(n_way, n_shot, n_query)
            y_shot = y_shot.cuda();y_query = y_query.cuda()

            with torch.no_grad():
                logits,_,_ = model(x_shot, y_shot, x_query, y_query)
                logits = logits.view(-1, n_way)
                loss = F.cross_entropy(logits, y_query)
                acc = utils.compute_acc(logits, y_query)

                aves['test_loss'].add(loss.item(), len(data))
                aves['test_acc'].add(acc, len(data))
                va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
            epoch, aves['test_acc'].item() * 100, mean_confidence_interval(va_lst) * 100, aves['test_loss'].item()))

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
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)
