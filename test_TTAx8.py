import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import torch
import time
import os.path as osp
from torch.nn.parallel import DataParallel
import collections
from torch.autograd import Variable
import visdom
from UW.utils import Config
from UW.core.Models import build_network
from UW.core.Datasets import build_dataset, build_dataloader
from UW.core.Optimizer import build_optimizer, build_scheduler
from UW.utils import (mkdir_or_exist, get_root_logger, normimage_test,
                      save_epoch, save_latest, save_item, forward_x8,
                      resume, load)
import numpy as np
from UW.utils.save_image import (save_image, normimage,
                                 save_ensemble_image, save_ensemble_image_8)

'''
test time augmentation
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str,
                        default='/home/dong/GitHub_Frame/UW/config/UWCNN.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='/home/dong/GitHub_Frame/UW/checkpoints/UWCNN/UWCNN_type3.pth',
                        help='the dir to save logs and models,')
    parser.add_argument('--savepath', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.load_from is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.load_from = args.load_from
    if args.savepath is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.savepath = args.savepath
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.savepath = osp.join('./results',
                                osp.splitext(osp.basename(args.config))[0])
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    mata = dict()

    # make dirs
    mkdir_or_exist(osp.abspath(cfg.savepath))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.log_file = osp.join(cfg.savepath, f'{timestamp}.log')

    # create text log
    # build model
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load(cfg.load_from, model, None)
    # build dataset
    datasets = build_dataset(cfg.data.test)
    # put model on gpu
    if torch.cuda.is_available():
        # model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
        model = model.cuda()
    # create data_loader
    data_loader = build_dataloader(
        datasets,
        cfg.data.val_samples_per_gpu,
        cfg.data.val_workers_per_gpu,
        len(cfg.gpu_ids))

    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True

    save_path = osp.join(cfg.savepath, cfg.load_from.split('/')[-1].split('.')[0])
    mkdir_or_exist(save_path)
    # before run
    model.eval()
    t = time.time()
    for i, data in enumerate(data_loader):
        inputs = data['image']
        with torch.no_grad():
            im = forward_x8(inputs, forward_function=model.forward).unsqueeze(0)
            # out_rgb = model(inputs)
        # rgb_numpy = normimage(out_rgb)
        im_numpy = normimage_test(im, save_cfg=save_cfg, usebytescale=cfg.usebytescale)
        print('writing' + data['image_id'][0] + '.png')
        save_path = osp.join(cfg.savepath, cfg.load_from.split('/')[-1].split('.')[0])
        mkdir_or_exist(save_path)
        outsavepath = osp.join(save_path, data['image_id'][0] + '.png')
        # outsavepath_1 =  osp.join(save_path, data['image_id'][0] + '_orig.png')
        save_image(im_numpy, outsavepath,  usebytescale=cfg.usebytescale)
        # save_image(rgb_numpy, outsavepath_1)
        # out_rgb.save(outsavepath_1)