import argparse
import os
import warnings
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

# build the model and load checkpoint
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
from UW.configs import Config
from UW.core.Models import build_network
from UW.core.Datasets import build_dataset, build_dataloader
from UW.core.Optimizer import build_optimizer, build_scheduler
from UW.utils import (mkdir_or_exist, get_root_logger,
                          save_epoch, save_latest, save_item,
                          resume, load)
from UW.core.Losses import build_loss
from UW.utils.Visualizer import Visualizer
import numpy as np
from UW.utils.save_image import (save_image, normimage, save_ensemble_image, save_ensemble_image_8)



def viz(module, input):
    print('0=', input[0].size())

    x = input[0][0]
    feature_map = torch.mean(x, 0, keepdim=True).cpu().numpy()
    # feature_map = torch.sum(x, 0, keepdim=True).cpu().numpy()
    feature_maps.append(feature_map[0])
    plt.imshow(feature_map[0], cmap='jet')
    plt.axis('off')
    plt.margins(0, 0)
    # plt.axis('off')
    # feature_map = input[0][0][0].cpu().numpy()
    # feature_maps.append(feature_map)
    # plt.imshow(feature_map, cmap='jet')
    # plt.show()
    plt.rcParams['figure.dpi'] = 300
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(output_root)
    #
    # plt.savefig('/home/dong/python-project/mmdetection/Samples/Starfish.png')




'''
test time augmentation
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument("--inp", default="/home/dong/python-project/Dehaze/DATA/Test/train/", type=str,
                        help="test images path")
    parser.add_argument("--opt", default="/home/dong/python-project/Dehaze/results/ZZ/a", type=str,
                        help="output images path")

    parser.add_argument('--config',type=str,
                        default='/home/dong/python-project/Dehaze/configs/UIEC2Net.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='/home/dong/python-project/Dehaze/checkpoints/wyd/New/dehaze_backbone_1/epoch_1000.pth',
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

os.environ["CUDA_VISIBLE_DEVICES"]='0'

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

model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

# put model on gpu
if torch.cuda.is_available():
    model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)

load(cfg.load_from, model, None)

name_list = [
    # 'neck.fpn_convs.0.conv','neck.fpn_convs.1.conv','neck.fpn_convs.2.conv','neck.fpn_convs.3.conv',
    # 'backbone.layer1.0.conv1',
    # 'backbone.layer2.1.conv2',
    # 'backbone.layer3.4.conv3',
    'module.backbone.block2.block2.denselayer1.conv1',
    # 'backbone.features.29',
    # 'backbone.extra.4'
]
global output_root
output_root = '/home/dong/python-project/Dehaze/results/wyd/New/featuremap/1.jpg'
global feature_maps
feature_maps = []
for name, m in model.named_modules():
    if name in name_list:
        # title = str(conv_index)
        m.register_forward_pre_hook(viz)
# image_root = '/home/dong/python-project/mmdetection/data/UW/image_underwaterDCP/VOC2007/JPEGImages/000001.jpg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize((256, 256)),  # 要改一下 比例，怎么成比例。
                        # transforms.Resize((300, 300)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
image_root = '/home/dong/python-project/Dehaze/DATA/Test/gt/01.png'
img = cv2.imread(image_root)
img = t(img).unsqueeze(0).to(device)
with torch.no_grad():
    model.forward(img)
    print()





