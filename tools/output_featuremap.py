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
import os.path as osp
from UW.utils.read_file import Config
from UW.core.Models import build_network
from UW.utils import load


def viz(module, input):
    print('0=', input[0].size())

    x = input[0][0]
    feature_map = torch.mean(x, 0, keepdim=True).cpu().numpy()
    # feature_map = torch.sum(x, 0, keepdim=True).cpu().numpy()
    feature_maps.append(feature_map[0])
    plt.imshow(feature_map[0], cmap='jet')
    plt.axis('off')
    plt.margins(0, 0)
    plt.rcParams['figure.dpi'] = 300
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(output_root)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str,
                        default='/home/dong/GitHub_Frame/UW/config/UIEC2Net.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='/home/dong/GitHub_Frame/UW/checkpoints/UIEC2Net/UIEC2Net.pth',
                        help='the dir to save logs and models,')
    parser.add_argument('--work_dir', help='the dir to save logs and models,')
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


args = parse_args()
cfg = Config.fromfile(args.config)
if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    # use config filename as default work_dir if cfg.work_dir is None
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids
else:
    cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

load(args.load_from, model, None)

# put model on gpu
if torch.cuda.is_available():
    model = model.cuda()


name_list = [
    'module.backbone.block2.block2.denselayer1.conv1',
    'rgb_con5'
    # 'backbone.features.29'
]
global output_root
output_root = '/home/dong/GitHub_Frame/UW/results/UIEC2Net/featuremap/01.png'
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
image_root = '/home/dong/GitHub_Frame/UW/DATA/Test/test/2_img_.png'
img = cv2.imread(image_root)
img = t(img).unsqueeze(0).to(device)
with torch.no_grad():
    model.forward(img)
    print()





