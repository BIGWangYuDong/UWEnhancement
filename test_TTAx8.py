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
from UW.utils import (mkdir_or_exist, get_root_logger,
                      save_epoch, save_latest, save_item,
                      resume, load)
import numpy as np
from UW.utils.save_image import (save_image, normimage,
                                 save_ensemble_image, save_ensemble_image_8)

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
                        default='/home/dong/python-project/Dehaze/configs/DCP_New.py',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default='/home/dong/python-project/'
                                'Dehaze/checkpoints/wyd/'
                                'New/znew/dehaze_3/epoch_1000.pth',
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

datasets = build_dataset(cfg.data.test)
# put model on gpu
if torch.cuda.is_available():
    model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
# create data_loader
data_loader = build_dataloader(
    datasets,
    cfg.data.val_samples_per_gpu,
    cfg.data.val_workers_per_gpu,
    len(cfg.gpu_ids))

load(cfg.load_from, model, None)

model.eval()

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

images = [os.path.join(args.inp, x) for x in os.listdir(args.inp) if is_image_file(x)]
total_t=0

def forward_chop(*args, forward_function=None,shave=12, min_size=16000000):#160000
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    # if self.input_large:
    #     scale = 1
    # else:
    #     scale = self.scale[self.idx_scale]
    scale = 1
    # n_GPUs = min(self.n_GPUs, 4)
    n_GPUs = 1
    _, _, h, w = args[0].size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    list_x = [[
        a[:, :, 0:h_size, 0:w_size],
        a[:, :, 0:h_size, (w - w_size):w],
        a[:, :, (h - h_size):h, 0:w_size],
        a[:, :, (h - h_size):h, (w - w_size):w]
    ] for a in args]

    list_y = []
    if w_size * h_size < min_size:
        for i in range(0, 4, n_GPUs):
            x = [torch.cat(_x[i:(i + n_GPUs)], dim=0) for _x in list_x]
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.extend(_y.chunk(n_GPUs, dim=0))
    else:
        for p in zip(*list_x):
            y = forward_chop(*p, forward_function=forward_function,shave=shave, min_size=min_size)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    b, c, _, _ = list_y[0][0].size()
    y = [_y[0].new(b, c, h, w) for _y in list_y]
    for _list_y, _y in zip(list_y, y):
        _y[:, :, :h_half, :w_half] \
            = _list_y[0][:, :, :h_half, :w_half]
        _y[:, :, :h_half, w_half:] \
            = _list_y[1][:, :, :h_half, (w_size - w + w_half):]
        _y[:, :, h_half:, :w_half] \
            = _list_y[2][:, :, (h_size - h + h_half):, :w_half]
        _y[:, :, h_half:, w_half:] \
            = _list_y[3][:, :, (h_size - h + h_half):, (w_size - w + w_half):]

    if len(y) == 1: y = y[0]

    return y

def forward_x8(*args, forward_function=None):
    # These codes are from https://github.com/thstkdgus35/EDSR-PyTorch
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    list_x = []
    for a in args:
        x = [a]
        for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

        list_x.append(x)

    list_y = []
    for x in zip(*list_x):
        y = forward_function(*x)
        if not isinstance(y, list): y = [y]
        if not list_y:
            list_y = [[_y] for _y in y]
        else:
            for _list_y, _y in zip(list_y, y): _list_y.append(_y)

    for _list_y in list_y:
        for i in range(len(_list_y)):
            if i > 3:
                _list_y[i] = _transform(_list_y[i], 't')
            if i % 4 > 1:
                _list_y[i] = _transform(_list_y[i], 'h')
            if (i % 4) % 2 == 1:
                _list_y[i] = _transform(_list_y[i], 'v')

    y = [torch.cat(_y, dim=0).mean(dim=0) for _y in list_y]#, keepdim=True
    if len(y) == 1: y = y[0]

    return y

def norm(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

for i, data in enumerate(data_loader):
    inputs = data['image']
    with torch.no_grad():
        im = forward_x8(inputs, forward_function=model.forward)
        # out_rgb = model(inputs)
    # rgb_numpy = normimage(out_rgb)
    im_numpy = norm(im)
    print('writing' + data['image_id'][0] + '.png')
    save_path = osp.join(cfg.savepath, cfg.load_from.split('/')[-1].split('.')[0])
    mkdir_or_exist(save_path)
    outsavepath = osp.join(save_path, data['image_id'][0] + '.png')
    # outsavepath_1 =  osp.join(save_path, data['image_id'][0] + '_orig.png')
    save_image(im_numpy, outsavepath)
    # save_image(rgb_numpy, outsavepath_1)
    # out_rgb.save(outsavepath_1)

print(total_t)