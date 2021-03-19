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
from UW.utils import Config
from UW.core.Models import build_network
from UW.core.Datasets import build_dataset, build_dataloader
from UW.core.Optimizer import build_optimizer, build_scheduler
from UW.utils import (mkdir_or_exist, get_root_logger,
                      save_epoch, save_latest, save_item, normimage_test,
                      resume, load, normPRED)

from UW.utils.save_image import (save_image, normimage,
                                     save_ensemble_image, save_ensemble_image_8)


from tensorboardX import SummaryWriter
# TORCH_VERSION = torch.__version__
# if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
#     try:
#         from tensorboardX import SummaryWriter
#     except ImportError:
#         raise ImportError('Please install tensorboardX to use '
#                           'TensorboardLoggerHook.')
# else:
#     try:
#         from torch.utils.tensorboard import SummaryWriter
#     except ImportError:
#         raise ImportError(
#             'Please run "pip install future tensorboard" to install '
#             'the dependencies to use torch.utils.tensorboard '
#             '(applicable to PyTorch 1.1 or higher)')

from getpass import getuser
from socket import gethostname
def get_host_info():
    return f'{getuser()}@{gethostname()}'


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
        # before iter

        inputs = data['image']
        with torch.no_grad():
            out_rgb = model(inputs)
        print('writing' + data['image_id'][0] + '.png')
        # input_numpy = normimage_test(inputs, save_cfg=save_cfg)
        rgb_numpy = normimage_test(out_rgb, save_cfg=save_cfg, usebytescale=cfg.usebytescale)

        outsavepath = osp.join(save_path, data['image_id'][0] + '.png')
        inputsavepath = osp.join(save_path, data['image_id'][0] + '_input.png')

        # save_image(input_numpy, inputsavepath)
        save_image(rgb_numpy, outsavepath, usebytescale=cfg.usebytescale)


