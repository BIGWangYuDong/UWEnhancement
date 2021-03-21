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
import visdom
from UW.utils.read_file import Config
from UW.core.Models import build_network
from UW.core.Datasets import build_dataset, build_dataloader
from UW.core.Optimizer import build_optimizer, build_scheduler
from UW.utils import (mkdir_or_exist, get_root_logger,
                      save_epoch, save_latest, save_item,
                      resume, load)
from UW.core.Losses import build_loss
from UW.utils.Visualizer import Visualizer
from UW.utils.save_image import normimage, normPRED

from tensorboardX import SummaryWriter
TORCH_VERSION = torch.__version__

from getpass import getuser
from socket import gethostname
def get_host_info():
    return f'{getuser()}@{gethostname()}'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='/home/dong/GitHub_Frame/UW/config/UWCNN.py',
                        help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models,')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
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

    mata = dict()

    # make dirs
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    # create text log
    logger = get_root_logger(log_file=cfg.log_file, log_level=cfg.log_level)
    dash_line = '-' * 60 + '\n'
    logger.info(dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    model = build_network(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # create optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    Scheduler = build_scheduler(cfg.lr_config)
    logger.info('-' * 20 + 'finish build optimizer' + '-' * 20)

    if cfg.resume_from:
        start_epoch, ite_num = resume(cfg.resume_from, model, optimizer, logger, )
    elif cfg.load_from:
        load(cfg.load_from, model, logger)

    logger.info('-' * 20 + 'finish build model' + '-' * 20)
    logger.info('Total Parameters: %d,   Trainable Parameters: %s',
                model.net_parameters['Total'],
                str(model.net_parameters['Trainable']))
    # build dataset
    datasets = build_dataset(cfg.data.train)
    logger.info('-' * 20 + 'finish build dataset' + '-' * 20)
    # put model on gpu
    if torch.cuda.is_available():
        if len(cfg.gpu_ids) == 1:
            model = model.cuda()
            logger.info('-' * 20 + 'model to one gpu' + '-' * 20)
        else:
            model = DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
            logger.info('-' * 20 + 'model to multi gpus' + '-' * 20)
    # create data_loader
    data_loader = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids))
    logger.info('-' * 20 + 'finish build dataloader' + '-' * 20)

    visualizer = Visualizer()
    vis = visdom.Visdom()
    criterion_ssim_loss = build_loss(cfg.loss_ssim)
    # criterion_l1_loss = build_loss(cfg.loss_l1)
    # criterion_perc_loss = build_loss(cfg.loss_perc)
    # criterion_tv_loss = build_loss(cfg.loss_tv)

    ite_num = 0
    start_epoch = 1     # start range at 1-1 = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    max_iters = cfg.total_epoch * len(data_loader)

    save_cfg = False
    for i in range(len(cfg.test_pipeling)):
        if 'Normalize' == cfg.test_pipeling[i].type:
            save_cfg = True

    model.train()
    print("---start training...")
    scheduler = Scheduler(optimizer, cfg)
    # before run
    t = time.time()
    log_dir = osp.join(cfg.work_dir, 'tf_logs')
    write = SummaryWriter(log_dir)
    logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), cfg.work_dir)
    logger.info('max: %d epochs, %d iters', cfg.total_epoch, max_iters)
    for epoch in range(start_epoch-1, cfg.total_epoch):
        # before epoch
        logger.info('\nStart Epoch %d -------- ', epoch+1)
        for i, data in enumerate(data_loader):
            # before iter
            data_time = time.time()-t
            ite_num = ite_num + 1
            ite_num4val = ite_num*cfg.data.samples_per_gpu
            inputs, gt = data['image'], data['gt']
            out_rgb = model(inputs)

            optimizer.zero_grad()
            # loss_up_1 = criterion_l1_loss(up_1_out, gt)
            # loss_up_2 = criterion_l1_loss(up_2_out, gt)
            # loss_up_3 = criterion_l1_loss(up_3_out, gt)
            # loss_fft = (loss_up_1 + loss_up_2 + loss_up_3) / 6
            # loss_l1 = criterion_l1_loss(out_rgb, gt)
            loss_ssim = criterion_ssim_loss(out_rgb, gt)
            # loss_perc = criterion_perc_loss(out_rgb, gt, cfg)
            # loss_perc = criterion_tv_loss(out_rgb, gt)
            # loss_tv = criterion_tv_loss(out_rgb, gt)
            # loss = loss_l1 + loss_ssim +loss_tv
            loss_l1 = loss_perc = loss_tv = loss_ssim
            loss = loss_ssim
            loss.backward()
            optimizer.step()

            write.add_scalar('loss_l1', loss_l1, ite_num)
            write.add_scalar('loss_ssim', loss_ssim, ite_num)
            write.add_scalar('loss_perc', loss_perc, ite_num)
            write.add_scalar('loss_tv', loss_tv, ite_num)

            logger.info('Epoch: [%d][%d/%d]  lr: %f  time: %.3f loss_l1: %f  loss_ssim: %f  loss_perc: %f   loss_tv: %f   loss: %f',
                        epoch+1, ite_num, max_iters, optimizer.param_groups[0]['lr'],
                        data_time, loss_l1, loss_ssim, loss_perc, loss_tv, loss)
            losses = collections.OrderedDict()
            losses['loss_l1'] = loss_l1.data.cpu()
            losses['loss_ssim'] = loss_ssim.data.cpu()
            losses['loss_tv'] = loss_tv.data.cpu()
            losses['loss_perc'] = loss_perc.data.cpu()
            losses['total_loss'] = loss.data.cpu()
            visualizer.plot_current_losses(epoch + 1,
                                           float(i) / len(data_loader),
                                           losses)
            # after iter
            time_ = time.time() - t
            t = time.time()
            if ite_num4val % 5 == 0:

                inputshow = normimage(inputs, save_cfg=save_cfg)
                gtshow = normimage(gt, save_cfg=save_cfg)
                outshow = normimage(out_rgb, save_cfg=save_cfg)

                shows = []
                # shows.append(inputs_show)
                # shows.append(gt_show)
                # shows.append(outputs_show)
                # shows.append(outputs_show1)
                shows.append(inputshow.transpose([2, 0, 1]))
                shows.append(gtshow.transpose([2, 0, 1]))
                shows.append(outshow.transpose([2, 0, 1]))
                vis.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
                ite_num4val = 0
            if ite_num % 100 == 0:
                save_latest(model, optimizer, cfg.work_dir, epoch, ite_num)
                model.train()
        if epoch % 20 == 0 or epoch == cfg.total_epoch - 1:
            # print('-'*30, 'saving model')
            save_epoch(model, optimizer, cfg.work_dir, epoch, ite_num)
            model.train()
        # after eppoch
        # update learning rate
        # print(optimizer.param_groups[0]['lr'])
        # if cfg.lr_config.step[1] >= (epoch+1) >= cfg.lr_config.step[0]:
        scheduler.step()
    # after run
    write.close()
        # print(optimizer.param_groups[0]['lr'])
    print()
    save_epoch(model, optimizer, cfg.work_dir, epoch, ite_num)
    logger.info('Finish Training')


    # import matplotlib.pyplot as plt
    # plt.plot(list(range(start_epoch-1, cfg.total_epoch)), lr_list)
    # plt.xlabel("epoch")
    # plt.ylabel("lr")
    # plt.title("CosineAnnealingLR")
    # plt.show()