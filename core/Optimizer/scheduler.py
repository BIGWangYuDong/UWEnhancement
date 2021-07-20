import copy
import torch
import inspect
from core.Registry import Registry, build_from_cfg
from torch.optim import lr_scheduler

SCHEDULER = Registry('scheduler')
SCHEDULER_BUILDERS = Registry('scheduler builder')


def build_scheduler(cfg, default_args=None):
    scheduler = build_from_cfg(cfg, SCHEDULER, default_args)
    return scheduler

@SCHEDULER.register_module()
class Epoch(object):
    '''
    examples:
    >>> lr_config = dict(type='Epoch',          # Epoch or Iter
    >>>                  warmup='exp',       # liner, step, exp,
    >>>                  step=[10, 20],          # start with 1
    >>>                  liner_end=0.00001,
    >>>                  step_gamma=0.1,
    >>>                  exp_gamma=0.9)
    or
    >>>lr_config = dict(type='Epoch',          # Epoch or Iter
    >>>                 warmup='linear',       # liner, step, exp,
    >>>                 step=[10, 20],          # start with 1
    >>>                 liner_end=0.00001,
    >>>                 step_gamma=0.1,
    >>>                 exp_gamma=0.9)
    '''
    def __init__(self,
                 warmup,
                 step,
                 liner_end,
                 step_gamma=0.1,
                 exp_gamma=0.9,
                 ):
        self.warmup = warmup
        self.step_gamma = step_gamma
        self.exp_gamma = exp_gamma
        self.step = step
        self.liner_end = liner_end
        assert isinstance(self.step, list)

    def __call__(self, optimizer, cfg):
        assert isinstance(self.step, list) or isinstance(self.step, int)
        if type(self.step) == list:
            assert len(self.step) == 2 or len(self.step) == 1
            epoch_start = self.step[0]
            if len(self.step) == 2:
                epoch_end = self.step[1]
            else:
                epoch_end = cfg.total_epoch
        else:
            return NotImplementedError('Error')

        self.lr = optimizer.defaults['lr']
        if self.warmup == 'linear':
            def lambda_rule(epoch, lr=self.lr, lrend=self.liner_end):
                if epoch_end >= epoch+1 >= epoch_start:
                    lr_l = 1.0 - max(0, epoch + 1 - epoch_start) / float(epoch_end - epoch_start + 1)
                elif epoch_end <= epoch+1:
                    lr_l = lrend / lr
                else:
                    lr_l = 1
                return lr_l
            # lambda1 = lambda epoch: 1.0 - max(0, epoch + 1 - epoch_start) / float(epoch_end - epoch_start + 1) \
            #     if epoch_end >= epoch+1 >= epoch_start else 1
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.warmup == 'step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, [epoch_start, epoch_end], gamma=self.step_gamma)
        elif self.warmup == 'exp':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_gamma )
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.warmup)
        return scheduler
        #
# @SCHEDULER.register_module()
# class Iter(object):
#     '''
#     # Assuming optimizer uses lr = 0.05 for all groups
#     # lr = 0.05     if epoch < 30
#     # lr = 0.005    if 30 <= epoch < 80
#     # lr = 0.0005   if epoch >= 80
#     scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
#     for epoch in range(100):
#         train(...)
#         validate(...)
#         scheduler.step()
#     '''
