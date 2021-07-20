import torch
from utils import get_root_logger
from collections import OrderedDict
import os
import os.path as osp

def load(filename,
         model,
         logger):
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(filename,
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = torch.load(filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(checkpoint)
    if logger is not None:
        logger.info('load checkpoint from %s', filename)


def resume(filename,
           model,
           optimizer,
           logger,
           resume_optimizer=True,):
    assert isinstance(filename, str)
    assert os.path.exists(filename)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(filename,
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = torch.load(filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(checkpoint)
    logger.info('load checkpoint from %s', filename)
    epoch = checkpoint['meta']['epoch']
    iter = checkpoint['meta']['iter']
    if 'optimizer' in checkpoint and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info('resumed epoch %d, iter %d', epoch, iter)
    return epoch, iter

def load_part(filename,
              model,
              logger):
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(filename,
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = torch.load(filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')

    check_keys(model, checkpoint, logger)
    model.load_state_dict(checkpoint, strict=False)
    if logger is not None:
        logger.info('load checkpoint from %s', filename)
    return model


def save_epoch(model,
               optimizer,
               out_dir,
               epoch,
               iters,
               save_optimizer=True,
               meta=None,
               create_symlink=True):
    if meta is None:
        meta = dict(epoch=epoch + 1, iter=iters)
    elif isinstance(meta, dict):
        meta.update(epoch=epoch + 1, iter=iters)
    else:
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if save_optimizer:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'optimizer': optimizer.state_dict()}
    else:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())}

    save_path = out_dir + '/epoch_{}.pth'.format(epoch + 1)
    torch.save(checkpoint, save_path)


def save_item(model,
              optimizer,
              out_dir,
              epoch,
              iters,
              save_optimizer=True,
              meta=None,
              create_symlink=True):
    if meta is None:
        meta = dict(epoch=epoch + 1, iter=iters)
    elif isinstance(meta, dict):
        meta.update(epoch=epoch + 1, iter=iters)
    else:
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if save_optimizer:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'optimizer': optimizer.state_dict()}
    else:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())}
    save_path = out_dir + 'iters_{}.pth'.format(iters)
    torch.save(checkpoint, save_path)


def save_latest(model,
                optimizer,
                out_dir,
                epoch,
                iters,
                save_optimizer=True,
                meta=None,
                create_symlink=True):
    if meta is None:
        meta = dict(epoch=epoch + 1, iter=iters)
    elif isinstance(meta, dict):
        meta.update(epoch=epoch + 1, iter=iters)
    else:
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if save_optimizer:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'optimizer': optimizer.state_dict()}
    else:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())}
    save_path = osp.join(out_dir, 'latest.pth')
    torch.save(checkpoint, save_path)


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict, logger):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if logger is not None:
        logger.info('Missing keys:{}'.format(len(missing_keys)))
        logger.info('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        logger.info('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True