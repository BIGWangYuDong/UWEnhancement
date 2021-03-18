import torch
import numpy as np
from PIL import Image
import os


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    return dn

def normimage(input_image, save_cfg=True, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            if save_cfg:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            else:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def save_ensemble_image(image_numpy, image_flip_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_flip_pil = Image.fromarray(image_flip_numpy)
    image_flip_pil = image_flip_pil.transpose(Image.FLIP_LEFT_RIGHT)
    image_pil = np.asarray(image_pil).astype(np.uint16)
    image_flip_pil = np.asarray(image_flip_pil).astype(np.uint16)
    out = (image_flip_pil + image_pil) / 2
    out = out.astype(np.uint8)
    out = Image.fromarray(out)
    out.save(image_path)


def save_ensemble_image_8(rgb_numpy,
                          image_flip_lr_numpy,
                          image_rotate_270_numpy,
                          image_rotate_180_numpy,
                          image_rotate_90_numpy,
                          image_flip_lr_rotate_270_numpy,
                          image_flip_lr_rotate_180_numpy,
                          image_flip_lr_rotate_90_numpy,
                          image_path):
    image_rgb_numpy = Image.fromarray(rgb_numpy)
    image_flip_lr = Image.fromarray(image_flip_lr_numpy)
    image_rotate_270 = Image.fromarray(image_rotate_270_numpy)
    image_rotate_180 = Image.fromarray(image_rotate_180_numpy)
    image_rotate_90 = Image.fromarray(image_rotate_90_numpy)
    image_flip_lr_rotate_270 = Image.fromarray(image_flip_lr_rotate_270_numpy)
    image_flip_lr_rotate_180 = Image.fromarray(image_flip_lr_rotate_180_numpy)
    image_flip_lr_rotate_90 = Image.fromarray(image_flip_lr_rotate_90_numpy)

    image_flip_lr = image_flip_lr.transpose(Image.FLIP_LEFT_RIGHT)
    image_rotate_270 = image_rotate_270.transpose(Image.ROTATE_90)
    image_rotate_180 = image_rotate_180.transpose(Image.ROTATE_180)
    image_rotate_90 = image_rotate_90.transpose(Image.ROTATE_270)
    image_flip_lr_rotate_270 = image_flip_lr_rotate_270.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
    image_flip_lr_rotate_180 = image_flip_lr_rotate_180.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT)
    image_flip_lr_rotate_90 = image_flip_lr_rotate_90.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


    image_pil = np.asarray(image_rgb_numpy).astype(np.uint32)
    image_flip_pil = np.asarray(image_flip_lr).astype(np.uint32)
    image_rotate_270 = np.asarray(image_rotate_270).astype(np.uint32)
    image_rotate_180 = np.asarray(image_rotate_180).astype(np.uint32)
    image_rotate_90 = np.asarray(image_rotate_90).astype(np.uint32)
    image_flip_lr_rotate_270 = np.asarray(image_flip_lr_rotate_270).astype(np.uint32)
    image_flip_lr_rotate_180 = np.asarray(image_flip_lr_rotate_180).astype(np.uint32)
    image_flip_lr_rotate_90 = np.asarray(image_flip_lr_rotate_90).astype(np.uint32)

    out = (image_pil + image_flip_pil + image_rotate_270 + image_rotate_180 +
           image_rotate_90 + image_flip_lr_rotate_270 + image_flip_lr_rotate_180 + image_flip_lr_rotate_90) / 8
    out = out.astype(np.uint8)
    out = Image.fromarray(out)
    out.save(image_path)



def save_image(image_numpy, image_path, usebytescale=False, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    if not usebytescale:
        image_pil = Image.fromarray(image_numpy)
        h, w, _ = image_numpy.shape

        if aspect_ratio > 1.0:
            image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        if aspect_ratio < 1.0:
            image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    else:
        image_pil = image_numpy
    image_pil.save(image_path)


def normimage_test(input_image, save_cfg=True, usebytescale=False, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            if save_cfg is True:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            else:
                image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    if usebytescale:
        shape = list(image_numpy.shape)
        if shape[-1] == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'
        bytedata = bytescale(image_numpy)
        strdata = bytedata.tostring()
        shape = list(bytedata.shape)
        shape = (shape[1], shape[0])
        return Image.frombytes(mode, shape, strdata)
    else:
        return image_numpy.astype(imtype)


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    '''
    copy from scipy1.1.0
    '''
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)