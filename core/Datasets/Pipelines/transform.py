from core.Datasets.builder import PIPELINES
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F

# transforms.XXX((H,W))
# Image.open  ->  XX.size ->  W,H
@PIPELINES.register_module()
class Resize(object):
    def __init__(self,
                 img_scale=None,
                 ratio=0,
                 keep_ratio=True
                 ):
        if img_scale is None:
            self.img_scale = None
        else:
            assert isinstance(img_scale, (int, tuple))
            self.img_scale = img_scale
        if ratio > 0 and keep_ratio is True:
            self.ratio = ratio
            self.do_ratio = True
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        image = results['image']
        if self.do_ratio:
            w = image.size[0] * self.ratio
            h = image.size[1] * self.ratio
            osize = [int(h), int(w)]
            transform = transforms.Resize(osize)
            results['image'] = transform(image)
            if 'gt' in results:
                gt = results['gt']
                results['gt'] = transform(gt)
            if 'ce_image' and 'gc_image' and 'wb_image' in results:
                ce_image = results['ce_image']
                gc_image = results['gc_image']
                wb_image = results['wb_image']
                results['ce_image'] = transform(ce_image)
                results['gc_image'] = transform(gc_image)
                results['wb_image'] = transform(wb_image)
        else:
            if isinstance(self.img_scale, int):
                h, w = self.img_scale, self.img_scale
            else:
                h, w = self.img_scale
            osize = [h, w]
            transform = transforms.Resize(osize)
            results['image'] = transform(image)
            if 'gt' in results:
                gt = results['gt']
                results['gt'] = transform(gt)
            if 'ce_image' and 'gc_image' and 'wb_image' in results:
                ce_image = results['ce_image']
                gc_image = results['gc_image']
                wb_image = results['wb_image']
                results['ce_image'] = transform(ce_image)
                results['gc_image'] = transform(gc_image)
                results['wb_image'] = transform(wb_image)

        return results


@PIPELINES.register_module()
class RandomCrop(object):
    def __init__(self, img_scale=None):
        if img_scale is None:
            self.img_scale = None
        else:
            assert isinstance(img_scale, (int, tuple))
            self.img_scale = img_scale

    def __call__(self, results):
        image = results['image']
        if isinstance(self.img_scale, int):
            th, tw = self.img_scale, self.img_scale
        else:
            th, tw = self.img_scale
        # osize = [h, w]
        # transform = transforms.RandomCrop(osize)
        # results['image'] = transform(image)
        # results['gt'] = transform(gt)
        w, h = image.size
        # th = tw = opt.crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        results['image'] = F.crop(image, i, j, th, tw)
        if 'gt' in results:
            gt = results['gt']
            results['gt'] = F.crop(gt, i, j, th, tw)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            results['ce_image'] = F.crop(ce_image, i, j, th, tw)
            results['gc_image'] = F.crop(gc_image, i, j, th, tw)
            results['wb_image'] = F.crop(wb_image, i, j, th, tw)

        return results

@PIPELINES.register_module()
class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        if flip_ratio is None:
            self.flip_ratio = None
        else:
            assert isinstance(flip_ratio, float)
            self.flip_ratio = flip_ratio

    def __call__(self, results):
        image = results['image']
        # transform = transforms.RandomHorizontalFlip(p=self.flip_ratio)
        # results['image'] = transform(image)
        # results['gt'] = transform(gt)
        flip_prob = random.random()
        flip_transform = transforms.Compose([RandomHorizontalFlip(flip_prob)])
        results['image'] = flip_transform(image)
        if 'gt' in results:
            gt = results['gt']
            results['gt'] = flip_transform(gt)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            results['ce_image'] = flip_transform(ce_image)
            results['gc_image'] = flip_transform(gc_image)
            results['wb_image'] = flip_transform(wb_image)

        return results


class RandomHorizontalFlip(object):
    """
    Random horizontal flip.
    水平翻转
    prob = 0.5
    """

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, img):
        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)

        return img


@PIPELINES.register_module()
class Pad(object):
    def __init__(self, size_divisor=None, mode='pad'):
        '''
        mode = pad or resize
        '''
        if size_divisor is None:
            self.divisor = None
        else:
            assert isinstance(size_divisor, int)
            self.divisor = size_divisor
        self.mode = mode

    def __call__(self, results):
        image = results['image']
        pad_w = int(np.ceil(image.size[0] / self.divisor)) * self.divisor
        pad_h = int(np.ceil(image.size[1] / self.divisor)) * self.divisor
        if self.mode == 'pad':
            w = image.size[0]
            h = image.size[1]
            padding = (0, 0, pad_w - w, pad_h - h)
            transform = transforms.Pad(padding)
        else:
            transform = transforms.Resize((pad_h, pad_w))
        results['image'] = transform(image)

        if 'gt' in results:
            gt = results['gt']
            results['gt'] = transform(gt)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            results['ce_image'] = transform(ce_image)
            results['gc_image'] = transform(gc_image)
            results['wb_image'] = transform(wb_image)
        return results

# 再建一个 resize到可整除的

@PIPELINES.register_module()
class ImageToTensor(object):
    def __call__(self, results):
        image = results['image']
        totensor = transforms.ToTensor()
        if torch.cuda.is_available():
            results['image'] = totensor(image).cuda()
        else:
            results['image'] = totensor(image)

        if 'gt' in results:
            gt = results['gt']
            if torch.cuda.is_available():
                results['gt'] = totensor(gt).cuda()
            else:
                results['gt'] = totensor(gt)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            if torch.cuda.is_available():
                results['ce_image'] = totensor(ce_image).cuda()
                results['gc_image'] = totensor(gc_image).cuda()
                results['wb_image'] = totensor(wb_image).cuda()
            else:
                results['ce_image'] = totensor(ce_image)
                results['gc_image'] = totensor(gc_image)
                results['wb_image'] = totensor(wb_image)
        results['image_id'] = results['image_path'].split('/')[-1].split('.')[0]
        return results


@PIPELINES.register_module()
class Normalize(object):
    def __init__(self,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
    def __call__(self, results):
        image = results['image']
        Normalize = transforms.Normalize(mean=self.mean, std=self.std)
        results['image'] = Normalize(image)
        if 'gt' in results:
            gt = results['gt']
            results['gt'] = Normalize(gt)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            results['ce_image'] = Normalize(ce_image)
            results['gc_image'] = Normalize(gc_image)
            results['wb_image'] = Normalize(wb_image)
        return results


@PIPELINES.register_module()
class FlipEnsemble(object):
    def __init__(self, flip_ratio=0.5):
        if flip_ratio is None:
            self.flip_ratio = None
        else:
            assert isinstance(flip_ratio, float)
            self.flip_ratio = flip_ratio

    def __call__(self, results):
        image, gt = results['image'], results['gt']

        results['image_flip_lr'] = image.transpose(Image.FLIP_LEFT_RIGHT)
        results['image_rotate_270'] = image.transpose(Image.ROTATE_270)
        results['image_rotate_180'] = image.transpose(Image.ROTATE_180)
        results['image_rotate_90']= image.transpose(Image.ROTATE_90)
        results['image_flip_lr_rotate_270'] = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
        results['image_flip_lr_rotate_180'] = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
        results['image_flip_lr_rotate_90'] = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        return results

@PIPELINES.register_module()
class RandomRotate90(object):
    def __init__(self, ratio=0.5):
        if ratio is None:
            self.ratio = None
        else:
            assert isinstance(ratio, float)
            self.ratio = ratio

    def __call__(self, results):
        image = results['image']
        rotate_prob = random.random()
        if rotate_prob < self.ratio:
            results['image'] = image.transpose(Image.ROTATE_90)
        if 'gt' in results:
            gt = results['gt']
            if rotate_prob < self.ratio:
                results['gt'] = gt.transpose(Image.ROTATE_90)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            if rotate_prob < self.ratio:
                results['ce_image'] = ce_image.transpose(Image.ROTATE_90)
                results['gc_image'] = gc_image.transpose(Image.ROTATE_90)
                results['wb_image'] = wb_image.transpose(Image.ROTATE_90)
        return results


@PIPELINES.register_module()
class RandomRotate180(object):
    def __init__(self, ratio=0.5):
        if ratio is None:
            self.ratio = None
        else:
            assert isinstance(ratio, float)
            self.ratio = ratio

    def __call__(self, results):
        image = results['image']
        rotate_prob = random.random()
        if rotate_prob < self.ratio:
            results['image'] = image.transpose(Image.ROTATE_180)

        if 'gt' in results:
            gt = results['gt']
            if rotate_prob < self.ratio:
                results['gt'] = gt.transpose(Image.ROTATE_180)
        if 'ce_image' and 'gc_image' and 'wb_image' in results:
            ce_image = results['ce_image']
            gc_image = results['gc_image']
            wb_image = results['wb_image']
            if rotate_prob < self.ratio:
                results['ce_image'] = ce_image.transpose(Image.ROTATE_180)
                results['gc_image'] = gc_image.transpose(Image.ROTATE_180)
                results['wb_image'] = wb_image.transpose(Image.ROTATE_180)
        return results
