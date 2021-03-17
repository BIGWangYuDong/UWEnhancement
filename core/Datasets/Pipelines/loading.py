from UW.core.Datasets.builder import PIPELINES
from PIL import Image

@PIPELINES.register_module()
class LoadImageFromFile(object):
    def __init__(self, gt_type='color'):
        self.gt_type = gt_type

    def __call__(self, results):
        '''
        "image_path": self.img_prefix + data,
        "gt_path": self.img_prefix + data
        '''
        image = Image.open(results['image_path']).convert('RGB')
        results['image'] = image
        results['img_shape'] = image.size
        if self.gt_type == 'color':
            gt = Image.open(results['gt_path']).convert('RGB')
            results['gt'] = gt
            results['gt_shape'] = gt.size
        elif self.gt_type == 'gray':
            gt = Image.open(results['gt_path'])
            results['gt'] = gt
            results['gt_shape'] = gt.size
        else:
            gt = Image.open(results['gt_path'])
            results['gt'] = gt
            results['gt_shape'] = gt.size
        assert image.size == gt.size
        return results