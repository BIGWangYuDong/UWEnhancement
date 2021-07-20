from core.Datasets.builder import PIPELINES
from PIL import Image

@PIPELINES.register_module()
class LoadImageFromFile(object):
    def __init__(self, gt_type='color', get_gt=True):
        self.gt_type = gt_type
        self.get_gt = get_gt

    def __call__(self, results):
        '''
        "image_path": self.img_prefix + data,
        "gt_path": self.img_prefix + data
        '''
        image = Image.open(results['image_path']).convert('RGB')
        results['image'] = image
        results['img_shape'] = image.size
        if self.get_gt is True:
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

@PIPELINES.register_module()
class LoadWaterNetImage(object):
    def __init__(self, path=None):
        if path is not None:
            self.path = path

    def __call__(self, results):
        '''
        "image_path": self.img_prefix + data,
        "gt_path": self.img_prefix + data
        '''
        ce_path = results['image_path'].split('test')[0] + self.path + 'ce/' + results['image_path'].split('/')[-1]
        gc_path = results['image_path'].split('test')[0] + self.path + 'gc/' + results['image_path'].split('/')[-1]
        wb_path = results['image_path'].split('test')[0] + self.path + 'wb/' + results['image_path'].split('/')[-1]
        ce_image = Image.open(ce_path).convert('RGB')
        gc_image = Image.open(gc_path).convert('RGB')
        wb_image = Image.open(wb_path).convert('RGB')

        results['ce_image'] = ce_image
        results['gc_image'] = gc_image
        results['wb_image'] = wb_image

        results['ce_image_shape'] = ce_image.size
        results['gc_image_shape'] = gc_image.size
        results['wb_image_shape'] = wb_image.size
        # assert ce_image.size == gc_image.size == wb_image

        return results