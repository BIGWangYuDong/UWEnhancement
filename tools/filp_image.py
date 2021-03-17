from PIL import Image
import os
import os.path

rootdir = '/home/dong/python-project/Dehaze/DATA/Test/val/a/26.png'

image = Image.open(rootdir)

image_flip_lr = image.transpose(Image.FLIP_LEFT_RIGHT)
image_rotate_270 = image.transpose(Image.ROTATE_270)
image_rotate_180 = image.transpose(Image.ROTATE_180)
image_rotate_90 = image.transpose(Image.ROTATE_90)
image_flip_lr_rotate_270 = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
image_flip_lr_rotate_180 = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
image_flip_lr_rotate_90 = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)

image_flip_lr_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/flip_lr.png'
image_rotate_270_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/rotate_270.png'
image_rotate_180_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/rotate_180.png'
image_rotate_90_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/rotate_90.png'
image_flip_lr_rotate_270_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/flip_lr_rotate_270.png'
image_flip_lr_rotate_180_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/flip_lr_rotate_180.png'
image_flip_lr_rotate_90_path = '/home/dong/python-project/Dehaze/DATA/Test/val/a/flip_lr_rotate_90.png'

image_flip_lr.save(image_flip_lr_path)
image_rotate_270.save(image_rotate_270_path)
image_rotate_180.save(image_rotate_180_path)
image_rotate_90.save(image_rotate_90_path)
image_flip_lr_rotate_270.save(image_flip_lr_rotate_270_path)
image_flip_lr_rotate_180.save(image_flip_lr_rotate_180_path)
image_flip_lr_rotate_90.save(image_flip_lr_rotate_90_path)


image_flip_lr = image_flip_lr.transpose(Image.FLIP_LEFT_RIGHT)
image_rotate_270 = image_rotate_270.transpose(Image.ROTATE_90)
image_rotate_180 = image_rotate_180.transpose(Image.ROTATE_180)
image_rotate_90 = image_rotate_90.transpose(Image.ROTATE_270)
image_flip_lr_rotate_270 = image_flip_lr_rotate_270.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
image_flip_lr_rotate_180 = image_flip_lr_rotate_180.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT)
image_flip_lr_rotate_90 = image_flip_lr_rotate_90.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


image_flip_lr.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_flip_lr.png')
image_rotate_270.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_rotate_270.png')
image_rotate_180.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_rotate_180.png')
image_rotate_90.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_rotate_90.png')
image_flip_lr_rotate_270.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_flip_lr_rotate_270.png')
image_flip_lr_rotate_180.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_flip_lr_rotate_180.png')
image_flip_lr_rotate_90.save('/home/dong/python-project/Dehaze/DATA/Test/val/a/re_image_flip_lr_rotate_90.png')

