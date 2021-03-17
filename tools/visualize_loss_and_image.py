import visdom
from UW.utils.Visualizer import Visualizer
import collections
import random
from PIL import Image
import numpy as np


# a demo for visdom
# visualize loss and image during training
# should enter "python -m visdom.server" in a terminal
vis = visdom.Visdom()

visualizer = Visualizer()
# visualize loss plot
batch_size = 2
i = 0
for epoch in range(100):
    i += 1
    losses = collections.OrderedDict()
    l1loss = random.random()
    l2loss = random.random()
    l3loss = random.random()
    losses['l1loss'] = l1loss
    losses['ssimloss'] = l2loss
    losses['loss'] = l3loss
    visualizer.plot_current_losses(epoch+1, float(i * batch_size) / 100, losses)

# visualize image during training
img = Image.open('/home/dong/python-project/ZZ/XianZhuXing/image/531dataset/underwater/set_f104_531dataset.jpg')
img = np.array(img).transpose([2, 0, 1])
img_2 = Image.open('/home/dong/python-project/ZZ/XianZhuXing/image/531dataset/watergt/set_f104_531dataset.png')
# img_2 = np.array(img_2).transpose([2, 0, 1])
img_2 = np.tile(np.array(img_2),(3,1,1))
np.tile(img_2, (3, 1, 1))
images = []
images.append(img)
images.append(img_2)
images.append(img_2)
# images.append(img_2)
# images.append(img_2)
# vis.image(img)
# vis.image(img)
vis.images(images, nrow=3, padding=3, win=1, opts=dict(title='Output images'))
