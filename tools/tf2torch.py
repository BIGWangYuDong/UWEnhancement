import tensorflow as tf
import os
import numpy as np
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn


flags = tf.app.flags
flags.DEFINE_integer("label_name", 230, "The size of label to produce [230]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
FLAGS = flags.FLAGS


class UWCNN(nn.Module):
    def __init__(self):
        super(UWCNN, self).__init__()
        self.conv2d_dehaze1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(3+16+16+16, 16, 3, 1, 1)
        self.dehaze4_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze5_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze6_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze7 = nn.Conv2d(51+48, 16, 3, 1, 1)
        self.dehaze7_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze8_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze9_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze10 = nn.Conv2d(99+48, 3, 3, 1, 1)

    def forward(self, x):
        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(x))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.dehaze4_relu(self.conv2d_dehaze4(dehaze_concat1))
        image_conv5 = self.dehaze5_relu(self.conv2d_dehaze5(image_conv4))
        image_conv6 = self.dehaze6_relu(self.conv2d_dehaze6(image_conv5))

        dehaze_concat2 = torch.cat([dehaze_concat1, image_conv4, image_conv5, image_conv6], dim=1)
        image_conv7 = self.dehaze7_relu(self.conv2d_dehaze7(dehaze_concat2))
        image_conv8 = self.dehaze8_relu(self.conv2d_dehaze8(image_conv7))
        image_conv9 = self.dehaze9_relu(self.conv2d_dehaze9(image_conv8))

        dehaze_concat3 = torch.cat([dehaze_concat2, image_conv7, image_conv8, image_conv9], dim=1)
        image_conv10 = self.dehaze10_relu(self.conv2d_dehaze10(dehaze_concat3))
        out = x + image_conv10

        return out


def load(model, checkpoint_dir=None, label_name=None):
    '''
    torch conv name should same as tf !!!
    '''
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", label_name)        # checkpoint document name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    tfvars = []
    init_vars = tf.train.list_variables(ckpt.model_checkpoint_path)
    for name, shape in init_vars:
        if name.split('/')[0] == 'model_h':
            array = tf.train.load_variable(ckpt.model_checkpoint_path, name)
            tfvars.append((name, array.squeeze()))
            convname = name.split('/')[1]
            if name.split('/')[2] == 'biases':
                array_new = np.ascontiguousarray(array.transpose())
                array_new_torch = torch.from_numpy(array_new)
                print(getattr(model, convname).bias)
                getattr(model, convname).bias = Parameter(array_new_torch)
                print(getattr(model, convname).bias)
            elif name.split('/')[2] == 'w':
                array_new = np.ascontiguousarray(array.transpose(3, 2, 0, 1))
                array_new_torch = torch.from_numpy(array_new)
                print(getattr(model, convname).weight)
                getattr(model, convname).weight = Parameter(array_new_torch)
                print(getattr(model, convname).weight)
    print()
    return model


def save_epoch(model):

    checkpoint = {
        'state_dict': model.state_dict()}

    save_path = 'UWCNN_type9.pth'
    torch.save(checkpoint, save_path)


if __name__ == '__main__':
    checkpoint_dir = FLAGS.checkpoint_dir
    label_name = FLAGS.label_name
    model = UWCNN()
    model = load(model, checkpoint_dir, label_name)
    save_epoch(model)
    print()

