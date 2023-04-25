import tensorflow as tf
import os
import numpy as np
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class UWGANUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_conv_1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.layer1_conv_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.layer1_pooling = nn.MaxPool2d(2)

        self.layer2_conv_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.layer2_conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer2_pooling = nn.MaxPool2d(2)

        self.layer3_conv_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.layer3_conv_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer3_pooling = nn.MaxPool2d(2)

        self.layer4_conv_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.layer4_conv_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer4_up = nn.ConvTranspose2d(256, 128, 2, 2)

        self.layer5_conv_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.layer5_conv_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer5_up = nn.ConvTranspose2d(128, 64, 2, 2)

        self.layer6_conv_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.layer6_conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer6_pooling = nn.MaxPool2d(2)
        self.layer6_up = nn.ConvTranspose2d(64, 32, 2, 2)

        self.layer7_conv_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.layer7_conv_2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.out = nn.Conv2d(32, 3, 1, 1, 1)

    def forward(self, inputs):
        layer1_out = F.relu(self.layer1_conv_1(inputs))
        layer1_out = F.relu(self.layer1_conv_2(layer1_out))

        layer2_out = self.layer1_pooling(layer1_out)
        layer2_out = F.relu(self.layer2_conv_1(layer2_out))
        layer2_out = F.relu(self.layer2_conv_2(layer2_out))

        layer3_out = self.layer2_pooling(layer2_out)
        layer3_out = F.relu(self.layer3_conv_1(layer3_out))
        layer3_out = F.relu(self.layer3_conv_2(layer3_out))

        layer4_out = self.layer3_pooling(layer3_out)
        layer4_out = F.relu(self.layer4_conv_1(layer4_out))
        layer4_out = F.relu(self.layer4_conv_2(layer4_out))

        layer5_out = self.layer4_up(layer4_out)
        layer5_out = torch.cat([layer5_out, layer3_out], dim=1)
        layer5_out = F.relu(self.layer5_conv_1(layer5_out))
        layer5_out = F.relu(self.layer5_conv_2(layer5_out))

        layer6_out = self.layer5_up(layer5_out)
        layer6_out = torch.cat([layer6_out, layer2_out], dim=1)
        layer6_out = F.relu(self.layer6_conv_1(layer6_out))
        layer6_out = F.relu(self.layer6_conv_2(layer6_out))

        layer7_out = self.layer6_up(layer6_out)
        layer7_out = torch.cat([layer7_out, layer1_out], dim=1)
        layer7_out = F.relu(self.layer7_conv_1(layer7_out))
        layer7_out = F.relu(self.layer7_conv_2(layer7_out))

        out = torch.tanh(self.out(layer7_out))

        return out


def load(model, checkpoint_dir=None, label_name=None):
    '''
    torch conv name should same as tf !!!
    '''
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    tfvars = []
    init_vars = tf.train.list_variables(ckpt.model_checkpoint_path)
    for name, shape in init_vars:
        if 'layer' in name:
            name_list = name.split('/')
            if len(name_list) == 4:
                continue
            elif len(name_list) == 3:
                layer = name_list[0]
                conv = name_list[1]
                array = tf.train.load_variable(ckpt.model_checkpoint_path, name)
                tfvars.append((name, array.squeeze()))
                if name_list[2] == 'bias':
                    array_new = np.ascontiguousarray(array.transpose())
                    array_new_torch = torch.from_numpy(array_new)
                    if '5' in name:
                        print()
                    # print(getattr(model, f'{layer}_{conv}').bias)
                    before_shape = getattr(model, f'{layer}_{conv}').bias.shape
                    getattr(model, f'{layer}_{conv}').bias = Parameter(array_new_torch)
                    # print(getattr(model, f'{layer}_{conv}').bias)
                    after_shape = getattr(model, f'{layer}_{conv}').bias.shape
                    assert before_shape == after_shape
                elif name_list[2] == 'kernel':
                    array_new = np.ascontiguousarray(array.transpose(3, 2, 0, 1))
                    array_new_torch = torch.from_numpy(array_new)
                    if '5' in name:
                        print()
                    before_shape = getattr(model, f'{layer}_{conv}').weight.shape
                    # print(getattr(model, f'{layer}_{conv}').weight)
                    getattr(model, f'{layer}_{conv}').weight = Parameter(array_new_torch)
                    after_shape = getattr(model, f'{layer}_{conv}').weight.shape
                    assert before_shape == after_shape
                    # print(getattr(model, f'{layer}_{conv}').weight)
                else:
                    print('Error:', name_list)
            else:
                print('Error:', name_list)
        elif 'upsample' in name:
            name_list = name.split('/')
            if len(name_list) == 3:
                continue
            elif len(name_list) == 2:
                layer = name_list[0].split('_')[1]
                array = tf.train.load_variable(ckpt.model_checkpoint_path, name)
                tfvars.append((name, array.squeeze()))
                if name_list[1] == 'bias':
                    array_new = np.ascontiguousarray(array.transpose())
                    array_new_torch = torch.from_numpy(array_new)
                    before_shape = getattr(model, f'layer{int(layer)-1}_up').bias.shape
                    # print(getattr(model, f'layer{int(layer)-1}_up').bias)
                    getattr(model, f'layer{int(layer)-1}_up').bias = Parameter(array_new_torch)
                    after_shape = getattr(model, f'layer{int(layer) - 1}_up').bias.shape
                    assert before_shape == after_shape
                    # print(getattr(model, f'layer{int(layer)-1}_up').bias)
                elif name_list[1] == 'kernel':
                    array_new = np.ascontiguousarray(array.transpose(3, 2, 0, 1))
                    array_new_torch = torch.from_numpy(array_new)
                    # print(getattr(model, f'layer{int(layer)-1}_up').weight)
                    before_shape = getattr(model, f'layer{int(layer) - 1}_up').weight.shape
                    getattr(model, f'layer{int(layer)-1}_up').weight = Parameter(array_new_torch)
                    after_shape = getattr(model, f'layer{int(layer) - 1}_up').weight.shape
                    assert before_shape == after_shape
                    # print(getattr(model, f'layer{int(layer)-1}_up').weight)
            else:
                print('error')
        elif name =='conv2d/bias':
            array = tf.train.load_variable(ckpt.model_checkpoint_path, name)
            tfvars.append((name, array.squeeze()))
            array_new = np.ascontiguousarray(array.transpose())
            array_new_torch = torch.from_numpy(array_new)
            # print(getattr(model, 'out').bias)
            before_shape = getattr(model, 'out').bias.shape
            getattr(model, 'out').bias = Parameter(array_new_torch)
            after_shape = getattr(model, 'out').bias.shape
            assert before_shape == after_shape
            # print(getattr(model, 'out').bias)
        elif name =='conv2d/kernel':
            array = tf.train.load_variable(ckpt.model_checkpoint_path, name)
            tfvars.append((name, array.squeeze()))
            array_new = np.ascontiguousarray(array.transpose(3, 2, 0, 1))
            array_new_torch = torch.from_numpy(array_new)
            before_shape = getattr(model, 'out').weight.shape
            # print(getattr(model, 'out').weight)
            getattr(model, 'out').weight = Parameter(array_new_torch)
            after_shape = getattr(model, 'out').weight.shape
            assert before_shape == after_shape
            # print(getattr(model, 'out').weight)
    print()
    return model


def save_epoch(model):

    checkpoint = {
        'state_dict': model.state_dict()}

    save_path = '/home/dong/GitHub_Frame/UW/checkpoints/UGAN_UNet/UWGANUNet.pth'
    torch.save(checkpoint, save_path)


if __name__ == '__main__':
    checkpoint_dir = "/home/dong/code_sensetime/uw_enhancement/UWGAN_UIE/ckpt/unet_ckpt/WaterType2/gdl_l1/"
    model = UWGANUNet()
    model = load(model, checkpoint_dir)
    save_epoch(model)
    print()