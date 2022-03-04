from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import alexnet, vgg19, inception_v3, resnet50
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


val_resize_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


val_transform = transforms.Compose([
    *val_resize_crop.transforms,
    transforms.ToTensor(),
    normalize,
])


train_resize_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
])


train_transform = transforms.Compose([
    *train_resize_crop.transforms,
    transforms.ToTensor(),
    normalize,
])


def get_cnn(name, pretrained=True):
    if name == 'alexnet':
        return alexnet(pretrained=pretrained)
    elif name == 'vgg19':
        return vgg19(pretrained=pretrained)
    elif name == 'inception_v3':
        return inception_v3(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)