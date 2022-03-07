from collections import OrderedDict
from itertools import chain

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
    """
    Return a CNN.
    """
    if name == 'alexnet':
        return alexnet(pretrained=pretrained)
    elif name == 'vgg19':
        return vgg19(pretrained=pretrained)
    elif name == 'inception_v3':
        return inception_v3(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Unknown model {name}')


def get_cnn_clf(name, num_classes, pretrained=True):
    """
    Return a CNN to train as a classifier with a certain number of classes.
    """
    model = get_cnn(name, pretrained)

    if name == 'alexnet':
        model.classifier[-1] = nn.Linear(in_features=4096,
                                         out_features=num_classes)
    elif name == 'vgg19':
        model.classifier[-1] = nn.Linear(in_features=4096,
                                         out_features=num_classes)
    elif name == 'inception_v3':
        model.fc = nn.Linear(in_features=2048,
                             out_features=num_classes)
    elif name == 'resnet50':
        model.fc = nn.Linear(in_features=2048,
                             out_features=num_classes)

    return model


def get_top_parameters(model):
    name = model.__class__.__name__.lower()

    if name in ['alexnet', 'vgg']:
        return list(model.classifier.parameters())
    elif name == 'inception3':
        return [
            *model.Mixed_7c.parameters(),
            *model.avgpool.parameters(),
            *model.dropout.parameters(),
            *model.fc.parameters(),
        ]
    elif name == 'resnet':
        return [
            *model.layer4.parameters(),
            *model.avgpool.parameters(),
            *model.fc.parameters(),
        ]
    else:
        raise ValueError(f'Unknown model {name}')