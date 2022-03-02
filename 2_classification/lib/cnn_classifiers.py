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


def get_cut_off_cnn(name, pretrained=True):
    if name == 'alexnet':
        model = alexnet(pretrained=pretrained)
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1]
        )
        return model
    elif name == 'vgg19':
        model = vgg19(pretrained=pretrained)
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-3]
        )
        return model
    elif name == 'inception_v3':
        model = inception_v3(pretrained=pretrained)
        model = nn.Sequential(
            OrderedDict(
                list(model.named_children())[:-2]
            )
        )
        return model
    elif name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model = nn.Sequential(
            OrderedDict(
                list(model.named_children())[:-1]
            )
        )
        return model


def get_im_embedding(im, model, normalize=True):
    x = val_transform(im)
    x = x.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        y = model(x)

    y = y.squeeze()

    if normalize:
        return y / y.norm()
    else:
        return y