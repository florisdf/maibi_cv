from collections import OrderedDict

from PIL import Image
from torchvision.models import alexnet, vgg19, inception_v3, resnet50
from torch import nn
import torch

from .cnn_classifiers import train_transform, val_transform


def match_pretrained_embeddings(gallery, queries, model_name='resnet50'):
    model = get_cut_off_cnn(model_name).eval()

    gallery_embeddings = torch.vstack([
        get_im_embedding(Image.fromarray(img), model)
        for img in gallery
    ])

    query_embeddings = torch.vstack([
        get_im_embedding(Image.fromarray(img), model)
        for img in queries
    ])

    return match_embeddings(gallery_embeddings, query_embeddings)


def match_embeddings(gallery_embeddings, query_embeddings):
    return query_embeddings.matmul(gallery_embeddings.T)


def get_cut_off_cnn(name, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    if name == 'alexnet':
        model = alexnet(weights=weights)
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1]
        )
        return model
    elif name == 'vgg19':
        model = vgg19(weights=weights)
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-3]
        )
        return model
    elif name == 'inception_v3':
        model = inception_v3(weights=weights)
        model = nn.Sequential(
            OrderedDict(
                list(model.named_children())[:-2]
            )
        )
        return model
    elif name == 'resnet50':
        model = resnet50(weights=weights)
        model = nn.Sequential(
            OrderedDict(
                list(model.named_children())[:-1]
            )
        )
        return model


def get_im_embedding(im, model, normalize=True, train_tfm=False):
    """
    Pass the image through the model and return the embedding output.
    
    Args:
        im: The image (PIL Image)
        model: The neural network
        normalize: If True, normalize the embedding to have norm = 1
        train: If True, apply a train transform on the image.
    """
    x = val_transform(im) if not train_tfm else train_transform(im)
    x = x.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        y = model(x)

    y = y.squeeze()

    if normalize:
        return y / y.norm()
    else:
        return y