import matplotlib.pyplot as plt
import numpy as np

from .cnn_classifiers import val_resize_crop


def plot_transformed_val_input(im):
    im_resize_cropped = val_resize_crop(im)

    fig, axes = plt.subplots(ncols=2, figsize=(6, 12))

    axes[0].set_title('Original image')
    axes[0].imshow(np.array(im))

    axes[1].set_title('Resized + cropped image')
    axes[1].imshow(np.array(im_resize_cropped))