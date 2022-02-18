import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import ConnectionPatch
from itertools import product
from fractions import Fraction

from .contrast_enhance import (
    linear_contrast_enhance,
    gamma_correct,
    hist_equalize,
    get_hist_eq_mapping,
    clahe,
)

from .denoising_sharpening import (
    build_mean_kernel,
    build_gaussian_kernel,
    apply_median_kernel,
    apply_mean_kernel,
    apply_gaussian_kernel,
)


def plot_img_repr(img, x0, y0, size):
    height, width, channels = img.shape

    y1 = min(y0 + size, height)
    x1 = min(x0 + size, width)
    slice_y = slice(y0, y1)
    slice_x = slice(x0, x1)

    ncols = channels + 1
    fig, axes = plt.subplots(ncols=ncols, figsize=(6*ncols, 6))

    channel_names = ['Red values', 'Green values', 'Blue values'] if channels == 3 else ['Gray values']
    for i, ax in enumerate(axes[1:]):
        channel_name = channel_names[i]
        img_slice = img[slice_y, slice_x, i]
        
        print(f'{channel_name}:')
        print(img_slice)
        print('\n')

        ax.set_title(channel_name)
        ax.imshow(img_slice, cmap='gray', vmin=0, vmax=255)

    img_copy = np.copy(img)
    cv2.rectangle(img_copy, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=5)
    axes[0].imshow(img_copy, cmap='gray' if channels == 1 else None)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    for (x_a, y_a), (x_b, y_b) in zip(product([x0], [y0, y1]),
                                      product([0], [1, 0])):
        con = ConnectionPatch(xyA=(x_a, y_a), xyB=(x_b, y_b), coordsA="data",
                              coordsB="axes fraction",
                              axesA=axes[0], axesB=axes[1], color="red")
        axes[1].add_artist(con)


def plot_img_hist(img, axes=None, log=False):
    is_color = (img.shape[-1] == 3)

    if is_color:
        if axes is None:
            fig, axes = plt.subplots(ncols=3, figsize=(3*6, 4), sharey=True)

        axes[0].set_title('Red channel')
        axes[0].set_ylabel('Pixel count')
        axes[0].hist(img[..., 0].flatten(), bins=256, range=[0, 256], color='red')

        axes[1].set_title('Green channel')
        axes[1].hist(img[..., 1].flatten(), bins=256, range=[0, 256], color='green')

        axes[2].set_title('Blue channel')
        axes[2].hist(img[..., 2].flatten(), bins=256, range=[0, 256], color='blue');

        for ax in axes:
            ax.set_xlabel('Pixel value')
            if log:
                ax.set_yscale('log')
            try:
                axes[0].sharey(ax)
            except ValueError:
                continue
    else:
        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes

        ax.set_ylabel('Pixel count')
        ax.set_xlabel('Pixel value')
        ax.hist(img.flatten(), bins=256, range=[0, 256], color='black')


def plot_linear_contrast_enhance(img, alpha, beta, log=False):
    new_img = linear_contrast_enhance(img, alpha, beta)
    plot_contrast_enhance(img, new_img, log=log)


def plot_gamma_correct(img, gamma, log=False):
    new_img = gamma_correct(img, gamma)

    axes = plot_contrast_enhance(img, new_img, nrows=3, log=log)
    xs = [x for x in range(256)]
    ys = [(x/255)**gamma*255 for x in xs]

    axes[2][0].set_title('Gamma function')
    axes[2][0].plot(xs, ys)
    
    for ax in axes[2][1:]:
        ax.set_axis_off()


def plot_hist_equalize(img, log=False):
    new_img = hist_equalize(img)
    axes = plot_contrast_enhance(img, new_img, nrows=3, log=log)
    
    xs = [x for x in range(256)]
    mapping = get_hist_eq_mapping(img)
    ys = [mapping[x] for x in xs]

    axes[2][0].set_title('Mapping function (scaled CDF)')
    axes[2][0].plot(xs, ys)
    
    for ax in axes[2][1:]:
        ax.set_axis_off()


def plot_clahe(img, clip_limit, grid_size, log=False):
    new_img = clahe(img, clip_limit, grid_size)
    plot_contrast_enhance(img, new_img, log=log)


def plot_contrast_enhance(img, new_img, nrows=2, log=False):
    if img.shape[-1] == 3:
        is_color = True
        ncols = 4
    else:
        is_color = False
        ncols = 2

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(ncols*6, nrows*5))

    axes[0][0].set_title('Old image')
    axes[0][0].imshow(img, vmin=0, vmax=255, cmap='gray' if not is_color else None)
    plot_img_hist(img, axes[0][1:] if is_color else axes[0][1], log=log)

    axes[1][0].set_title('New image')
    axes[1][0].imshow(new_img, vmin=0, vmax=255, cmap='gray' if not is_color else None)
    plot_img_hist(new_img, axes[1][1:] if is_color else axes[1][1], log=log)

    fig.tight_layout()

    return axes


def plot_kernel(img, kernel, use_frac=True):
    ncols = 3
    fig, axes = plt.subplots(ncols=ncols, figsize=(6*ncols, 6))

    axes[0].set_title('Kernel')
    axes[0].imshow(kernel, cmap='gray')

    kernel_mean = kernel.mean()

    if kernel.shape[0] <= 10:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                value = kernel[i, j]
                if isinstance(value, float):
                    if use_frac:
                        text = str(Fraction(value).limit_denominator())
                    else:
                        text = f'{value:.2f}'
                else:
                    text = str(value)
                axes[0].text(j, i, text,
                             ha="center", va="center", color="w" if value <= kernel_mean else "black")

    # Turn spines off and create white grid.
    axes[0].spines[:].set_visible(False)
    axes[0].set_xticks(np.arange(kernel.shape[1]+1)-.5)
    axes[0].set_yticks(np.arange(kernel.shape[0]+1)-.5)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].grid(color="w", linewidth=2)

    axes[1].set_title('Old image')
    axes[1].imshow(img)
    
    axes[2].set_title('New image')
    new_img = cv2.filter2D(img, -1, kernel)
    axes[2].imshow(new_img)


def plot_mean_kernel(img, size):
    kernel = build_mean_kernel(size)
    plot_kernel(img, kernel)


def plot_gaussian_kernel(img, size, sigma=0):
    kernel = build_gaussian_kernel(size, sigma)
    plot_kernel(img, kernel, use_frac=False)


def plot_median_kernel(img, size):
    new_img = apply_median_kernel(img, size)
    ncols = 2
    fig, axes = plt.subplots(ncols=ncols, figsize=(6*ncols, 6))

    axes[0].set_title('Old image')
    axes[0].imshow(img)
    
    axes[1].set_title('New image')
    axes[1].imshow(new_img)


def plot_multi_filter(img, size):
    med_img = apply_median_kernel(img, size)
    mean_img = apply_mean_kernel(img, size)
    gauss_img = apply_gaussian_kernel(img, size)

    ncols = 4
    fig, axes = plt.subplots(ncols=ncols, figsize=(6*ncols, 6))

    axes[0].set_title('Original')
    axes[0].imshow(img)

    axes[1].set_title('Mean filter')
    axes[1].imshow(mean_img)

    axes[2].set_title('Gaussian filter')
    axes[2].imshow(gauss_img)

    axes[3].set_title('Median filter')
    axes[3].imshow(med_img)