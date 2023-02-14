import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import ConnectionPatch
from itertools import product
from fractions import Fraction
from .utils import lightness_only

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
    build_laplacian_kernel,
    build_sharpening_kernel,
    apply_median_kernel,
    apply_mean_kernel,
    apply_gaussian_kernel,
    apply_laplacian_kernel,
    apply_sharpening_kernel,
    apply_unsharp_mask,
    apply_DoG,
)


def get_xy_slices(img, x0, y0, size):
    height, width = img.shape[:2]

    y1 = min(y0 + size, height)
    x1 = min(x0 + size, width)
    slice_y = slice(y0, y1)
    slice_x = slice(x0, x1)

    return slice_x, slice_y

def plot_img_repr(img, crop_x0, crop_y0, crop_size):
    if len(img.shape) == 2:
        img = img[..., None]

    height, width, channels = img.shape

    ncols = channels + 1
    fig, axes = plt.subplots(ncols=ncols, figsize=(6*ncols, 6))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)
    plot_img_slice(img, slice_x, slice_y, axes=axes[0:2])

    channel_names = ['Red values', 'Green values', 'Blue values'] if channels == 3 else ['Gray values']
    for i, ax in enumerate(axes[1:]):
        channel_name = channel_names[i]
        img_slice = img[slice_y, slice_x, i]
        
        print(f'{channel_name}:')
        print(img_slice)
        print('\n')

        ax.set_title(channel_name)
        ax.imshow(img_slice, cmap='gray', vmin=0, vmax=255)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

        
def plot_img_slice(img, slice_x, slice_y, axes=None):
    is_color = (img.shape[-1] == 3)

    if axes is None:
        ncols = 2
        fig, axes = plt.subplots(ncols=ncols, figsize=(ncols*6, 4))

    x0, y0 = slice_x.start, slice_y.start
    x1, y1 = slice_x.stop, slice_y.stop

    img_copy = np.copy(img)
    cv2.rectangle(img_copy, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=5)
    axes[0].imshow(img_copy, cmap='gray' if not is_color else None)

    img_slice = img[slice_y, slice_x]
    axes[1].imshow(img_slice)

    for (x_a, y_a), (x_b, y_b) in zip(product([x0], [y0, y1]),
                                      product([0], [1, 0])):
        con = ConnectionPatch(xyA=(x_a, y_a), xyB=(x_b, y_b), coordsA="data",
                              coordsB="axes fraction",
                              axesA=axes[0], axesB=axes[1], color="red")
        axes[1].add_artist(con)
    

def plot_img_hist(img, axes=None, log=False, lightness_only=False):
    is_color = (img.shape[-1] == 3)

    if is_color and not lightness_only:
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

        if is_color and lightness_only:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[0]
            img = list(cv2.split(lab))[0]
            ax.set_title('Lightness')

        ax.set_ylabel('Pixel count')
        ax.set_xlabel('Pixel value')
        ax.hist(img.flatten(), bins=256, range=[0, 256], color='black')


def plot_linear_contrast_enhance(img, alpha, beta, log=False):
    new_img = linear_contrast_enhance(img, alpha, beta)
    plot_contrast_enhance(img, new_img, log=log, lightness_only=False)


def plot_gamma_correct(img, gamma, log=False):
    new_img = gamma_correct(img, gamma)

    axes = plot_contrast_enhance(img, new_img, nrows=3, log=log, lightness_only=False)
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


def plot_clahe(img, clip_limit=2.0, grid_size=8, log=False):
    new_img = clahe(img, clip_limit, grid_size)
    plot_contrast_enhance(img, new_img, log=log)


def plot_contrast_enhance(img, new_img, nrows=2, log=False, lightness_only=True):
    is_color = img.shape[-1] == 3
    mult_ax = is_color and not lightness_only
        
    ncols = 4 if mult_ax else 2

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(ncols*6, nrows*5))

    axes[0][0].set_title('Old image')
    axes[0][0].imshow(img, vmin=0, vmax=255, cmap='gray' if not is_color else None)
    plot_img_hist(img, axes[0][1:] if mult_ax else axes[0][1], log=log, lightness_only=lightness_only)

    axes[1][0].set_title('New image')
    axes[1][0].imshow(new_img, vmin=0, vmax=255, cmap='gray' if not is_color else None)
    plot_img_hist(new_img, axes[1][1:] if mult_ax else axes[1][1], log=log, lightness_only=lightness_only)

    fig.tight_layout()

    return axes


def plot_kernel(img, kernel, name, crop_x0=0, crop_y0=0, crop_size=100, use_frac=True,
                axes=None):
    if axes is None:
        ncols = 2
        nrows = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows))

    axes[0][0].set_title('Kernel')
    axes[0][0].imshow(kernel, cmap='gray')
    axes[0][1].set_axis_off()

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
                axes[0][0].text(j, i, text,
                                ha="center", va="center",
                                color="w" if value <= kernel_mean else "black")

    # Turn spines off and create white grid.
    axes[0][0].set_xticks(np.arange(kernel.shape[1]+1)-.5)
    axes[0][0].set_yticks(np.arange(kernel.shape[0]+1)-.5)
    axes[0][0].set_xticklabels([])
    axes[0][0].set_yticklabels([])
    axes[0][0].grid(color="w", linewidth=2)

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[1][0].set_title('Original image')
    plot_img_slice(img, slice_x, slice_y, axes=axes[1])
    
    axes[2][0].set_title(f'Filtered image ({name})')
    new_img = cv2.filter2D(img, -1, kernel)
    plot_img_slice(new_img, slice_x, slice_y, axes=axes[2])


def plot_mean_kernel(img, size, crop_x0=0, crop_y0=0, crop_size=100):
    kernel = build_mean_kernel(size)
    plot_kernel(img, kernel, f'{size}x{size} mean', crop_x0, crop_y0, crop_size)


def plot_gaussian_kernel(img, size, crop_x0=0,
                         crop_y0=0, crop_size=100):
    kernel = build_gaussian_kernel(size)
    plot_kernel(img, kernel, f'{size}x{size} Gaussian', crop_x0, crop_y0, crop_size, use_frac=False)


def plot_median_kernel(img, size, crop_x0=0,
                       crop_y0=0, crop_size=100):
    new_img = apply_median_kernel(img, size)

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[0][0].set_title('Original image')
    plot_img_slice(img, slice_x, slice_y, axes=axes[0])

    axes[1][0].set_title(f'Filtered image ({size}x{size} median)')
    plot_img_slice(new_img, slice_x, slice_y, axes=axes[1])


def plot_multi_filter(img, size, crop_x0=0,
                      crop_y0=0, crop_size=100):
    med_img = apply_median_kernel(img, size)
    mean_img = apply_mean_kernel(img, size)
    gauss_img = apply_gaussian_kernel(img, size)

    ncols = 2
    nrows = 4
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[0][0].set_title('Original')
    plot_img_slice(img, slice_x, slice_y, axes=axes[0])

    axes[1][0].set_title(f'Filtered with {size}x{size} mean')
    plot_img_slice(mean_img, slice_x, slice_y, axes=axes[1])

    axes[2][0].set_title(f'Filtered with {size}x{size} Gaussian filter')
    plot_img_slice(gauss_img, slice_x, slice_y, axes=axes[2])
    
    axes[3][0].set_title(f'Filtered with {size}x{size} median filter')
    plot_img_slice(med_img, slice_x, slice_y, axes=axes[3])



def plot_laplacian_kernel(img, crop_x0=0, crop_y0=0, crop_size=100):
    kernel = build_laplacian_kernel()
    plot_kernel(img, kernel, f'Laplacian', crop_x0, crop_y0, crop_size)

    
def plot_sharpening_kernel(img, crop_x0=0, crop_y0=0, crop_size=100):
    kernel = build_sharpening_kernel()
    plot_kernel(img, kernel, f'Sharpening', crop_x0, crop_y0, crop_size)


def plot_unsharp_mask(img, size, crop_x0=0, crop_y0=0, crop_size=100):
    new_img = apply_unsharp_mask(img, size)

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[0][0].set_title('Original image')
    plot_img_slice(img, slice_x, slice_y, axes=axes[0])

    axes[1][0].set_title(f'Filtered with unsharp mask (size {size}x{size})')
    plot_img_slice(new_img, slice_x, slice_y, axes=axes[1])


def plot_DoG(img, size_1, size_2, crop_x0=0, crop_y0=0, crop_size=100):
    new_img = apply_DoG(img, size_1, size_2)

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[0][0].set_title('Original image')
    plot_img_slice(img, slice_x, slice_y, axes=axes[0])

    axes[1][0].set_title(f'Filtered with DoG (size 1 = {size_1}; size 2 = {size_2})')
    plot_img_slice(new_img, slice_x, slice_y, axes=axes[1])


def plot_multi_sharpening(img, size, crop_x0=0,
                          crop_y0=0, crop_size=100):
    sharpening_img = apply_sharpening_kernel(img)
    unsharp_img = apply_unsharp_mask(img, size)
    size_2 = size * 5
    dog_img = apply_DoG(img, size, size_2)

    ncols = 2
    nrows = 4
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 6*nrows))

    slice_x, slice_y = get_xy_slices(img, crop_x0, crop_y0, crop_size)

    axes[0][0].set_title('Original')
    plot_img_slice(img, slice_x, slice_y, axes=axes[0])

    axes[1][0].set_title(f'Filtered with sharpening')
    plot_img_slice(sharpening_img, slice_x, slice_y, axes=axes[1])

    axes[2][0].set_title(f'Filtered with {size}x{size} unsharp mask')
    plot_img_slice(unsharp_img, slice_x, slice_y, axes=axes[2])
    
    axes[3][0].set_title(f'Filtered with DoG (size 1 = {size}; size 2 = {size_2})')
    plot_img_slice(dog_img, slice_x, slice_y, axes=axes[3])
