import cv2
from ipywidgets import interact, widgets


from .plots import (
    plot_img_repr,
    plot_linear_contrast_enhance,
    plot_gamma_correct,
    plot_hist_equalize,
    plot_clahe,
    plot_mean_kernel,
    plot_gaussian_kernel,
    plot_median_kernel,
    plot_multi_filter,
)

from .denoising_sharpening import (
    add_salt_pepper_noise,
    add_gaussian_noise,
)


def read_and_convert_img(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def build_plot_wrapper(func):
    def plot_wrapper(img_file, *args, **kwargs):
        img = read_and_convert_img(img_file)
        func(img, *args, **kwargs)
    return plot_wrapper

def interact_img_repr(img_files=['data/verona_bw.png', 'data/verona_gray.jpg', 'data/verona_color.jpg']):
    x_slider = widgets.IntSlider(min=0, max=960, value=782, continuous_update=False)
    y_slider = widgets.IntSlider(min=0, max=1280, value=186, continuous_update=False)
    size_slider = widgets.IntSlider(min=1, max=30, value=15, continuous_update=False)

    def plot_wrapper(img_file, x0, y0, size):
        img = read_and_convert_img(img_file)

        if len(img.shape) == 2:
            img = img[..., None]

        height, width, channels = img.shape
        x_slider.max = width
        y_slider.max = height
        return plot_img_repr(img, x0, y0, size)

    interact(
        plot_wrapper,
        img_file=img_files,
        x0=x_slider, y0=y_slider, size=size_slider
    )


def interact_linear_contrast_enhance(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    alpha_slider = widgets.FloatSlider(min=0.1, max=3, value=1, continuous_update=False)
    beta_slider = widgets.IntSlider(min=-255, max=255, value=0, continuous_update=False)
    
    def plot_wrapper(img_file, alpha, beta, log=False):
        img = read_and_convert_img(img_file)
        return plot_linear_contrast_enhance(img, alpha, beta, log)

    interact(
        plot_wrapper,
        img_file=img_files,
        alpha=alpha_slider,
        beta=beta_slider
    )


def interact_gamma_correct(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    gamma_slider = widgets.FloatLogSlider(min=-1.4, max=1.4, value=1.0, step=0.1, continuous_update=False)
    
    def plot_wrapper(img_file, gamma, log=False):
        img = read_and_convert_img(img_file)
        return plot_gamma_correct(img, gamma, log)

    interact(
        plot_wrapper,
        img_file=img_files,
        gamma=gamma_slider,
    )

    
def interact_hist_equalize(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):    
    def plot_wrapper(img_file, log=False):
        img = read_and_convert_img(img_file)
        return plot_hist_equalize(img, log)

    interact(
        plot_wrapper,
        img_file=img_files,
    )


def interact_clahe(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    clip_slider = widgets.FloatSlider(min=1.0, max=3.0, value=2.0, continuous_update=False)
    grid_size_slider = widgets.IntSlider(min=1, max=32, value=8, continuous_update=False)
    
    def plot_wrapper(img_file, clip_limit, grid_size, log=False):
        img = read_and_convert_img(img_file)
        return plot_clahe(img, clip_limit, grid_size, log)

    interact(
        plot_wrapper,
        img_file=img_files,
        clip_limit=clip_slider,
        grid_size=grid_size_slider
    )
    

def interact_mean_kernel(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    size_slider = widgets.IntSlider(min=1, max=50, value=3, continuous_update=False)

    interact(
        build_plot_wrapper(plot_mean_kernel),
        img_file=img_files,
        size=size_slider,
    )


def interact_gaussian_kernel(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    size_slider = widgets.IntSlider(min=2, max=50, value=3, continuous_update=False)
    sigma_slider = widgets.FloatSlider(min=0, max=3, value=0, continuous_update=False)

    interact(
        build_plot_wrapper(plot_gaussian_kernel),
        img_file=img_files,
        size=size_slider,
        sigma=sigma_slider,
    )


def interact_median_kernel(img_files=['data/verona_color.jpg', 'data/verona_gray.jpg']):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)

    interact(
        build_plot_wrapper(plot_median_kernel),
        img_file=img_files,
        size=size_slider,
    )
    
def interact_denoise_with_filters(
    img_files=['data/verona_color.jpg', 'data/verona_gray.jpg'],
):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)
    noise_amount_slider = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.01, continuous_update=False)

    def plot_wrapper(img_file, filter_size, noise, noise_amount):
        img = read_and_convert_img(img_file)[:100, :100, :]

        if noise == 'Salt and pepper':
            img = add_salt_pepper_noise(img, amount=1*noise_amount)
        elif noise == 'Gaussian':
            img = add_gaussian_noise(img, 0, 255*noise_amount)

        return plot_multi_filter(img, filter_size)
    
    interact(
        plot_wrapper,
        img_file=img_files,
        filter_size=size_slider,
        noise=['Salt and pepper', 'Gaussian'],
        noise_amount=noise_amount_slider,
    )