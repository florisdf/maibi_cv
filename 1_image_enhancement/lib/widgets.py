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
    plot_laplacian_kernel,
    plot_sharpening_kernel,
    plot_multi_filter,
    plot_unsharp_mask,
    plot_DoG,
    plot_multi_sharpening,
)

from .denoising_sharpening import (
    add_salt_pepper_noise,
    add_gaussian_noise,
)


DEFAULT_IMGS = ['data/verona_color.jpg', 'data/verona_gray.jpg']


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


def build_plot_wrapper_with_crop(
    func, crop_x_slider, crop_y_slider,
    crop_size_slider
):
    def plot_wrapper(img_file, *args, **kwargs):
        img = read_and_convert_img(img_file)

        height, width = img.shape[:2]
        crop_x_slider.min = 0
        crop_y_slider.min = 0
        crop_x_slider.max = width
        crop_y_slider.max = height
        crop_size_slider.max = min(height, width, crop_size_slider.max)

        func(img, *args, **kwargs)

    return plot_wrapper


def interact_with_crop(func, x0_value=0, y0_value=0,
                       size_value=10,
                       size_max=100,
                       *args, **kwargs):
    crop_x_slider = widgets.IntSlider(min=0, max=x0_value, value=x0_value, continuous_update=False)
    crop_y_slider = widgets.IntSlider(min=0, max=y0_value, value=y0_value, continuous_update=False)
    crop_size_slider = widgets.IntSlider(min=1, max=size_max, value=size_value, continuous_update=False)

    interact(
        build_plot_wrapper_with_crop(
            func, crop_x_slider, crop_y_slider,
            crop_size_slider
        ),
        *args,
        **kwargs,
        crop_x0=crop_x_slider,
        crop_y0=crop_y_slider,
        crop_size=crop_size_slider,
    )


def interact_img_repr(img_files=['data/verona_bw.png', *DEFAULT_IMGS]):
    return interact_with_crop(
        plot_img_repr,
        img_file=img_files,
        x0_value=782,
        y0_value=186,
        size_value=15,
        size_max=30,
    )


def interact_linear_contrast_enhance(img_files=DEFAULT_IMGS):
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


def interact_gamma_correct(img_files=DEFAULT_IMGS):
    gamma_slider = widgets.FloatLogSlider(min=-1.4, max=1.4, value=1.0, step=0.1, continuous_update=False)
    
    def plot_wrapper(img_file, gamma, log=False):
        img = read_and_convert_img(img_file)
        return plot_gamma_correct(img, gamma, log)

    interact(
        plot_wrapper,
        img_file=img_files,
        gamma=gamma_slider,
    )

    
def interact_hist_equalize(img_files=DEFAULT_IMGS):
    def plot_wrapper(img_file, log=False):
        img = read_and_convert_img(img_file)
        return plot_hist_equalize(img, log)

    interact(
        plot_wrapper,
        img_file=img_files,
    )


def interact_clahe(img_files=DEFAULT_IMGS):
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
    

def interact_mean_kernel(img_files=DEFAULT_IMGS):
    size_slider = widgets.IntSlider(min=1, max=50, value=3, continuous_update=False)

    interact_with_crop(
        plot_mean_kernel,
        img_file=img_files,
        size=size_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_gaussian_kernel(img_files=DEFAULT_IMGS):
    size_slider = widgets.IntSlider(min=2, max=50, value=3, continuous_update=False)

    interact_with_crop(
        plot_gaussian_kernel,
        img_file=img_files,
        size=size_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_laplacian_kernel(img_files=DEFAULT_IMGS):
    interact_with_crop(
        plot_laplacian_kernel,
        img_file=img_files,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_sharpening_kernel(img_files=DEFAULT_IMGS):
    interact_with_crop(
        plot_sharpening_kernel,
        img_file=img_files,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_unsharp_mask(img_files=DEFAULT_IMGS):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)

    interact_with_crop(
        plot_unsharp_mask,
        img_file=img_files,
        size=size_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_DoG(img_files=DEFAULT_IMGS):
    size_1_slider = widgets.IntSlider(min=3, max=31, value=3, step=2, continuous_update=False)
    size_2_slider = widgets.IntSlider(min=5, max=51, value=5, step=2, continuous_update=False)

    interact_with_crop(
        plot_DoG,
        img_file=img_files,
        size_1=size_1_slider,
        size_2=size_2_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_median_kernel(img_files=DEFAULT_IMGS):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)

    interact_with_crop(
        plot_median_kernel,
        img_file=img_files,
        size=size_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )
    
def interact_denoise_with_filters(
    img_files=DEFAULT_IMGS,
):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)
    noise_amount_slider = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.01, continuous_update=False)

    def plot_multi_filter_wrapper(
        img, filter_size, noise, noise_amount,
        crop_x0, crop_y0, crop_size
    ):
        if noise == 'Salt and pepper':
            img = add_salt_pepper_noise(img, amount=1*noise_amount)
        elif noise == 'Gaussian':
            img = add_gaussian_noise(img, 0, 255*noise_amount)

        plot_multi_filter(
            img, filter_size, crop_x0,
            crop_y0, crop_size
        )

    interact_with_crop(
        plot_multi_filter_wrapper,
        img_file=img_files,
        filter_size=size_slider,
        noise=['Salt and pepper', 'Gaussian'],
        noise_amount=noise_amount_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )


def interact_sharpening_with_noise(
    img_files=DEFAULT_IMGS,
):
    size_slider = widgets.IntSlider(min=3, max=51, value=3, step=2, continuous_update=False)
    noise_amount_slider = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.01, continuous_update=False)

    def plot_multi_sharpening_wrapper(
        img, filter_size, noise, noise_amount,
        crop_x0, crop_y0, crop_size
    ):
        if noise == 'Salt and pepper':
            img = add_salt_pepper_noise(img, amount=1*noise_amount)
        elif noise == 'Gaussian':
            img = add_gaussian_noise(img, 0, 255*noise_amount)

        plot_multi_sharpening(
            img, filter_size, crop_x0,
            crop_y0, crop_size
        )

    interact_with_crop(
        plot_multi_sharpening_wrapper,
        img_file=img_files,
        filter_size=size_slider,
        noise=['Salt and pepper', 'Gaussian'],
        noise_amount=noise_amount_slider,
        x0_value=782,
        y0_value=186,
        size_value=100,
        size_max=500,
    )