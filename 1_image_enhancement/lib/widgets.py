import cv2
from ipywidgets import interact, widgets


from .plots import plot_img_repr, plot_linear_contrast_enhance, plot_gamma_correct, plot_hist_equalize, plot_clahe


def read_and_convert_img(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


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
    gamma_slider = widgets.FloatSlider(min=0.04, max=25, value=1, continuous_update=False)
    
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