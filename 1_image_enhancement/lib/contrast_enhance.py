import cv2
import numpy as np
from .utils import lightness_only


def linear_contrast_enhance(img, alpha, beta):
    new_img = float(alpha) * img + beta
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img


def gamma_correct(img, gamma):
    new_img = np.power(img/255, gamma) * 255
    new_img = new_img.astype(np.uint8)
    return new_img


@lightness_only
def hist_equalize(img):
    mapping = get_hist_eq_mapping(img)
    new_img = mapping[img]
    new_img = new_img.astype(np.uint8)
    return new_img


def get_hist_eq_mapping(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    scaled_cdf = (cdf / cdf.max()) * 255

    return scaled_cdf


@lightness_only
def clahe(img, clip_limit=2.0, grid_size=8):
    cv_clahe = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(grid_size,grid_size))
    return cv_clahe.apply(img)
        