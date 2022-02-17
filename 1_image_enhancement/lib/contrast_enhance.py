import cv2
import numpy as np


def linear_contrast_enhance(img, alpha, beta):
    new_img = alpha * img + beta
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img


def gamma_correct(img, gamma):
    new_img = np.power(img/255, gamma) * 255
    new_img = new_img.astype(np.uint8)
    return new_img


def hist_equalize(img):
    if img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    rel_cdf = cdf / cdf.max()

    new_img = rel_cdf[img] * 255
    new_img = new_img.astype(np.uint8)

    return new_img
    


def clahe(img, clip_limit=2.0, grid_size=8):
    cv_clahe = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(grid_size,grid_size))

    if img.shape[-1] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = cv_clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        return cv_clahe.apply(img)
        