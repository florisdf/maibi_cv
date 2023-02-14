import cv2
import numpy as np
from .utils import lightness_only
    
    
def build_mean_kernel(size):
    return np.ones((size, size))/(size**2)


def apply_mean_kernel(img, size):
    kernel = build_mean_kernel(size)
    return cv2.filter2D(img, -1, kernel)


def build_gaussian_kernel(size):
    k1 = cv2.getGaussianKernel(size, sigma=0)
    return np.dot(k1, k1.T)


def apply_gaussian_kernel(img, size):
    return cv2.GaussianBlur(img, (size, size), 0)


def build_laplacian_kernel():
    return np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])


def apply_laplacian_kernel(img):
    kernel = build_laplacian_kernel()
    return cv2.filter2D(img, -1, kernel)


def build_sharpening_kernel():
    return np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
    ])


def apply_sharpening_kernel(img):
    kernel = build_sharpening_kernel()
    return cv2.filter2D(img, -1, kernel)


def apply_median_kernel(img, size):
    return cv2.medianBlur(img, size)


def apply_unsharp_mask(img, size):
    smoothed_img = cv2.GaussianBlur(img, (size, size), 0)
    return cv2.addWeighted(img, 2.0, smoothed_img, -1.0, 0)


def apply_DoG(img, size_1, size_2):
    smoothed_img_1 = cv2.GaussianBlur(img, (size_1, size_1), 0)
    smoothed_img_2 = cv2.GaussianBlur(img, (size_2, size_2), 0)
    return smoothed_img_1 - smoothed_img_2


def add_gaussian_noise(img, mean, sigma):
    noise = np.random.normal(mean, sigma, size=img.shape)
    noisy_img = noise + img
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.01):
    width, height = img.shape[0], img.shape[1]
    num_noise = int(np.ceil(amount * width * height))

    coords_y, coords_x = [np.random.randint(0, i, num_noise)
                          for i in img.shape[:2]]
    
    coords_y_salt, coords_y_pepper = np.split(coords_y, 2)
    coords_x_salt, coords_x_pepper = np.split(coords_x, 2)
    
    new_img = np.copy(img)
    new_img[coords_y_salt, coords_x_salt] = 0
    new_img[coords_y_pepper, coords_x_pepper] = 255

    return new_img