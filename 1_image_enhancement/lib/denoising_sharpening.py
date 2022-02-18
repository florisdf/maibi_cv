import cv2
import numpy as np
    
    
def build_mean_kernel(size):
    return np.ones((size, size))/(size**2)


def build_gaussian_kernel(size, sigma=0):
    k1 = cv2.getGaussianKernel(size, sigma)
    return np.dot(k1, k1.T)


def apply_gaussian_kernel(img, size, sigma=0):
    kernel = build_gaussian_kernel(size, sigma)
    return cv2.filter2D(img, -1, kernel)


def apply_mean_kernel(img, size):
    kernel = build_mean_kernel(size)
    return cv2.filter2D(img, -1, kernel)


def apply_median_kernel(img, size):
    return cv2.medianBlur(img, size)


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
    new_img[coords_y_salt, coords_x_salt, :] = 0
    new_img[coords_y_pepper, coords_x_pepper, :] = 255

    return new_img