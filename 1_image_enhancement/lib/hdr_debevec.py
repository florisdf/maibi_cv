# Reimplemented from https://github.com/SSARCandy/HDR-imaging/blob/0ffcc2018ffe6f283d75b0eaae5feef3e2f44fd1/HDR-playground.py

import cv2
import numpy as np
import math


def get_radiometric_response_curve(imgs, exposure_times, n_sample_per_dim=20):
    height, width = imgs.shape[1:3]
    width_step = width // n_sample_per_dim
    height_step = height // n_sample_per_dim

    mask = np.zeros((height, width), dtype=np.bool8)
    mask[::height_step, ::width_step] = True

    gs = []
    for channel in range(imgs.shape[-1]):
        Z = imgs[..., mask, channel]
        g, _ = response_curve_solver(Z, exposure_times)
        gs.append(g)
    
    return np.hstack(gs)


def response_curve_solver(Z, exposure_times, l=50):
    """
    Args:
        l: determines the amount of smoothness
    """
    B = [math.log(t, 2) for t in exposure_times]
    w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
 
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b, rcond=-1)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


def construct_radiance_map(imgs, response_curve, exposure_times):
    Z = imgs.flatten().reshape(len(imgs), -1, 3)

    resp = np.stack([response_curve[Z[..., i], i]
                     for i in range(Z.shape[-1])], axis=-1)
    ln_t = np.log2(exposure_times)[..., None, None]
    w = np.array([z if z <= 0.5*255 else 255 - z
                  for z in range(256)])
    w_z = w[Z]
    acc_e = np.sum(w_z * (resp - ln_t), 0)
    acc_w = np.sum(w_z, 0)
    acc_w[acc_w <= 0] = 1

    E = acc_e/acc_w

    return np.reshape(np.exp(E), imgs.shape[1:])