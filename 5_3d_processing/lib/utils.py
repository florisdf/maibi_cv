import numpy as np


def pack_rgb(colors):
    """
    Convert an (N, 3) float array of RGB colors into an (N,) integer
    array of int-packed RGB colors.
    """
    r, g, b = (colors * 255).astype(np.uint32).T
    return (r << 16) + (g << 8) + b