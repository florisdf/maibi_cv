import cv2
import matplotlib.pyplot as plt
import numpy as np


def match_opp_color_hist(gallery, queries):
    """
    Return a N_q x N_g similarity matrix obtained by matching query images with gallery images
    using opponent color histograms, like in Swain & Ballard (1992).

    Args:
        gallery: a sequence of N_g RGB gallery images.
        queries: a sequence of N_q RGB query images.
    """
    gallery_hists = [
        calc_opp_color_hist(img)
        for img in gallery
    ]

    query_hists = [
        calc_opp_color_hist(img)
        for img in queries
    ]

    return np.array([
        [
            opp_hists_match_score(q_hist, g_hist)
            for g_hist in gallery_hists
        ]
        for q_hist in query_hists
    ])


def calc_opp_color_hist(img):
    """
    Computes the opponent color histogram of an image, like in
    Swain & Ballard (1992).

    Args:
        img: The image (should be RGB)
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError('Input image must be RGB.')

    # Cast to int so that negative numbers and numbers > 255 can be represented
    img = img.astype(np.float32)

    # Separate color channels
    r, g, b = img.transpose(2, 0, 1)

    # Compute opponent color channels
    rg, by, wb = rgb_to_opp_color(r, g, b)

    # Image with opponent color channels
    opp_img = cv2.merge([rg, by, wb])

    rg_bins, by_bins, wb_bins = get_opp_color_hist_bins()

    return cv2.calcHist(
        images=[opp_img],
        channels=[0, 1, 2],
        mask=None,
        histSize=[len(rg_bins), len(by_bins), len(wb_bins)],
        ranges=[
            rg_bins.min(), rg_bins.max(),  # rg range
            by_bins.min(), by_bins.max(),  # by range
            wb_bins.min(), wb_bins.max()   # wb range
        ]
    )


def opp_hists_match_score(img_hist, model_hist):
    """
    Computes the matching score between two opponent color histograms, like in
    Swain & Ballard (1992).
    """
    # Element-wise minimum of histograms
    min_hists = np.minimum(img_hist, model_hist)

    # Sum of minima
    intersection = np.sum(min_hists)

    # Normalize the intersection with the model histogram
    match_score = intersection / np.sum(model_hist)

    return match_score


def get_opp_color_hist_bins():
    """
    Return the bins used in the oppenent color histogram.
    """
    # 16 bins in rg channel
    # min value -255 when r = 0,   g = 255 => rg = r - g = -255
    # max value  255 when r = 255, g = 0   => rg = r - g = 255
    rg_bins = np.linspace(-255, 255, 16)

    # 16 bins in by channel
    # min value -510 when r = 255, g = 255, b = 0   => by = 2*b - r - g = -510
    # max value  510 when r = 0,   g = 0,   b = 255 => by = 2*b - r - g = 510
    by_bins = np.linspace(-510, 510, 16)

    # 8 bins in wb channel
    # min value 0   when r = 0,   g = 0,   b = 0   => wb = r + g + b = 0
    # max value 765 when r = 255, g = 255, b = 255 => wb = r + g + b = 765
    wb_bins = np.linspace(0, 765, 8)

    return (
        rg_bins,
        by_bins,
        wb_bins
    )


def rgb_to_opp_color(r, g, b):
    """
    Convert r, g and b values (can be entire arrays as well) to
    opponent colors (i.e. rg, by, wb).
    """
    rg = r - g
    by = 2*b - r - g
    wb = r + g + b
    return rg, by, wb


def opp_color_to_rgb(rg, by, wb):
    """
    Convert opponent color values (can be entire arrays as well)
    back to r, g and b values.
    """
    r = ( 3*rg -   by + 2*wb)//6
    g = (-3*rg -   by + 2*wb)//6
    b = (        2*by + 2*wb)//6
    return r, g, b


## PLOTTING FUNCTIONS ##


def plot_imgs_with_opp_color_hist(imgs, hists, labels=None):
    """
    Plot the opponent color histogram of each image and show the image itself as well.
    
    Args:
        imgs: A sequence of images
        hists: The opponent color histogram of each of the images
        labels: (optional) The image labels. Will be put in the title above the images.
    """
    nrows = 2
    ncols = len(imgs)

    fig = plt.figure(figsize=(6*ncols, 6*nrows))

    img_axes = [
        fig.add_subplot(nrows, ncols, i + 1)
        for i in range(ncols)
    ]

    hist_axes = [
        fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        for i in range(ncols, 2*ncols)
    ]

    for i, img in enumerate(imgs):
        if labels is not None:
            img_axes[i].set_title(labels[i])

        img_axes[i].imshow(img)
        plot_opp_color_hist(hists[i], hist_axes[i])


def plot_opp_color_hist(hist, ax=None):
    """
    Plot an opponent color histogram. The count of a bin is represented by the size of the scatter dot.
    The color of the dot corresponds to that bin.
    
    If you want an interactive 3D-plot, add `%matplotlib notebook` to the notebook cell.
    
    Args:
        hist: The opponent color histogram
        ax: (optional) A matpotlib axis to plot in.
    """
    # Get a flat list of all indices in the 3D histogram
    xs, ys, zs = np.indices(hist.shape).reshape((3, -1))

    # Calculate the RGB color of each bin in the 3D histogram
    rg_bins, by_bins, wb_bins = get_opp_color_hist_bins()
    colors = [opp_color_to_rgb(rg_bins[x], by_bins[y], wb_bins[z])
              for x, y, z in zip(xs, ys, zs)]

    # Convert colors to RGB-values between 0 and 1
    colors = [
        [np.clip(c/255, 0, 1) for c in color]
        for color in colors
    ]

    # Put everything in a 3D plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    # We simply put a dot at each index location
    # and use the histogram count as the dot's size.
    # The color of each dot is taken from the colors list.
    ax.scatter(xs, ys, zs,
               s=hist[xs, ys, zs],
               c=colors)

    ax.set_xlabel('rg')
    ax.set_ylabel('by')
    ax.set_zlabel('wb')
    return ax


## TESTS ##


def test_rgb2opp2rgb():
    """
    Test that rgb_to_opp_color() and opp_color_to_rgb() are indeed each other's inverse.
    """
    r, g, b = np.random.randint(0, 255, 3)
    rg, by, wb = rgb_to_opp_color(r, g, b)
    r2, g2, b2 = opp_color_to_rgb(rg, by, wb)
    assert r == r2 and g == g2 and b == b2