import matplotlib.pyplot as plt
import numpy as np

from .cnn_classifiers import val_resize_crop


def plot_transformed_val_input(im):
    im_resize_cropped = val_resize_crop(im)

    fig, axes = plt.subplots(ncols=2, figsize=(6, 12))

    axes[0].set_title('Original image')
    axes[0].imshow(np.array(im))

    axes[1].set_title('Resized + cropped image')
    axes[1].imshow(np.array(im_resize_cropped))


def plot_matches(gallery, queries, match_idxs, gallery_labels=None):
    """
    Plot the matched gallery item for each query.

    Args:
        gallery: A sequence of N_g gallery images
        query: A sequence of N_q query images
        match_idxs: Sequence of length N_q containing for each query the index
            of the matched gallery item.
        gallery_labels: (optional) The label of each item in the gallery.
    """

    nrows = len(match_idxs)
    ncols = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 6*nrows))

    for i, match_idx in enumerate(match_idxs):
        q_img = queries[i]
        match_img = gallery[match_idx]

        axes[i][0].set_title(f'Query {i}')
        axes[i][0].imshow(q_img)

        if gallery_labels is not None:
            match_text = f'Match: {gallery_labels[match_idx]}'
        else:
            match_text = f'Match: {match_idx}'
        axes[i][1].set_title(match_text)
        axes[i][1].imshow(match_img)


def plot_sim_mat(sim_mat, gallery_labels=None, query_labels=None):
    """
    Plot a given similarity matrix.
    
    Args:
        sim_mat: The similarity matrix.
        gallery_labels: (optional) The gallery labels; will be shown on top of each column.
        query_labels: (optional) The query labels; will be shown to the left of each row.
    """
    N_Q, N_G = sim_mat.shape

    fig, ax = plt.subplots(figsize=(N_G/2, N_Q/2))
    im = ax.imshow(sim_mat)

    ax.set_xlabel('Gallery item')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(N_G))

    if gallery_labels is not None:
        ax.set_xticklabels(gallery_labels, rotation=90)

    ax.set_ylabel('Query')
    ax.set_yticks(np.arange(N_Q))

    if query_labels is not None:
        ax.set_xticklabels(query_labels)

    plt.colorbar(im)


def plot_conf_mat(cmat, labels=None):
    """
    Plot a confusion matrix.
    
    Args:
        cmat: The confusion matrix.
        labels: The labels corresponding to the confusion matrix.
    """
    fig, ax = plt.subplots()
    ax.imshow(cmat)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    N_LABELS = len(cmat)

    ax.set_xticks(np.arange(N_LABELS))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)

    ax.set_yticks(np.arange(N_LABELS))
    if labels is not None:
        ax.set_yticklabels(labels)

    for i in range(N_LABELS):
        for j in range(N_LABELS):
            ax.text(
                x=i, y=j, s=str(cmat[i, j]),
                ha="center", va="center",
            )