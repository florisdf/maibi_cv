import numpy as np
from ipywidgets import interact, widgets

from .plot import plot_pseudo_detection



def interact_pseudo_detection(dataset, pos_label=1, all_labels=[1]):
    def plot_pseudo_detection_wrapper(img_idx, box_sigma, prob_dropout,
                                      iou_thresh, pos_label, seed):
        return plot_pseudo_detection(
            dataset,
            img_idx=img_idx,
            box_sigma=box_sigma,
            prob_dropout=prob_dropout,
            iou_thresh=iou_thresh,
            pos_label=pos_label,
            all_labels=all_labels,
            seed=seed,
        )
    interact(
        plot_pseudo_detection_wrapper,
        img_idx=widgets.IntSlider(min=0, max=len(dataset), continuous_update=False),
        box_sigma=widgets.IntSlider(min=0, max=100, value=0, continuous_update=False),
        prob_dropout=widgets.FloatSlider(min=0, max=1, value=0, continuous_update=False),
        iou_thresh=widgets.FloatSlider(min=0, max=1, value=0.5, continuous_update=False),
        pos_label=all_labels,
        seed=widgets.IntSlider(min=0, max=50, continuous_update=False),
    )