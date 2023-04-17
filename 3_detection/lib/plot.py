import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .evaluation import get_tp_fp_fn_boxes


def listize(value, default_value, length):
    if value is None:
        value = [default_value] * length
    elif not isinstance(value, list):
        value = [value] * length

    return value


def plot_boxes(boxes, edgecolors=None, facecolors=None,
               alphas=None, linestyles=None,
               ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    length = len(boxes)
    
    edgecolors = listize(edgecolors, 'none', length)
    alphas = listize(alphas, 1.0, length)
    facecolors = listize(facecolors, 'none', length)
    linestyles = listize(linestyles, '-', length)
    
    for (box, edgecolor, facecolor,
         alpha, linestyle) in zip(boxes, edgecolors,
                                  facecolors, alphas,
                                  linestyles):
        width = box[2] - box[0]
        height = box[3] - box[1]

        rect = patches.Rectangle((box[0], box[1]), width, height,
                                 linewidth=3, edgecolor=edgecolor,
                                 facecolor=facecolor, alpha=alpha,
                                 linestyle=linestyle)

        ax.add_patch(rect)

    return ax


def plot_tp_fp_fn(true_boxes, true_labels, pred_boxes, pred_labels,
                  pos_label, iou_thresh=0.5, ax=None):
    tp_boxes, fp_boxes, fn_boxes = get_tp_fp_fn_boxes(
        true_boxes, true_labels, pred_boxes, pred_labels,
        pos_label, iou_thresh
    )

    ax = plot_boxes(true_boxes, edgecolors='black', ax=ax, linestyles='--')
    plot_boxes(tp_boxes, edgecolors='green', ax=ax)
    plot_boxes(fp_boxes, edgecolors='red', ax=ax)
    plot_boxes(fn_boxes, edgecolors='red', ax=ax, linestyles='--')

    return ax


def pseudo_detection(boxes, labels, all_labels,
                     box_sigma=10,
                     pct_rand_labels=0.1,
                     prob_dropout=0.5):
    boxes = np.copy(boxes)
    labels = np.copy(labels)

    n_rand_labels = int(pct_rand_labels * len(labels))
    rand_label_idxs = np.random.choice(
        np.arange(len(labels)),
        n_rand_labels
    )
    labels[rand_label_idxs] = np.random.choice(
        all_labels, n_rand_labels,
        replace=True
    )

    boxes = boxes + np.random.randn(*boxes.shape) * box_sigma

    drop = [np.random.rand() > prob_dropout
            for _ in range(len(boxes))]
    boxes = boxes[drop]
    labels = labels[drop]

    return boxes, labels


def plot_pseudo_detection(
    dataset,
    img_idx=0,
    box_sigma=0,
    prob_dropout=0.0,
    iou_thresh=0.5,
    pos_label=1,
    all_labels=[1],
    seed=42,
):
    np.random.seed(seed)
    img, target = dataset[img_idx]
    true_boxes = target['boxes'].numpy()
    true_labels = target['labels'].numpy()

    pred_boxes, pred_labels = pseudo_detection(
        true_boxes, true_labels, all_labels,
        box_sigma=box_sigma,
        prob_dropout=prob_dropout
    )

    fig, ax = plt.subplots()
    ax.imshow(img.numpy().transpose(1, 2, 0))

    plot_tp_fp_fn(true_boxes, true_labels, pred_boxes, pred_labels,
                  pos_label=pos_label, iou_thresh=iou_thresh, ax=ax)
    tp_boxes, fp_boxes, fn_boxes = get_tp_fp_fn_boxes(
        true_boxes, true_labels,
        pred_boxes, pred_labels,
        pos_label, iou_thresh=iou_thresh)
    tp = len(tp_boxes)
    fp = len(fp_boxes)
    fn = len(fn_boxes)
    iou_str = f'@{iou_thresh:.2f}'
    tp_str = f'TP{iou_str}'
    fp_str = f'FP{iou_str}'
    fn_str = f'FN{iou_str}'
    print(f'{tp_str:^10}|{fp_str:^10}|{fn_str:^10}')
    print(f'{"-"*10}|{"-"*10}|{"-"*10}')
    print(f'{tp:^10}|{fp:^10}|{fn:^10}')
    print()
    print(f'Precision{iou_str}: {tp/(tp + fp)}')
    print(f'Recall{iou_str}: {tp/(tp + fn)}')