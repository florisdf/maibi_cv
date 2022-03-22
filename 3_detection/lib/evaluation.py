import numpy as np

from .iou import get_pairwise_intersection_boxes, get_pairwise_iou


def get_tp_fp_fn_boxes(true_boxes, true_labels, pred_boxes, pred_labels,
                       pos_label, iou_thresh=0.5):
    pair_ious = get_pairwise_iou(true_boxes, pred_boxes)
    max_iou_idx_per_pred = pair_ious.argmax(axis=0)

    ious_pred = pair_ious[max_iou_idx_per_pred, np.arange(pair_ious.shape[1])]
    true_labels_overlap = true_labels[max_iou_idx_per_pred]

    tp_mask = (
        (ious_pred >= iou_thresh)
        & (pred_labels == pos_label)
        & (true_labels_overlap == pred_labels)
    )
    fp_mask = (
        (pred_labels == pos_label)
        & (
            (true_labels_overlap != pred_labels)
            | (ious_pred < iou_thresh)
        )
    )
    tp_boxes = pred_boxes[tp_mask]
    fp_boxes = pred_boxes[fp_mask]

    if pair_ious.shape[1] == 0:
        fn_boxes = true_boxes[true_labels == pos_label]
    else:
        max_iou_idx_per_true = pair_ious.argmax(axis=1)
        ious_true = pair_ious[np.arange(pair_ious.shape[0]),
                              max_iou_idx_per_true]
        pred_labels_overlap = pred_labels[max_iou_idx_per_true]
        fn_mask = (
            (true_labels == pos_label)
            & (
                (pred_labels_overlap != true_labels)
                | (ious_true < iou_thresh)
            )
        )
        fn_boxes = true_boxes[fn_mask]

    return tp_boxes, fp_boxes, fn_boxes