import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def calc_tp_tn_fp_fn(label, cmat, cmat_labels):
    """
    Return the number of true positives (TPs), true negatives (TNs),
    false positives (FPs) and false negatives (FNs) for a given class label.

    Args:
        label: The class label for which to calculate the TPs, TNs, FPs and FNs.
        cmat: The confusion matrix. The rows should correspond to the true labels,
            the columns should correspond to the predicted labels.
        cmat_labels: The class labels in the confusion matrix.
    """
    label_idx = get_label_idx(label, cmat_labels)

    tps = cmat[label_idx, label_idx]
    fps = np.sum(cmat[:, label_idx]) - tps
    fns = np.sum(cmat[label_idx, :]) - tps
    tns = np.sum(cmat) - fps - fns - tps

    return tps, tns, fps, fns



def calc_precision(label, cmat, cmat_labels):
    """
    Return the precision for a given class label.

    Args:
        label: The class label for which to calculate the precision.
        cmat: The confusion matrix. The rows should correspond to the true labels,
            the columns should correspond to the predicted labels.
        cmat_labels: The class labels in the confusion matrix.
    """
    label_idx = get_label_idx(label, cmat_labels)

    tps = cmat[label_idx, label_idx]
    all_positives = np.sum(cmat[:, label_idx])
    
    return tps / all_positives


def calc_recall(label, cmat, cmat_labels):
    """
    Return the recall for a given class label.

    Args:
        label: The class label for which to calculate the recall.
        cmat: The confusion matrix. The rows should correspond to the true labels,
            the columns should correspond to the predicted labels.
        cmat_labels: The class labels in the confusion matrix.
    """
    label_idx = get_label_idx(label, cmat_labels)

    tps = cmat[label_idx, label_idx]
    all_true = np.sum(cmat[label_idx, :])

    return tps / all_true


def calc_pr_curve(label, sim_mat, gallery_labels, query_labels):
    """
    Return the precision-recall curve for a given class label. The PR-curve is returned
    as three arrays: precision, recall, thresholds.

    Args:
        label: The class label for which to calculate the precision-recall curve.
        sim_mat: The similarity matrix. It should have the same amount of rows as
            the number of queries and the same amount of columns as the number of
            class labels in the gallery.
        gallery_labels: The labels that correspond to the columns in `sim_mat`,
            i.e. the labels in the gallery.
        query_labels: The labels that correspond to the rows in `sim_mat`, i.e.
            the true label of each query.
    """
    y_score = get_yscore(label, sim_mat, gallery_labels)

    return precision_recall_curve(query_labels, y_score, pos_label=label)


def calc_ap(label, sim_mat, gallery_labels, query_labels):
    """
    Return the average precision for a given class label.

    Args:
        label: The class label for which to calculate the precision-recall curve.
        sim_mat: The similarity matrix. It should have the same amount of rows as
            the number of queries and the same amount of columns as the number of
            class labels in the gallery.
        gallery_labels: The labels that correspond to the columns in `sim_mat`,
            i.e. the labels in the gallery.
        query_labels: The labels that correspond to the rows in `sim_mat`, i.e.
            the true label of each query.
    """
    y_score = get_yscore(label, sim_mat, gallery_labels)
    y_true = query_labels == label

    return average_precision_score(y_true, y_score)


def get_label_idx(label, cmat_labels):
    if label not in cmat_labels:
        raise ValueError(f'Label {label} not found')

    idxs = np.where(cmat_labels == label)[0]

    if len(idxs) > 1:
        raise ValueError(f'Label {label} should only occur once')

    return idxs[0]


def get_yscore(label, sim_mat, gallery_labels):
    """
    Return the similarity score between the queries and the gallery
    item of a label.
    
    Args:
        label: The class label for which to return y_score.
        sim_mat: The similarity matrix. It should have the same amount of rows as
            the number of queries and the same amount of columns as the number of
            class labels in the gallery.
    """
    g_idx = get_label_idx(label, gallery_labels)
    return sim_mat[:, g_idx]