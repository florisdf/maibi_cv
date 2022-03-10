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
    as three arrays: precision, recall, thresholds. Doing this correctly, however, is not
    trivial when we have multiple classes. The following text goes through it step by step.

    Each query will be classified as the label with which it has the highest similarity. For
    a certain class, queries that are classified as that class are (true or false) positives
    and queries that are classified as another class are (true or false) negatives. As such,
    we can compute the precision and recall for each class.

    Since each classification also has a similarity score, we can choose to put a *threshold*
    on the similarity scores. Setting a threshold value for a certain class means that we
    classify all queries with a score lower than the threshold as *negative* and those with a
    score higher than or equal to the threshold as *positive*. The lowest (resp. highest)
    possible threshold is thus the lowest (resp. highest) similarity score.

    By increasing the threshold value for a certain class from the lowest to the highest
    possible value, we will get less and less queries that are classified as that class. Hence,
    the recall of the classifier will decrease (or stay the same, but never increase). However,
    meanwhile, the precision might increase since we are raising the bar on which scores we
    consider high enough. A PR-curve visualizes this trade-off between precision and recall for
    all possible thresholds.

    The PR-curve that evaluates the classification of a certain class is drawn based on a list
    of scores (`y_score`) and a binary list (`y_true`) that indicates which of the queries truly
    belonged to the chosen class. The scores are sorted from high to low and at each rank the
    precision and recall are evaluated by classifying the samples up until the rank as *positive*
    and those after the rank as *negative*. This is equivalent to choosing thresholds and
    classifying the queries under the threshold as negative and those equal to or above the
    threshold as negative.

    An important issue is how to compose `y_score` and `y_true` correctly for the evaluation of
    the classification of a certain class label, given the similarity matrix. As described above,
    the scores in `y_score` will be sorted and will be considered as a predicted positive
    classification at certain ranks. There are scores in the similarity matrix, however, that will
    *never* lead to a positive classification for the chosen class. First, the similarity scores
    with other classes will obviously never lead to a classification as the chosen class. Therefore,
    these scores should be excluded from `y_score`. Second, the similarity scores with the chosen
    class that are *not* the maximum similarity score of the query, will also *never* lead to a
    positive classification for the chosen class. This is because, when classifying a query, we
    only take the class with maximum similarity score into consideration.

    From the above reasoning, it follows that `y_score` should only consist of scores that have
    a maximum value for the class label that we are evaluating. `y_true` then indicates which of
    the queries corresponding to the scores in `y_scores` truely belong to that class.

    The problem with this is, however, that **we will miss a constant number of false negatives**.
    This is because the queries that *do* belong to the chosen class but that will never be classified
    *as* the chosen class (because there score is not the maximum similarity), will also not
    show up in `y_scores` and `y_true`. Hence, the denominator of the recall will be too small and
    the recall will be overestimated. Luckily, the denominator of the recall is constant as it is
    equal to the number of queries that truely have the chosen label. Hence, we can easily correct
    the recalls by multiplying away the wrong denominator and dividing it again with the correct
    denominator.

    One might suggest to add the non-maximum queries that do belong to the chosen class to `y_scores`
    and `y_true` as well. This, however, will lead to erroneous results, since those scores are not
    guaranteed to be lower than the other scores in `y_scores`, while they *should* be. A more valid
    suggestion would be to assign a score of $-\infty$ to these queries, which illustrates the fact
    that these queries will never be classified as positive. Similarity scores of infinite value
    might not be supported, however (e.g. in `sklearn`).

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
    # y_score should only contain scores that have a maximum value
    # for `label`
    match_idxs = sim_mat.argmax(axis=1)
    label_idx = get_label_idx(label, gallery_labels)
    row_idxs, = np.where(match_idxs == label_idx)
    y_score = sim_mat[row_idxs, match_idxs[row_idxs]]

    # y_score tells which of these truely correspond to `label`
    y_true = query_labels[row_idxs] == label

    # Calculate the PR-curve from y_true and y_score
    p, r, t = precision_recall_curve(y_true, y_score)

    # Compensate for the underestimation of false negatives
    true_num_tp_fn = (query_labels == label).sum()
    selected_num_tp_fn = y_true.sum()
    r *= selected_num_tp_fn / true_num_tp_fn

    return p, r, t


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
    precision, recall, _ = calc_pr_curve(label, sim_mat, gallery_labels,
                                         query_labels)
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def get_label_idx(label, cmat_labels):
    if label not in cmat_labels:
        raise ValueError(f'Label {label} not found')

    idxs = np.where(cmat_labels == label)[0]

    if len(idxs) > 1:
        raise ValueError(f'Label {label} should only occur once')

    return idxs[0]