import numpy as np


# Based on https://minibatchai.com/cv/detection/2021/07/18/VectorizingIOU.html


def get_box_area(corners: np.array) -> float:
    """
    Calculate the area of a box given the
    corners:

    Args:
      corners: float array of shape (N, 4)
        with the values [x1, y1, x2, y2] for
        each batch element.

    Returns:
      area: (N, 1) tensor of box areas for
        all boxes in the batch
    """
    x1 = corners[..., 0]
    y1 = corners[..., 1]
    x2 = corners[..., 2]
    y2 = corners[..., 3]

    return (x2 - x1) * (y2 - y1)


def get_pairwise_intersection_boxes(boxes1, boxes2):
    """
    Calculate the intersection box for each pair
    of boxes.

    Args:
      boxes1: array of shape (N, 4)
      boxes2: array of shape (M, 4)
    Returns:
      intersection_boxes: array of shape (N, M, 4) giving
        the intersection boxes.
    """
    x1 = np.maximum(boxes1[..., 0][..., None],
                    boxes2[..., 0][None, ...])
    y1 = np.maximum(boxes1[..., 1][..., None],
                    boxes2[..., 1][None, ...])
    x2 = np.minimum(boxes1[..., 2][..., None],
                    boxes2[..., 2][None, ...])
    y2 = np.minimum(boxes1[..., 3][..., None],
                    boxes2[..., 3][None, ...])

    return np.stack([x1, y1, x2, y2], axis=-1)


def get_pairwise_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Calculate the intersection over union for each pair
    of bounding boxes.

    Args:
      boxes1: array of shape (N, 4)
      boxes2: array of shape (M, 4)
    Returns:
      iou: array of shape (N, M) giving
        the intersection over union of each pair of boxes.
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    intersection_boxes = get_pairwise_intersection_boxes(boxes1,
                                                         boxes2)
    assert intersection_boxes.shape == (N, M, 4)

    intersection_area = get_box_area(intersection_boxes)
    box1_area = get_box_area(boxes1)
    box2_area = get_box_area(boxes2)

    union_area = (box1_area[..., None] + box2_area[None, ...]) - intersection_area

    # If x1 is greater than x2 or y1 is greater than y2
    # then there is no overlap in the bounding boxes.
    # Find the indices where there is a valid overlap.
    x1 = intersection_boxes[...,  0]
    y1 = intersection_boxes[...,  1]
    x2 = intersection_boxes[...,  2]
    y2 = intersection_boxes[...,  3]

    valid = np.logical_and(x1 <= x2, y1 <= y2)
    assert valid.shape == (N, M)

    # For the valid overlapping boxes, calculate the intersection
    # over union. For the invalid overlaps, set the value to 0.  
    iou = np.where(valid, (intersection_area / union_area), 0)

    return iou