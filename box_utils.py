import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Calculates the area of a set of bounding boxes.

    Args:
        boxes (np.ndarray): A NumPy array of shape (N, 4) representing bounding boxes.
                             Each row is [x1, y1, x2, y2], where (x1, y1) is the top-left
                             corner and (x2, y2) is the bottom-right corner of the box.

    Returns:
        np.ndarray: A NumPy array of shape (N) containing the area of each box.
    """
    return (
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if len(boxes) > 0
        else np.zeros(0)
    )


def box_inter_union(
    boxes1: np.ndarray, boxes2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the intersection and union areas between two sets of bounding boxes.

    Args:
        boxes1 (np.ndarray): An array of shape (N, 4) representing the first set of boxes.
                             Each row is [x1, y1, x2, y2], where (x1, y1) is the top-left
                             corner and (x2, y2) is the bottom-right corner of the box.
        boxes2 (np.ndarray): An array of shape (M, 4) representing the second set of boxes.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       - inter: An array of shape (N, M) representing the
                                                intersection area between each pair of boxes.
                                       - union: An array of shape (N, M) representing the
                                                union area between each pair of boxes.
    """
    area1 = box_area(boxes1)  # Assuming you have the NumPy version of box_area
    area2 = box_area(boxes2)

    lt = np.maximum(
        boxes1[:, np.newaxis, :2], boxes2[:, :2]
    )  # np.maximum for element-wise max
    rb = np.minimum(
        boxes1[:, np.newaxis, 2:], boxes2[:, 2:]
    )  # np.minimum for element-wise min

    wh = np.clip((rb - lt), a_min=0, a_max=None)  # np.clip for clamping
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, np.newaxis] + area2 - inter

    return inter, union


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculates the IoU (Intersection over Union) matrix between two sets of boxes.

    Args:
        boxes1 (np.ndarray): An array of shape (N, 4) representing the first set of boxes.
                             Each row is [x1, y1, x2, y2], where (x1, y1) is the top-left
                             corner and (x2, y2) is the bottom-right corner of the box.
        boxes2 (np.ndarray): An array of shape (M, 4) representing the second set of boxes.

    Returns:
        np.ndarray: An array of shape (N, M) representing the IoU between each pair of boxes.
    """
    inter, union = box_inter_union(boxes1, boxes2)
    iou_matrix = inter / union
    return iou_matrix


def match_gts_dts(
    gts: np.ndarray, dts: np.ndarray, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches ground truth (GT) boxes to detected (DT) boxes based on IoU.

    Args:
        gts (np.ndarray): An array of shape (N, 4) representing ground truth boxes.
        dts (np.ndarray): An array of shape (M, 4) representing detected boxes.
        iou_thresh (float): The IoU threshold used for matching.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       - gt_matches: A binary array of shape (N,) indicating
                                                     whether each GT box has a match (1) or not (0).
                                       - dt_matches: A binary array of shape (M,) indicating
                                                     whether each DT box has a match (1) or not (0).
    """
    iou_matrix = box_iou(dts, gts)

    # Initialize matches
    dt_matches = np.zeros(len(dts))
    gt_matches = np.zeros(len(gts))

    for dind, _ in enumerate(dts):
        iou = iou_thresh
        match = -1
        for gind, _ in enumerate(gts):
            # If GT already matched
            if gt_matches[gind] != 0:
                continue
            # Continue to next GT unless better match made
            if iou_matrix[dind, gind] < iou:
                continue
            # If match successful and best so far, store appropriately
            iou = iou_matrix[dind, gind]
            match = gind

        if match != -1:
            dt_matches[dind] = 1
            gt_matches[match] = 1

    return gt_matches, dt_matches
