from typing import Any, Tuple
import numpy as np

def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Calculate the area of bounding boxes.

    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format.

    Returns:
        Array of shape (N,) with the area of each box.
    """
    width = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    height = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    return width * height

def get_classify_ground_truth(predict_boxes: np.ndarray,
                              gt_boxes: np.ndarray,
                              gt_classes: np.ndarray,
                              background_iou_thresh: float = 0.5):
    best_match = match_dts_to_best_gt(gt_boxes, predict_boxes, background_iou_thresh)
    gt_classes = np.concatenate((gt_classes, np.array([0])))
    filtered_best_match = np.where(best_match > -1, best_match, np.zeros_like(best_match))
    return np.where(best_match > -1, gt_classes[filtered_best_match], np.zeros_like(best_match, dtype=gt_classes.dtype))


# Helper function to ensure array
def ensure_array(data: Any) -> np.ndarray:
    """Converts data to a numpy array if it isn't already."""
    if isinstance(data, np.ndarray):
        return data
    # Attempt conversion, assuming data is list-like or compatible
    try:
        return np.asarray(data)
    except Exception as e:
        print(
            f"Warning: Could not convert data of type {type(data)} to array: {e}. Returning empty array."
        )
        return np.empty(0)  # Or raise error, or return None


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between boxes1 and boxes2.
    
    Args:
        boxes1: array of shape (N, 4) with [x1, y1, x2, y2] format
        boxes2: array of shape (M, 4) with [x1, y1, x2, y2] format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expand dimensions to compute IoU matrix
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    
    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / np.maximum(union, 1e-10)


def match_gts_dts(
    gts: np.ndarray, dts: np.ndarray, iou_thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches ground truth (GT) boxes to detected (DT) boxes based on IoU using a greedy approach.

    Assumes DT boxes are sorted by confidence score in descending order. Each DT box is matched
    to the available GT box with the highest IoU >= iou_thresh. Once a GT box is matched,
    it cannot be matched again.

    Args:
        gts: Array of shape (N, 4) representing ground truth boxes [x1, y1, x2, y2].
        dts: Array of shape (M, 4) representing detected boxes [x1, y1, x2, y2],
             sorted by confidence score (descending).
        iou_thresh: The IoU threshold used for matching.

    Returns:
        Tuple containing two arrays:
        - gt_matches: Array of shape (N,) containing the index of the matched DT box
                      for each GT box, or -1 if unmatched.
        - dt_matches: Array of shape (M,) containing the index of the matched GT box
                      for each DT box, or -1 if unmatched.
    """
    num_gts = gts.shape[0]
    num_dts = dts.shape[0]

    if num_gts == 0 or num_dts == 0:
        # Handle empty inputs
        gt_matches = np.full((num_gts,), -1, dtype=np.int64)
        dt_matches = np.full((num_dts,), -1, dtype=np.int64)
        return gt_matches, dt_matches

    # Calculate IoU matrix (M x N)
    # rows correspond to detections (dts), columns correspond to ground truths (gts)
    iou_matrix = box_iou(dts, gts)  # Shape: (num_dts, num_gts)

    # Initialize matches with -1 (indicating no match)
    dt_matches = np.full((num_dts,), -1, dtype=np.int64)
    gt_matches = np.full((num_gts,), -1, dtype=np.int64)

    # Keep track of which GTs have been matched
    gt_taken = np.zeros(num_gts, dtype=bool)

    # Iterate through detections (already sorted by score)
    for dt_idx in range(num_dts):
        # Get IoU values for the current detection with all GTs
        ious = iou_matrix[dt_idx, :]  # Shape: (num_gts,)

        # Find potential GT matches for this DT
        # 1. IoU must be >= threshold
        valid_iou_mask = ious >= iou_thresh
        # 2. GT must not already be taken
        available_gt_mask = ~gt_taken
        # 3. Combine masks: GTs that are both available and meet IoU threshold
        potential_match_mask = valid_iou_mask & available_gt_mask

        if np.any(potential_match_mask):
            # If there are potential matches, find the best one (highest IoU)
            # Set IoUs of non-potential matches to a low value (-1) so argmax ignores them
            ious_filtered = np.where(potential_match_mask, ious, -1.0)
            best_gt_idx = np.argmax(ious_filtered)

            # Assign the match
            dt_matches[dt_idx] = best_gt_idx
            gt_matches[best_gt_idx] = dt_idx
            gt_taken[best_gt_idx] = True  # Mark this GT as taken

    return gt_matches, dt_matches


def match_dts_to_best_gt(
    gts: np.ndarray, dts: np.ndarray, iou_thresh: float
) -> np.ndarray:
    """
    Matches each detected (DT) box to the ground truth (GT) box with the highest IoU,
    provided the IoU is >= iou_thresh. Multiple DT boxes can match the same GT box.

    Args:
        gts: Array of shape (N, 4) representing ground truth boxes [x1, y1, x2, y2].
        dts: Array of shape (M, 4) representing detected boxes [x1, y1, x2, y2].
             (Sorting by score is not strictly required for this logic).
        iou_thresh: The minimum IoU threshold for a match to be considered valid.

    Returns:
        Array of shape (M,) containing the index of the best matching GT box
        for each DT box, or -1 if no GT box meets the iou_thresh.
    """
    num_gts = gts.shape[0]
    num_dts = dts.shape[0]

    if num_gts == 0 or num_dts == 0:
        # Handle empty inputs
        dt_matches = np.full((num_dts,), -1, dtype=np.int64)
        return dt_matches

    # Calculate IoU matrix (M x N)
    # iou_matrix[i, j] is the IoU between dt[i] and gt[j]
    iou_matrix = box_iou(dts, gts)  # Shape: (num_dts, num_gts)

    # Find the highest IoU and the corresponding GT index for each DT
    max_ious = np.max(iou_matrix, axis=1)  # Shape: (num_dts,)
    best_gt_indices = np.argmax(iou_matrix, axis=1)  # Shape: (num_dts,)

    # Determine which matches meet the threshold
    valid_match_mask = max_ious >= iou_thresh

    # Create the result array
    dt_matches = np.where(valid_match_mask, best_gt_indices, -1)

    return dt_matches