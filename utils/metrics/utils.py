from typing import Any
import torch
from torchvision.ops import boxes as ops

# Helper function to ensure tensor and correct device
def ensure_tensor(data: Any, ref_tensor: torch.Tensor = None) -> torch.Tensor:
    """Converts data to a tensor if it isn't already, placing it on the ref_tensor's device."""
    if isinstance(data, torch.Tensor):
        if ref_tensor is not None and data.device != ref_tensor.device:
            return data.to(ref_tensor.device)
        return data
    # If not a tensor, create one. Use ref_tensor's device if available, else default (e.g., CPU).
    device = ref_tensor.device if ref_tensor is not None else None
    # Attempt conversion, assuming data is list-like or numpy-like if not tensor
    try:
        return torch.as_tensor(data, device=device)
    except Exception as e:
        # Handle cases where conversion isn't straightforward (e.g., None, incompatible types)
        # This might need specific handling based on expected data types
        print(
            f"Warning: Could not convert data of type {type(data)} to tensor: {e}. Returning empty tensor."
        )
        return torch.empty(0, device=device)  # Or raise error, or return None


def match_gts_dts(
    gts: torch.Tensor, dts: torch.Tensor, iou_thresh: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matches ground truth (GT) boxes to detected (DT) boxes based on IoU using a greedy approach.

    Assumes DT boxes are sorted by confidence score in descending order. Each DT box is matched
    to the available GT box with the highest IoU >= iou_thresh. Once a GT box is matched,
    it cannot be matched again.

    Args:
        gts (torch.Tensor): A tensor of shape (N, 4) representing ground truth boxes [x1, y1, x2, y2].
        dts (torch.Tensor): A tensor of shape (M, 4) representing detected boxes [x1, y1, x2, y2],
                            sorted by confidence score (descending).
        iou_thresh (float): The IoU threshold used for matching.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - gt_matches: A tensor of shape (N,) containing the index of the matched DT box
                          for each GT box, or -1 if unmatched.
            - dt_matches: A tensor of shape (M,) containing the index of the matched GT box
                          for each DT box, or -1 if unmatched.
    """
    num_gts = gts.shape[0]
    num_dts = dts.shape[0]

    if num_gts == 0 or num_dts == 0:
        # Handle empty inputs: return tensors of -1s with correct shape and device
        gt_matches = torch.full((num_gts,), -1, dtype=torch.long, device=gts.device)
        dt_matches = torch.full((num_dts,), -1, dtype=torch.long, device=dts.device)
        return gt_matches, dt_matches

    # Ensure tensors are on the same device, default to dts device if different
    if gts.device != dts.device:
        gts = gts.to(dts.device)
        # Consider raising an error or warning if devices differ unexpectedly
        # print(f"Warning: Moving gts tensor to device {dts.device} to match dts.")

    # Calculate IoU matrix ( M x N )
    # rows correspond to detections (dts), columns correspond to ground truths (gts)
    iou_matrix = ops.box_iou(dts, gts)  # Shape: (num_dts, num_gts)

    # Initialize matches with -1 (indicating no match)
    # dt_matches[i] = index of GT matched to DT i, or -1
    # gt_matches[j] = index of DT matched to GT j, or -1
    dt_matches = torch.full((num_dts,), -1, dtype=torch.long, device=dts.device)
    gt_matches = torch.full((num_gts,), -1, dtype=torch.long, device=gts.device)

    # Keep track of which GTs have been matched (use boolean mask for efficiency)
    gt_taken = torch.zeros(num_gts, dtype=torch.bool, device=gts.device)

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

        if potential_match_mask.any():
            # If there are potential matches, find the best one (highest IoU)
            # We only consider IoUs where potential_match_mask is True
            # Set IoUs of non-potential matches to a low value (-1) so argmax ignores them
            ious_filtered = torch.where(
                potential_match_mask, ious, torch.tensor(-1.0, device=ious.device)
            )
            best_gt_idx = torch.argmax(
                ious_filtered
            )  # Index relative to the full gts list

            # Assign the match
            dt_matches[dt_idx] = best_gt_idx
            gt_matches[best_gt_idx] = dt_idx
            gt_taken[best_gt_idx] = True  # Mark this GT as taken

    return gt_matches, dt_matches


def match_dts_to_best_gt(
    gts: torch.Tensor, dts: torch.Tensor, iou_thresh: float
) -> torch.Tensor:
    """
    Matches each detected (DT) box to the ground truth (GT) box with the highest IoU,
    provided the IoU is >= iou_thresh. Multiple DT boxes can match the same GT box.

    Args:
        gts (torch.Tensor): A tensor of shape (N, 4) representing ground truth boxes [x1, y1, x2, y2].
        dts (torch.Tensor): A tensor of shape (M, 4) representing detected boxes [x1, y1, x2, y2].
                            (Sorting by score is not strictly required for this logic).
        iou_thresh (float): The minimum IoU threshold for a match to be considered valid.

    Returns:
        torch.Tensor: A tensor of shape (M,) containing the index of the best matching GT box
                      for each DT box, or -1 if no GT box meets the iou_thresh.
    """
    num_gts = gts.shape[0]
    num_dts = dts.shape[0]

    if num_gts == 0 or num_dts == 0:
        # Handle empty inputs: return tensor of -1s with correct shape and device
        dt_matches = torch.full((num_dts,), -1, dtype=torch.long, device=dts.device)
        return dt_matches

    # Ensure tensors are on the same device, default to dts device if different
    if gts.device != dts.device:
        gts = gts.to(dts.device)
        # Consider logging a warning or raising an error if devices differ unexpectedly
        # print(f"Warning: Moving gts tensor to device {dts.device} to match dts.")

    # Calculate IoU matrix ( M x N )
    # iou_matrix[i, j] is the IoU between dt[i] and gt[j]
    iou_matrix = ops.box_iou(dts, gts)  # Shape: (num_dts, num_gts)

    # Find the highest IoU and the corresponding GT index for each DT (along the GT dimension, dim=1)
    # max_ious shape: (num_dts,) - contains the max IoU value for each DT
    # best_gt_indices shape: (num_dts,) - contains the index of the GT with max IoU for each DT
    max_ious, best_gt_indices = torch.max(iou_matrix, dim=1)

    # Determine which matches meet the threshold
    # valid_match_mask shape: (num_dts,) - True where max_iou >= iou_thresh
    valid_match_mask = max_ious >= iou_thresh

    # Create the result tensor:
    # - where valid_match_mask is True, use the corresponding index from best_gt_indices
    # - where valid_match_mask is False, use -1
    dt_matches = torch.where(
        valid_match_mask,
        best_gt_indices,
        torch.tensor(
            -1, dtype=torch.long, device=dts.device
        ),  # Use -1 for no valid match
    )

    return dt_matches
