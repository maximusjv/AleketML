import torch
from torch import Tensor
from typing import Any 
import torch
from torchvision.ops import boxes as ops

from consts import VALIDATION_METRICS
from utils.metrics.utils import ensure_tensor, match_gts_dts


def area_relative_diff(gt_boxes: Tensor, dt_boxes: Tensor) -> float:
    """
    Calculates relative area difference using PyTorch tensors.

    Args:
        gt_boxes (Tensor): Ground truth boxes (N, 4).
        dt_boxes (Tensor): Detected boxes (M, 4).

    Returns:
        float: Relative difference: (Area(dt) - Area(gt)) / Mean_Area.
    """
    if gt_boxes.numel() == 0 and dt_boxes.numel() == 0:
        return 0.0
    # Assume box_area returns tensor of areas for each box
    gt_area_total = ops.box_area(gt_boxes).sum() if gt_boxes.numel() > 0 else torch.tensor(0.0, device=gt_boxes.device)
    dt_area_total = ops.box_area(dt_boxes).sum() if dt_boxes.numel() > 0 else torch.tensor(0.0, device=dt_boxes.device)

    mean_area = (gt_area_total + dt_area_total) / 2.0
    if mean_area == 0:
        return 0.0
    # Use .item() to return a standard Python float
    return ((dt_area_total - gt_area_total) / mean_area).item()


def count_relative_diff(gt_boxes: Tensor, dt_boxes: Tensor) -> float:
    """
    Calculates relative count difference using PyTorch tensors.

    Args:
        gt_boxes (Tensor): Ground truth boxes (N, 4).
        dt_boxes (Tensor): Detected boxes (M, 4).

    Returns:
        float: Relative difference: (Count(dt) - Count(gt)) / Mean_Count.
    """
    gt_count = gt_boxes.shape[0]
    dt_count = dt_boxes.shape[0]

    if gt_count == 0 and dt_count == 0:
        return 0.0

    mean_count = (gt_count + dt_count) / 2.0
    if mean_count == 0:
        return 0.0 # Should not happen if counts are non-zero, but safe check
    return (dt_count - gt_count) / mean_count




