# Standard Library
import io
import copy
from typing import Optional
from contextlib import redirect_stdout
import time
import math

# Third-party Libraries
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# PyTorch
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Torchvision
import torchvision.models.detection as tv_detection

# Utils
from utils import TrainingLogger


# COCO METRICS UTILS
def convert_to_coco_api(dataset: Dataset):
    """Converts a custom dataset to COCO API format.
    Args:
        dataset: The custom dataset to convert.

    Returns:
        A COCO dataset object.
    """

    coco_api_dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ann_id = 1

    for idx in range(len(dataset)):
        img, targets = dataset[idx]
        img_id = targets["image_id"]

        img_entry = {"id": img_id, "height": img.shape[-2], "width": img.shape[-1]}
        coco_api_dataset["images"].append(img_entry)

        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh (coco format)
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        for i in range(len(labels)):
            ann = {
                "image_id": img_id,
                "bbox": bboxes[i],
                "category_id": labels[i],
                "area": areas[i],
                "iscrowd": iscrowd[i],
                "id": ann_id,
            }
            categories.add(labels[i])
            coco_api_dataset["annotations"].append(ann)
            ann_id += 1

    coco_api_dataset["categories"] = [
        {"id": i} for i in sorted(categories)
    ]  # TODO add names

    with redirect_stdout(io.StringIO()):  # Suppress COCO output during creation
        coco_ds = COCO()
        coco_ds.dataset = coco_api_dataset
        coco_ds.createIndex()
    return coco_ds


def stats_dict(stats: np.ndarray):
    """Creates a dictionary of COCO evaluation statistics.
    Args:
        stats: A numpy array containing COCO statistics.
    Returns:
        A dictionary mapping statistic names to their values.
    """
    # According to https://cocodataset.org/#detection-eval
    return {
        "AP@.50:.05:.95": stats[0],
        "AP@0.5": stats[1],
        "AP@0.75": stats[2],
        "AP small": stats[3],
        "AP medium": stats[4],
        "AP large": stats[5],
        "AR max=1": stats[6],
        "AR max=10": stats[7],
        "AR max=100": stats[8],
        "AR small": stats[9],
        "AR medium": stats[10],
        "AR large": stats[11],
    }


class CocoEvaluator:
    """Evaluates object detection predictions using COCO metrics."""

    def __init__(self, gt_dataset):
        """Initializes the CocoEvaluator.
        Args:
            gt_dataset: The ground truth dataset, either a Dataset or a COCO dataset object.
        """
        if isinstance(gt_dataset, Dataset):
            gt_dataset = convert_to_coco_api(gt_dataset)

        self.coco_gt = copy.deepcopy(gt_dataset)
        self.coco_dt = []
        self.img_ids = set()

    def append(self, predictions: list[dict]):
        """Appends predictions to the evaluator.
        Args:
            predictions: A dictionary mapping image IDs to prediction dictionaries.
        """
        for image_id, prediction in predictions.items():
            if not prediction:  # Check if prediction is empty
                continue
            if image_id in self.img_ids:
                raise ValueError(f"Duplicate prediction for image ID: {image_id}")
            self.img_ids.add(image_id)

            boxes = prediction["boxes"].clone()
            boxes[:, 2:] -= boxes[:, :2]
            boxes = boxes.tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            self.coco_dt.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )

    def eval(self):
        """Evaluates the accumulated predictions.
        Returns:
            A dictionary of COCO evaluation statistics.
        """
        if not self.coco_dt:
            print("NO PREDICTIONS")
            return stats_dict(np.zeros(12))
        with redirect_stdout(io.StringIO()):  # Suppress COCO output during evaluation
            coco_dt = self.coco_gt.loadRes(self.coco_dt)
            coco = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
            coco.evaluate()
            coco.accumulate()
            coco.summarize()
        return stats_dict(coco.stats)


def filter_predictions_by_conf(predictions: list, conf_thresh: float) -> list:
    """Filters predictions based on a confidence threshold.

    Args:
        predictions: List of prediction dictionaries.
        conf_thresh: Confidence threshold for filtering.

    Returns:
        List of filtered prediction dictionaries.
    """
    filtered_predictions = []
    for prediction in predictions:
        keep_indices = torch.where(prediction["scores"] > conf_thresh)[0]
        filtered_prediction = {k: v[keep_indices] for k, v in prediction.items()}
        filtered_predictions.append(filtered_prediction)
    return filtered_predictions


def train_one_epoch(
    model: tv_detection.FasterRCNN,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    logger: TrainingLogger,
) -> dict[str, float]:
    """Trains the model for one epoch.
    Args:
        model: The Faster R-CNN model.
        optimizer: The optimizer for training.
        dataloader: The training dataloader.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        epoch: The current epoch number.
        print_freq: Frequency of logging (in batches).

    Returns:
        The average loss for the epoch.
    """
    model.train()
    size = len(dataloader)
    
    loss_values = {
        'loss': torch.zeros(size, dtype=torch.float32),
        'loss_classifier': torch.zeros(size, dtype=torch.float32),
        'loss_box_reg': torch.zeros(size, dtype=torch.float32),
        'loss_objectness': torch.zeros(size, dtype=torch.float32),
        'loss_rpn_box_reg': torch.zeros(size, dtype=torch.float32),
    }

    for batch_num, (images, targets) in enumerate(dataloader):
        start_time = time.time()

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        
        for loss_name, value in losses.items():
            loss_values[loss_name][batch_num] = value
            
        loss_values['loss'][batch_num] = loss

        if not math.isfinite(loss):
            print(f"Loss is {loss.item()}, stopping training")
            raise Exception("Loss is infinite")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_elapsed = time.time() - start_time
        logger.log_batch(batch_num, size,time_elapsed,losses, loss.item()  )

    for loss_name, loss_list in loss_values.items():
            loss_values[loss_name] = loss_list.mean().item()
            
    return loss_values


def evaluate(
    model: tv_detection.FasterRCNN,
    dataloader: DataLoader,
    device: str,
    logger: TrainingLogger,
    conf_thresh: float = 0.1,
) -> dict[str, float]:
    """Evaluates the model on a given dataloader.
    Args:
        model: The Faster R-CNN model.
        dataloader: The evaluation dataloader.
        device: The device to use for evaluation (e.g., 'cuda' or 'cpu').
        conf_thresh: Confidence threshold for filtering predictions.
        print_freq: Frequency of logging (in batches).

    Returns:
        COCO evaluation statistics.
    """
    size = len(dataloader)
    model.eval()

    start_time = time.time()
    coco_evaluator = CocoEvaluator(dataloader.dataset)

    for batch_num, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            predictions = filter_predictions_by_conf(predictions, conf_thresh)
            res = {
                target["image_id"]: output
                for target, output in zip(targets, predictions)
            }
            coco_evaluator.append(res)

        time_elapsed = time.time() - start_time
        logger.log_batch(batch_num, size, time_elapsed)

    stats = coco_evaluator.eval()
    return stats
