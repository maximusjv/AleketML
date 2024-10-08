# Standard Library
import io
from contextlib import redirect_stdout
import math

# Third-party Libraries
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# PyTorch
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Torchvision
import torchvision.models.detection as tv_detection


# COCO METRICS UTILS
def convert_to_coco(dataset: Dataset):
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

        bboxes = targets["boxes"]
        
        areas = ((bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])).tolist()
        
        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh (coco format)
        bboxes = bboxes.tolist()
        
        labels = targets["labels"].tolist()
        iscrowd = [0] * len(labels)

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

# METRICS NAMES
COCO_STATS_NAMES = ["AP@.50:.05:.95", "AP@0.5", "AP@0.75",
                    "AP small", "AP medium", "AP large", "AR max=1",
                    "AR max=10", "AR max=100", "AR small", "AR medium", "AR large"]
LOSSES_NAMES = ["loss", "loss_classifier", "loss_box_reg", 'loss_objectness', 'loss_rpn_box_reg']
 

class CocoEvaluator:
    """Evaluates object detection predictions using COCO metrics."""
   
    def __init__(self, gt_dataset):
        """Initializes the CocoEvaluator.
        Args:
            gt_dataset: The ground truth dataset, either a Dataset or a COCO dataset object.
        """
        if isinstance(gt_dataset, Dataset):
            gt_dataset = convert_to_coco(gt_dataset)

        self.coco_gt = gt_dataset
        self.coco_dt = []
        self.img_ids = set()
        
    def clear_detections(self):
        """Clears the stored detection results."""
        
        self.coco_dt = []
        self.img_ids = set()
            
    def append(self, predictions: dict[int, dict]):
        """Appends predictions to the evaluator.
        Args:
            predictions: A dictionary mapping image IDs to prediction dictionaries.
        """
        for image_id, prediction in predictions.items():
            
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
        stats = np.zeros(12)
        if self.coco_dt:
            with redirect_stdout(io.StringIO()):  # Suppress COCO output during evaluation
                coco_dt = self.coco_gt.loadRes(self.coco_dt)
                coco = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
                coco.evaluate()
                coco.accumulate()
                coco.summarize()
                stats = coco.stats
        return {key: value for key, value in zip(COCO_STATS_NAMES, stats)}


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
) -> dict[str, float]:
    """Trains the model for one epoch.
    Args:
        model: The Faster R-CNN model.
        optimizer: The optimizer for training.
        dataloader: The training dataloader.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
    Returns:
        The average loss for the epoch.
    """
    model.train()
    size = len(dataloader)
    
    loss_values = { 
        key: 0 for key in LOSSES_NAMES
    }

    for batch_num, (images, targets) in tqdm(enumerate(dataloader), desc="Training batches", total=size):
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        
        loss_values['loss'] += loss.item()
        for loss_name, value in losses.items():
            loss_values[loss_name] += value.item()
            
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise Exception("Loss is infinite")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for loss_name, value in loss_values.items():
            loss_values[loss_name] = value/size
            
    return loss_values


def evaluate(
    model: tv_detection.FasterRCNN,
    dataloader: DataLoader,
    coco_eval: CocoEvaluator,
    device: str,
) -> dict[str, float]:
    """Evaluates the model on the given dataloader using COCO metrics.
    Args:
        model: The Faster R-CNN model to evaluate.
        dataloader: The dataloader containing the evaluation data.
        coco_eval: The COCO evaluator object for calculating metrics.
        device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the COCO evaluation statistics.
    """
    size = len(dataloader)
    model.eval()

    for batch_num, (images, targets) in tqdm(enumerate(dataloader), desc="Evaluating batches", total=size):
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            res = {
                target["image_id"]: output
                for target, output in zip(targets, predictions)
            }
            coco_eval.append(res)
    
    stats = coco_eval.eval()
    coco_eval.clear_detections()
    
    return stats
