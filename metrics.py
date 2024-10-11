# Standard Library
import io
from contextlib import redirect_stdout

# Third-party Libraries
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# PyTorch
from torch.utils.data import Dataset





COCO_STATS_NAMES = ["AP@.50:.05:.95", "AP@0.5", "AP@0.75",
                    "AP small", "AP medium", "AP large", "AR max=1",
                    "AR max=10", "AR max=100", "AR small", "AR medium", "AR large"]
LOSSES_NAMES = ["loss", "loss_classifier", "loss_box_reg", 'loss_objectness', 'loss_rpn_box_reg']

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


