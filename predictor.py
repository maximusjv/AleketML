# Standard Library
import csv
import os
from typing import  Optional

# Third-party Libraries
import PIL
import numpy as np
from PIL.Image import Image

# PyTorch
import torch

# Torchvision
import torchvision.tv_tensors as tv_tensors
from torchvision.models.detection import FasterRCNN
import torchvision.transforms.v2 as v2
from torchvision.ops import batched_nms

from aleket_dataset import AleketDataset
from dataset_statisics import visualize_bboxes
from metrics import LOSSES_NAMES, VALIDATION_METRICS, Evaluator
from utils import make_patches


class Predictor:
    def __init__(
        self,
        model: FasterRCNN,
        device: torch.device,
        detections_per_patch: int = 100,
        patch_size: int = 1024,
        patch_overlap: float = 0.2,
    ):

        self.device = device
        self.classes = list(AleketDataset.NUM_TO_CLASSES.values())
        self.classes = self.classes[1:] # remove background class
        self.model = model.to(device)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch

    def get_detections(
        self,
        image: Image | torch.Tensor,
        nms_thresh: float,
        score_thresh: float,
    ) -> dict[str, torch.Tensor]:
        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = nms_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch

        if isinstance(image, torch.Tensor):
            image = v2.functional.to_pil_image(image)

        patched_images, patches = make_patches(
            image, self.patch_size, self.patch_overlap
        )

        boxes = []
        labels = []
        scores = []

        with torch.no_grad():
            self.model.eval()
            for img, box in zip(patched_images, patches):
                xmin, ymin = box[0], box[1]
                img = v2.functional.to_dtype(
                    v2.functional.to_image(img), dtype=torch.float32, scale=True
                ).to(self.device)

                predictions = self.model([img])[0]
                predictions["boxes"][:, 0] += xmin
                predictions["boxes"][:, 2] += xmin
                predictions["boxes"][:, 1] += ymin
                predictions["boxes"][:, 3] += ymin

                if len(predictions["labels"]) != 0:
                    labels.append(predictions["labels"])
                    boxes.append(predictions["boxes"])
                    scores.append(predictions["scores"])

        labels = torch.cat(labels, dim=0)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        
        keep = batched_nms(boxes,scores,labels,nms_thresh)
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        return {
            "labels": labels,
            "boxes": boxes,
            "scores": scores,
        }

    def eval_dataset(
        self,
        dataset: AleketDataset,
        indices: list[int],
        nms_thresh: float,
        score_thresh: float,
        evaluator: Optional[Evaluator] = None,
    ) -> dict[str, float]:

        if not evaluator:
            evaluator = Evaluator(dataset, indices)
            
        with torch.no_grad():
            all_dts = {}
            for idx in indices:
                img, _ = dataset[idx]
                dts = self.get_detections(img, nms_thresh, score_thresh)
                all_dts[idx] = dts

        return evaluator.eval(all_dts)

    def infer(self,
              image: Image,
              nms_thresh: float,
              score_thresh: float,
              ) -> dict:
        dts = self.get_detections(image, nms_thresh, score_thresh)
        bboxes = dts["boxes"].cpu().numpy()
        labels = dts["labels"].cpu().numpy()
        
        count = {}
        area = {}
        with torch.no_grad():
            
            for class_id, class_name in enumerate(self.classes):  # skips background
           
                bboxes_by_class = bboxes[np.where(labels == class_id)]
                count[class_name] = len(bboxes_by_class)
                area[class_name] = np.sum(
                    (bboxes_by_class[:, 2] - bboxes_by_class[:, 0])
                    * (bboxes_by_class[:, 3] - bboxes_by_class[:, 1])
                    )

        return {
            "bboxes": bboxes.tolist(),
            "labels": labels.tolist(),
            "count": count,
            "area": area,
        }


    def infer_images(self,
                    images_path: list[str],
                    output_dir: str,
                    nms_thresh: float,
                    score_thresh: float,
                    save_bboxes: bool = True,
                    save_annotated_images: bool = False) -> None:

        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        stats_file_path = os.path.join(output_dir, "stats.csv")

        bboxes_dir = os.path.join(output_dir, "bboxes") if save_bboxes else None
        if bboxes_dir:
            os.makedirs(bboxes_dir, exist_ok=True)

        annotated_dir = os.path.join(output_dir, "annotated") if save_annotated_images else None
        if annotated_dir:
            os.makedirs(annotated_dir, exist_ok=True)

        with (open(stats_file_path, "w") as stats_file):
            stats_writer = csv.writer(stats_file, delimiter="")
            headers = ["Image"]
            headers.extend([f"{class_name} area" for class_name in self.classes])
            headers.extend([f"{class_name} count" for class_name in self.classes])
            stats_writer.writerow(headers)

            for img_path in images_path:
                img_name = os.path.basename(img_name)
                img = PIL.Image.open(img_path)
                stats = self.infer(img, nms_thresh, score_thresh)

                area = stats["area"]
                count = stats["count"]
                labels = stats["labels"]
                bboxes = stats["bboxes"]
                
                row = [os.path.basename(img_path)]
                row.extend([area[class_name] for class_name in self.classes])
                row.extend([count[class_name] for class_name in self.classes])
                stats_writer.writerows(row)

                if save_bboxes:
                    bboxes_file_path = os.path.join(bboxes_dir, f"{img_name}.csv")
                    with open(bboxes_file_path, "w") as bboxes_file:
                        bboxes_writer = csv.writer(bboxes_file, delimiter="")
                        headers = ["xmin", "ymin", "xmax", "ymax", "class name"]
                        bboxes_writer.writerow(headers)
                        rows = [bbox + [class_name] for bbox, class_name in zip(bboxes, labels)]
                        bboxes_writer.writerows(rows)
                
                if save_annotated_images:
                    annotated_image_path = os.path.join(annotated_dir, f"{img_name}_annotated.png")
                    visualize_bboxes(img, bboxes, labels, save_path=annotated_image_path)

                        









