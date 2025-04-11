from PIL import Image
import numpy as np
from ultralytics import YOLO
from ultralytics.data.loaders import autocast_list
from ultralytics.engine.results import Boxes, Results

import torch
import torchvision.ops as ops

from utils.patches import Patch, make_patches


def load(image: str | Image.Image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    
    return image

        


class Resizer(torch.nn.Module):
    def __init__(self, size_factor: float):
        super().__init__()
        self.size_factor = size_factor

    def forward(self, x: Image.Image) -> Image.Image:
        ht, wd = x.height, x.width
        ht = int(ht * self.size_factor)
        wd = int(wd * self.size_factor)
        return x.resize(size=[wd, ht])


class Patcher(torch.nn.Module):
    def __init__(self, patch_size: int, overlap: float):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

    def forward(self, x: Image.Image) -> tuple[list[Patch], list[Image.Image]]:
        ht, wd = x.height, x.width
        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.overlap
        )
        padded_img = Image.new("RGB", (padded_width, padded_height))  # Create a new image with padding
        padded_img.paste(x)

        patched_images = [patch.crop(padded_img) for patch in patches]

        return patches, patched_images


class Preprocessor(torch.nn.Module):
    def __init__(self, patch_size: int, overlap: float, size_factor: float):
        super().__init__()
        self.resizer = Resizer(size_factor)
        self.patcher = Patcher(patch_size, overlap)

    @torch.no_grad()
    def forward(
        self, x: str | Image.Image | torch.Tensor
    ) -> tuple[list[Patch], torch.Tensor]:
        x = load(x)
        x = self.resizer(x)
        x = self.patcher(x)
        return x


class PatchMerger(torch.nn.Module):
    def __init__(self, size_factor: float):
        super().__init__()
        self.size_factor = size_factor

    def forward(self, x: tuple[list[Patch], list[torch.Tensor]]) -> torch.Tensor:

        patches, results = x
        boxes = []

        # Adjust bounding boxes to original image coordinates
        for pred_boxes, patch in zip(results, patches):

            pred_boxes[:, [0, 2]] += patch.xmin  # Adjust x-coordinates of boxes
            pred_boxes[:, [1, 3]] += patch.ymin  # Adjust y-coordinates of boxes
            if len(pred_boxes) != 0:
                boxes.extend(pred_boxes)
                
        pred_boxes = torch.stack(boxes)
        pred_boxes[:, [0,1,2,3]] /= self.size_factor  # Rescale boxes to original image size
        

        return pred_boxes


class WeightedBoxesFusionProccessor(torch.nn.Module):
    def __init__(
        self, pre_wbf_detections: int, wbf_iou_thresh: float, post_wbf_detections: int
    ):
        super().__init__()

        self.pre_wbf_detections = pre_wbf_detections
        self.iou_thresh = wbf_iou_thresh
        self.post_wbf_detections = post_wbf_detections

    def wbf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merges object detections based on Weighted Boxes Fusion (WBF).

        Args:
            boxes (np.ndarray): A numpy array of shape (N, 4) representing bounding boxes.
            scores (np.ndarray): A numpy array of shape (N,) representing confidence scores.
            iou_threshold (float): The IoU threshold for merging boxes.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
                - merged_scores: A numpy array of shape (M,) representing merged scores.
        """
        # Assumed that detections are sorted
        if self.iou_thresh >= 1:  # no neeed in wbf
            return x

        merged_boxes, merged_scores, cluster_boxes, cluster_scores = [], [], [], []

        # 1. Iterate through predictions
        for current_box, current_score in zip(x[:,[0,1,2,3]], x[:,4]):
            found_cluster = False
            # 2. Find cluster
            for i, merged_box in enumerate(merged_boxes):
                # Calculate IoU between current box and merged box
                iou = ops.box_iou(
                    current_box[torch.newaxis, ...], merged_box[torch.newaxis, ...]
                )[0, 0]
                if iou > self.iou_thresh:  # 3. Cluster Found
                    found_cluster = True

                    cluster_boxes[i].append(current_box)
                    cluster_scores[i].append(current_score)

                    # Get all boxes and scores in the cluster
                    matched_boxes = torch.stack(cluster_boxes[i])
                    matched_scores = torch.stack(cluster_scores[i])

                    # Merge boxes using weighted average based on scores
                    merged_boxes[i] = (
                        matched_boxes * matched_scores[:, torch.newaxis]
                    ).sum(axis=0) / matched_scores.sum()
                    merged_scores[i] = matched_scores.mean()  # Average the scores
                    break  # Move to the next box

            # 4. Cluster not found
            if not found_cluster:
                # If no overlap, add the current box as a new merged box
                merged_boxes.append(current_box)
                merged_scores.append(current_score)

                # Create a new cluster for this box
                cluster_boxes.append([current_box])
                cluster_scores.append([current_score])

        boxes = torch.stack(merged_boxes)
        scores = torch.stack(merged_scores).unsqueeze(1)
        labels = torch.full_like(scores, x[0, 5])

        return torch.cat((boxes, scores, labels), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[: self.pre_wbf_detections]
        classes = torch.unique(x[:, 5])

        merged = []

        for class_id in classes:
            keep = torch.where(x[:, 5] == class_id)
            wbf_boxes = self.wbf(x[keep])
            merged.extend(wbf_boxes)
            
        merged = torch.stack(merged)
        indices = torch.argsort(merged[:, 4], descending=True)
        merged = merged[indices]

        return merged[: self.post_wbf_detections]


class Postprocessor(torch.nn.Module):
    def __init__(
        self,
        size_factor: float,
        pre_wbf_detections: int,
        wbf_iou_thresh: float,
        max_detections: int,
    ):
        super().__init__()
        self.merger = PatchMerger(size_factor)
        self.wbf = WeightedBoxesFusionProccessor(
            pre_wbf_detections, wbf_iou_thresh, max_detections
        )

    @torch.no_grad()
    def forward(self, x: tuple[list[Patch], list[torch.Tensor]]) -> torch.Tensor:
        boxes = self.merger(x)
        boxes = self.wbf(boxes)
        return boxes


class Detection(torch.nn.Module):

    def __init__(
        self,
        model_path: str,
        overlap: float = 0.2,
        patch_size: int = 1024,
        size_factor: float = 1.0,
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.7,
        max_patch_detections: int = 300,
        patch_per_batch: int = 4,
        pre_wbf_detections: int = 3000,
        wbf_iou_thresh: float = 0.5,
        max_detections: int = 1000,
    ) -> None:
        super().__init__()

        self.conf = conf_thresh
        self.iou = nms_iou_thresh
        self.max_det = max_patch_detections
        self.batch = patch_per_batch
        self.imgsz = patch_size

        self.preprocessor = Preprocessor(patch_size, overlap, size_factor)
        self.yolo = YOLO(model_path, task="detect")
        self.postproccessor = Postprocessor(
            size_factor, pre_wbf_detections, wbf_iou_thresh, max_detections
        )

    def forward(self, x):
        image = load(x)
        patches, images = self.preprocessor.forward(x)
        preds_result = self.yolo.predict(
            source=images,
        )

        preds_boxes = [res.boxes.data.clone() for res in preds_result]
        
        for res, img in zip(preds_result,images):
            img_np = np.array(img)
            img_np = img_np[..., ::-1]
            res.plot(show=True, img=img_np)
            
        preds_boxes = self.postproccessor.forward((patches, preds_boxes))

        return Results(
            np.asarray(image)[..., ::-1],
            "somePath",
            (
                self.yolo.model.module.names
                if hasattr(self.yolo.model, "module")
                else self.yolo.model.names
            ),
            boxes=preds_boxes,
        )
