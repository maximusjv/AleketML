from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils.patches import Patch, make_patches, crop_patches


# Resizer class (unchanged, as it’s efficient with PIL)
class Resizer:
    def __init__(self, size_factor: float):
        self.size_factor = size_factor

    def forward(self, x: Image.Image) -> Image.Image:
        ht, wd = x.height, x.width
        ht = int(ht * self.size_factor)
        wd = int(wd * self.size_factor)
        return x.resize(size=[wd, ht])


# Optimized Patcher: Returns a batched tensor instead of a list of PIL images
class Patcher:
    def __init__(self, patch_size: int, overlap: float):
        self.patch_size = patch_size
        self.overlap = overlap

    def forward(self, x: Image.Image) -> tuple[list[Patch], torch.Tensor]:
        ht, wd = x.height, x.width
        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.overlap
        )
        padded_img = Image.new("RGB", (padded_width, padded_height))
        padded_img.paste(x)
        padded_np = np.array(padded_img)  # Shape: (padded_height, padded_width, 3)

        # Extract patches efficiently using list comprehension
        patch_arrays = crop_patches(padded_np, patches)
        batch_np = np.stack(patch_arrays)  # Shape: (N, patch_size, patch_size, 3)
        batch_tensor = (
            torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
        )  # Shape: (N, 3, patch_size, patch_size)

        return patches, batch_tensor


# Preprocessor: Updated to handle batched tensor output
class Preprocessor:
    def __init__(self, patch_size: int, overlap: float, size_factor: float):
        self.resizer = Resizer(size_factor)
        self.patcher = Patcher(patch_size, overlap)

    @torch.no_grad()
    def forward(self, x: Image.Image) -> tuple[list[Patch], torch.Tensor]:
        x = self.resizer.forward(x)
        patches, batch_tensor = self.patcher.forward(x)
        return patches, batch_tensor


# PatchMerger (unchanged, as it’s reasonably efficient for now)
class PatchMerger:
    def __init__(self, size_factor: float):
        self.size_factor = size_factor

    def forward(self, x: tuple[list[Patch], list[torch.Tensor]]) -> torch.Tensor:
        patches, results = x
        boxes = []
        for pred_boxes, patch in zip(results, patches):
            pred_boxes[:, [0, 2]] += patch.xmin
            pred_boxes[:, [1, 3]] += patch.ymin
            if len(pred_boxes) != 0:
                boxes.extend(pred_boxes)
        pred_boxes = torch.stack(boxes)
        pred_boxes[:, [0, 1, 2, 3]] /= self.size_factor
        return pred_boxes


def _box_inter(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def _box_inter_over_small(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ops.box_area(boxes1)
    area2 = ops.box_area(boxes2)

    inter = _box_inter(boxes1, boxes2)  # Corrected line: Pass box coordinates
    small = torch.minimum(area1[:, None], area2[None, :]) # Ensure correct broadcasting
    return inter / small


class WeightedBoxesFusionProccessor:
    def __init__(
        self, pre_wbf_detections: int, wbf_ios_thresh: float, post_wbf_detections: int
    ):
        self.pre_wbf_detections = pre_wbf_detections
        self.ios_thresh = wbf_ios_thresh
        self.post_wbf_detections = post_wbf_detections

    def wbf(self, x: torch.Tensor) -> torch.Tensor:
        if self.ios_thresh >= 1:  # no need for WBF
            return x
        
        device = x.device
        sum_boxes = torch.empty((0, 4), device=device)
        sum_scores = torch.empty(0, device=device)
        counts = torch.empty(0, dtype=torch.long, device=device)

        for box, score in zip(x[:, :4], x[:, 4]):
            if sum_boxes.size(0) > 0:
                merged_boxes = sum_boxes / sum_scores.view(-1, 1)
                ious = _box_inter_over_small(box.unsqueeze(0), merged_boxes).squeeze(0)
                max_iou, idx = torch.max(ious, dim=0)
                if max_iou > self.ios_thresh:
                    sum_boxes[idx] += box * score
                    sum_scores[idx] += score
                    counts[idx] += 1
                else:
                    sum_boxes = torch.cat([sum_boxes, (box * score).unsqueeze(0)], dim=0)
                    sum_scores = torch.cat([sum_scores, score.unsqueeze(0)], dim=0)
                    counts = torch.cat([counts, torch.tensor([1], device=device)])
            else:
                sum_boxes = (box * score).unsqueeze(0)
                sum_scores = score.unsqueeze(0)
                counts = torch.tensor([1], device=device)

        if sum_boxes.size(0) == 0:
            return torch.empty((0, 6), device=device)

        merged_boxes = sum_boxes / sum_scores.view(-1, 1)
        merged_scores = sum_scores / counts.float()
        labels = torch.full_like(merged_scores, x[0, 5], device=device).unsqueeze(1)
        return torch.cat([merged_boxes, merged_scores.unsqueeze(1), labels], dim=1)

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
        return merged[indices][: self.post_wbf_detections]


# Postprocessor (unchanged, integrates optimized components)
class Postprocessor:
    def __init__(
        self,
        size_factor: float,
        pre_wbf_detections: int,
        wbf_ios_thresh: float, #intresection over small
        max_detections: int,
        single_cls: bool,
    ):
        self.single_cls = single_cls
        self.merger = PatchMerger(size_factor)
        self.wbf = WeightedBoxesFusionProccessor(
            pre_wbf_detections, wbf_ios_thresh, max_detections
        )

    @torch.no_grad()
    def forward(self, x: tuple[list[Patch], list[torch.Tensor]]) -> torch.Tensor:
        boxes = self.merger.forward(x)
        if self.single_cls:
            boxes[:, 5] = 0
        return self.wbf.forward(boxes)


# Optimized Detection class: Passes parameters to YOLO predict
class Detector:
    def __init__(
        self,
        device: int | str | list,
        model_path: str,
        overlap: float = 0.2,
        patch_size: int = 1024,
        size_factor: float = 1.0,
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.7,
        max_patch_detections: int = 300,
        patch_per_batch: int = 4,
        pre_wbf_detections: int = 3000,
        wbf_ios_thresh: float = 0.5,
        max_detections: int = 1000,
        single_cls: bool = True,
    ):
        self.device = device
        self.conf = conf_thresh
        self.iou = nms_iou_thresh
        self.max_det = max_patch_detections
        self.batch = patch_per_batch
        self.imgsz = patch_size
        self.preprocessor = Preprocessor(patch_size, overlap, size_factor)
        self.yolo = YOLO(model_path, task="detect")
        self.postproccessor = Postprocessor(
            size_factor, pre_wbf_detections, wbf_ios_thresh, max_detections, single_cls
        )

    def forward(self, image: Image.Image):
        patches, batch_tensor = self.preprocessor.forward(image)
        preds_result = self.yolo.predict(
            source=batch_tensor,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            device=self.device
        )
        preds_boxes = [res.boxes.data.clone() for res in preds_result]
        preds_boxes = self.postproccessor.forward((patches, preds_boxes))

        return Results(
            np.asarray(image)[..., ::-1],
            "",
            (
                self.yolo.model.module.names
                if hasattr(self.yolo.model, "module")
                else self.yolo.model.names
            ) if not self.postproccessor.single_cls else {0: "object"},
            boxes=preds_boxes,
        )
