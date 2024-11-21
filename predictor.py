# Standard Library
import math
from typing import Optional

# Third-party Libraries
import PIL
import numpy as np

# PyTorch
import torch
import torchvision.transforms.v2.functional as F
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection import FasterRCNN
import torchvision.ops as ops

from aleket_dataset import AleketDataset, collate_fn
from box_utils import box_area, box_iou
from metrics import Evaluator


def make_patches(
    width: int,
    height: int,
    patch_size: int,
    overlap: float,
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    overlap_size = int(patch_size * overlap)
    no_overlap_size = patch_size - overlap_size

    imgs_per_width = math.ceil(float(width) / no_overlap_size)
    imgs_per_height = math.ceil(float(height) / no_overlap_size)

    padded_height = imgs_per_width * no_overlap_size + overlap_size
    padded_width = imgs_per_width * no_overlap_size + overlap_size

    patch_boxes = []

    for row in range(imgs_per_height):
        for col in range(imgs_per_width):
            xmin, ymin = col * no_overlap_size, row * no_overlap_size
            xmax, ymax = xmin + patch_size, ymin + patch_size
            patch_box = (xmin, ymin, xmax, ymax)

            patch_boxes.append(patch_box)

    return padded_width, padded_height, patch_boxes


def merge_detections(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merges object detections

    Args:
      boxes: A numpy array of shape (N, 4) representing bounding boxes.
      scores: A numpy array of shape (N,) representing confidence scores.
      classes: A numpy array of shape (N,) representing class IDs.
      iou_threshold: The IoU threshold for merging boxes.

    Returns:
      A tuple containing:
        - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
        - merged_scores: A numpy array of shape (M,) representing merged scores.
        - merged_classes: A numpy array of shape (M,) representing merged classes.
    """

    # 1. Sort Detections
    indices = np.argsort(-scores)
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]

    merged_boxes = []
    merged_scores = []
    merged_labels = []

    while len(boxes) > 0:
        current_box = boxes[0]
        current_label = labels[0]

        ious = box_iou(current_box[np.newaxis, ...], boxes)
        matching_indices = np.where(ious[0] > iou_threshold)[0]

        matched_boxes = boxes[matching_indices]
        matched_scores = scores[matching_indices]

        # Merge boxes
        merged_box = np.array(
            [
                matched_boxes[:, 0].min(),
                matched_boxes[:, 1].min(),
                matched_boxes[:, 2].max(),
                matched_boxes[:, 3].max(),
            ]
        )

        # Weighted average of scores
        areas = box_area(matched_boxes)
        merged_score = (matched_scores * areas).sum() / areas.sum()

        merged_boxes.append(merged_box)
        merged_scores.append(merged_score)
        merged_labels.append(current_label)

        # Remove merged boxes
        boxes = np.delete(boxes, matching_indices, axis=0)
        scores = np.delete(scores, matching_indices, axis=0)
        labels = np.delete(labels, matching_indices, axis=0)

    return np.array(merged_boxes), np.array(merged_scores), np.array(merged_labels)


class Patcher(Dataset):
    def __init__(
        self,
        images: list[str | Image | torch.Tensor] | Dataset,
        size_factor: float,
        patch_size: int,
        patch_overlap: float,
    ):
        self.images = images
        self.size_factor = size_factor
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    @torch.no_grad()
    def preprocess(
        self, image: Image | torch.Tensor | str, idx: int
    ) -> tuple[list[tuple[int, int, int, int]], Tensor, int]:
        if isinstance(image, str):
            image = PIL.Image.open(image)

        if isinstance(image, Image):
            image = F.to_image(image)

        ht, wd = image.shape[-2:]
        ht = int(ht * self.size_factor)
        wd = int(wd * self.size_factor)

        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.patch_overlap
        )

        image = F.resize(image, size=[ht, wd])
        image = F.pad(
            image, padding=[0, 0, padded_width - wd, padded_height - ht], fill=0.0
        )
        image = F.to_dtype(image, dtype=torch.float32, scale=True)

        patched_images = torch.stack(
            [F.crop(image, y1, x1, y2 - y1, x2 - x1) for (x1, y1, x2, y2) in patches]
        )

        return patches, patched_images, idx

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)

    @torch.no_grad()
    def postprocess(
        self,
        patches: list[list[int]],
        predictions: list[dict[str, torch.Tensor]],
        post_postproces_detections: int,
        iou_thresh: float,
        use_merge: bool = True,
    ) -> dict[str, np.ndarray]:

        boxes = []
        labels = []
        scores = []

        for prediction, patch in zip(predictions, patches):
            x1, y1, _, _ = patch
            prediction["boxes"][:, [0, 2]] += x1
            prediction["boxes"][:, [1, 3]] += y1
            if len(prediction["boxes"]) != 0:
                boxes.extend(prediction["boxes"].numpy(force=True))
                labels.extend(prediction["labels"].numpy(force=True))
                scores.extend(prediction["scores"].numpy(force=True))

        if boxes:
            labels = np.stack(labels)
            boxes = np.stack(boxes)
            scores = np.stack(scores)

            boxes /= self.size_factor
            if use_merge:
                boxes, scores, labels = merge_detections(
                    boxes, scores, labels, iou_thresh
                )
            else:
                keep = ops.batched_nms(
                    torch.as_tensor(boxes),
                    torch.as_tensor(scores),
                    torch.as_tensor(labels),
                    iou_thresh,
                ).numpy(force=True)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            boxes = boxes[:post_postproces_detections]
            scores = scores[:post_postproces_detections]
            labels = labels[:post_postproces_detections]

        else:
            labels = np.zeros(0, dtype=np.int64)
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    @torch.no_grad()
    def __getitem__(
        self, idx: int
    ) -> tuple[list[tuple[int, int, int, int]], Tensor, int]:
        if isinstance(self.images, Dataset):
            image, target = self.images[idx]
            return self.preprocess(image, target["image_id"])
        r = self.preprocess(self.images[idx], idx)

        return r


class Predictor:
    def __init__(
        self,
        model: FasterRCNN,
        device: torch.device,
        images_per_batch: int = 4,
        image_size_factor: float = 1,
        detections_per_image: int = 300,
        detections_per_patch: int = 100,
        patches_per_batch: int = 4,
        patch_size: int = 1024,
        patch_overlap: float = 0.2,
    ):

        self.device = device
        self.classes = list(AleketDataset.NUM_TO_CLASSES.values())[
            1:
        ]  # remove background class
        self.model = model.to(device)
        self.images_per_batch = images_per_batch
        self.image_size_factor = image_size_factor
        self.per_image_detections = detections_per_image

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch
        self.patches_per_batch = patches_per_batch

    @torch.no_grad()
    def get_predictions(
        self,
        images: list[Image | torch.Tensor | str] | Dataset,
        iou_thresh: float,
        score_thresh: float,
        use_merge: bool = True,
    ) -> dict[int, dict[str, np.array]]:

        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = 1 if use_merge else iou_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch
        self.model.eval()

        patcher = Patcher(
            images, self.image_size_factor, self.patch_size, self.patch_overlap
        )
        dataloader = DataLoader(
            patcher,
            batch_size=self.images_per_batch,
            num_workers=self.images_per_batch,
            collate_fn=collate_fn,
        )

        result = {}

        for batched_patches, batched_images, batched_idxs in dataloader:
            for patches, imgs, idx in zip(
                batched_patches, batched_images, batched_idxs
            ):
                predictions = []
                for b_imgs in torch.split(imgs, self.patches_per_batch):
                    b_imgs = b_imgs.to(self.device)
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        predictions.extend(self.model(b_imgs))
                predictions = patcher.postprocess(
                    patches,
                    predictions,
                    self.per_image_detections,
                    iou_thresh,
                    use_merge=use_merge,
                )
                result[idx] = predictions

        del patcher
        del dataloader
        return result

    @torch.no_grad()
    def eval_dataset(
        self,
        dataset: AleketDataset,
        indices: list[int | str],
        iou_thresh: float,
        score_thresh: float,
        evaluator: Optional[Evaluator] = None,
        use_merge: bool = True,
    ) -> dict[str, float]:
        if not evaluator:
            evaluator = Evaluator(dataset.get_annots(indices))
        subset = Subset(dataset, indices)
        predictions = self.get_predictions(subset, iou_thresh, score_thresh, use_merge)
        return evaluator.eval(predictions)

