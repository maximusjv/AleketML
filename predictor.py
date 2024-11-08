# Standard Library
import csv
import os
import time
from typing import Optional, Dict, Any

# Third-party Libraries
import PIL
import numpy as np
from PIL.Image import Image

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

# Torchvision
import torchvision.tv_tensors as tv_tensors
from torchvision.models.detection import FasterRCNN
import torchvision.transforms.v2 as v2
from torchvision.ops import batched_nms

from aleket_dataset import AleketDataset, collate_fn
from dataset_statisics import visualize_bboxes
from metrics import LOSSES_NAMES, VALIDATION_METRICS, Evaluator
from utils import make_patches

class Pacther(Dataset): 
    def __init__(
        self,
        images: list[str | Image | torch.Tensor] | Dataset,
        patch_size: int,
        patch_overlap: float,
        ):
        self.images = images
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
    
    def preprocess(self, image: Image | torch.Tensor | str, idx: int) -> tuple[list[list[int]], torch.Tensor]:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        
        if isinstance(image, Image):
            image = v2.functional.to_image(image)
    
        ht, wd = image.shape[-2:]
        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.patch_overlap
        )
        image = v2.functional.pad(image, padding=[0, 0, padded_width-wd, padded_height-ht],fill=0.0)
        image = v2.functional.to_dtype(image,  dtype=torch.float32, scale=True)
        
        patched_images = torch.stack([v2.functional.crop(image, y1, x1, y2-y1, x2-x1) 
                               for (x1, y1, x2, y2) in patches])
        
        return patches, patched_images, idx
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)
    
    def postprocess(self,
                    patches: list[list[int]],
                    predictions: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        
        boxes = []
        labels = []
        scores = []
    
        for prediction, patch in zip(predictions, patches):
            x1, y1, _, _ = patch
            prediction["boxes"][:, [0, 2]] += x1
            prediction["boxes"][:, [1, 3]] += y1
            if len(prediction["boxes"]) != 0:
                boxes.append(prediction["boxes"])
                labels.append(prediction["labels"])
                scores.append(prediction["scores"])
           
        labels = torch.cat(labels, dim=0)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        
    def __getitem__(self, idx: int) -> tuple[list[list[int]], torch.Tensor]:
        if isinstance(self.images, Dataset):
            image, target = self.images[idx]
            return self.preprocess(image, target["image_id"])
        return self.preprocess(self.images[idx], idx)
        

class Predictor:
    def __init__(
        self,
        model: FasterRCNN,
        device: torch.device,
        detections_per_patch: int = 100,
        patch_size: int = 1024,
        patch_overlap: float = 0.2,
        images_per_batch: int = 4,
        patches_per_batch: int = 4,
    ):

        self.device = device
        self.classes = list(AleketDataset.NUM_TO_CLASSES.values())
        self.classes = self.classes[1:] # remove background class
        self.model = model.to(device)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch
        self.patches_per_batch = patches_per_batch
        self.images_per_batch = images_per_batch
        
        
    @torch.no_grad()
    def get_predictions(
        self,
        images: list[Image | torch.Tensor | str] | Dataset,
        nms_thresh: float,
        score_thresh: float,
    ) -> dict[int, dict[str, Tensor]]:

        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = nms_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch
        self.model.eval()
     
        patcher = Pacther(images, self.patch_size, self.patch_overlap)
        dataloader = DataLoader(patcher,
                                batch_size=self.images_per_batch,
                                num_workers=self.images_per_batch,
                                collate_fn=collate_fn,
                                )
        
        result = {}
        
        for (batched_patches, batched_images, batched_idxs) in dataloader:
            for patches, imgs, idx in zip(batched_patches, batched_images, batched_idxs):
                time_it = time.time()
                predictions = []
                for b_imgs in torch.split(imgs, self.patches_per_batch):
                    b_imgs = b_imgs.to(self.device)
                    predictions.extend(self.model(b_imgs))
                predictions = patcher.postprocess(patches, predictions)
                result[idx] = predictions
                print(time.time() - time_it)
    
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
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

        subset = Subset(dataset, indices)
        predictions = self.get_predictions(subset, nms_thresh, score_thresh)
      

        return evaluator.eval(predictions)

    @torch.no_grad()
    def infer(self,
              image: Image,
              nms_thresh: float,
              score_thresh: float,
              ) -> dict:
        dts = self.get_predictions([image], nms_thresh, score_thresh)[0]
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

    @torch.no_grad()
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

                        









