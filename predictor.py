# Standard Library
import csv
import math
import os
import time
from typing import Optional

# Third-party Libraries
import PIL
import numpy as np
from PIL.Image import Image

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

# Torchvision
from torchvision.models.detection import FasterRCNN
import torchvision.transforms.v2.functional as F
from torchvision.ops import batched_nms
from tqdm import tqdm

from aleket_dataset import AleketDataset, collate_fn
from dataset_statisics import visualize_bboxes
from metrics import Evaluator

def make_patches(
          width: int,
          height: int,
          patch_size: int,
          overlap: float,
):
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


class Pacther(Dataset):
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
    def preprocess(self, image: Image | torch.Tensor | str, idx: int) -> tuple[list[list[int]], torch.Tensor]:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        
        if isinstance(image, Image):
            image = F.to_image(image)
    
        ht, wd =  image.shape[-2:]
        ht = int(ht * self.size_factor)
        wd = int(wd * self.size_factor)
        
        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.patch_overlap
        )
        
        image = F.resize(image, size=(ht, wd))
        image = F.pad(image, padding=[0, 0, padded_width-wd, padded_height-ht],fill=0.0)
        image = F.to_dtype(image,  dtype=torch.float32, scale=True)
        
        patched_images = torch.stack([F.crop(image, y1, x1, y2-y1, x2-x1) 
                               for (x1, y1, x2, y2) in patches])
        
        return patches, patched_images, idx
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)
    @torch.no_grad()
    def postprocess(self,
                    patches: list[list[int]],
                    predictions: list[dict[str, torch.Tensor]],
                    nms_thresh: float,
                    max_detections: int,
    ) -> dict[str, torch.Tensor]:
        
        boxes = []
        labels = []
        scores = []
    
        for prediction, patch in zip(predictions, patches):
            x1, y1, _, _ = patch
            prediction["boxes"][:, [0, 2]] += x1
            prediction["boxes"][:, [1, 3]] += y1
            if len(prediction["boxes"]) != 0:
                boxes.extend(prediction["boxes"])
                labels.extend(prediction["labels"])
                scores.extend(prediction["scores"])
        
        if boxes:
            labels = torch.stack(labels)
            boxes = torch.stack(boxes)
            scores = torch.stack(scores)
            
            boxes /= self.size_factor
            keep = batched_nms(boxes, scores, labels, nms_thresh)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            boxes = boxes[:max_detections]
            scores = scores[:max_detections]
            labels = labels[:max_detections]

        else:
            labels = torch.zeros(0, dtype=torch.int64)
            boxes = torch.zeros((0,4), dtype=torch.float32)
            scores = torch.zeros(0, dtype=torch.float32)
        
       
        return {
            "boxes": boxes.cpu(),
            "scores": scores.cpu(),
            "labels": labels.cpu(),
        }
    @torch.no_grad()   
    def __getitem__(self, idx: int) -> tuple[list[list[int]], torch.Tensor]:
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
        image_size_factor: float = 0.8,
        detections_per_image: int = 300,
        
        detections_per_patch: int = 100,
        patches_per_batch: int = 4,
        patch_size: int = 1024,
        patch_overlap: float = 0.2,
    ):

        self.device = device
        self.classes = list(AleketDataset.NUM_TO_CLASSES.values())[1:] # remove background class
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
        nms_thresh: float,
        score_thresh: float,
    ) -> dict[int, dict[str, Tensor]]:

        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = nms_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch
        self.model.eval()
     
        patcher = Pacther(images, self.image_size_factor, self.patch_size, self.patch_overlap)
        dataloader = DataLoader(patcher,  
                                batch_size=self.images_per_batch,
                                num_workers=self.images_per_batch,
                                collate_fn=collate_fn,
                                )
        
        result = {}
        
        for (batched_patches, batched_images, batched_idxs) in dataloader:
            for patches, imgs, idx in zip(batched_patches, batched_images, batched_idxs):
                predictions = []
                for b_imgs in torch.split(imgs, self.patches_per_batch):
                    b_imgs = b_imgs.to(self.device)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        predictions.extend(self.model(b_imgs))
                predictions = patcher.postprocess(patches, predictions, nms_thresh, self.per_image_detections)
                result[idx] = predictions
        
    
        del patcher
        del dataloader
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
    def analyze(self,
              prediction: dict[str, Tensor],
              ) -> dict:
       
        bboxes = prediction["boxes"].numpy()
        labels = prediction["labels"].numpy()
        labels_names = [self.classes[i-1] for i in labels]
        count = {}
        area = {}
        for class_id, class_name in enumerate(self.classes):  # skips background
            bboxes_by_class = bboxes[np.where(labels == class_id+1)]
            count[class_name] = len(bboxes_by_class)
            area[class_name] = np.sum(
                (bboxes_by_class[:, 2] - bboxes_by_class[:, 0]) / 2.0
                * (bboxes_by_class[:, 3] - bboxes_by_class[:, 1]) / 2.0
                ) * math.pi if len(bboxes_by_class) > 0 else 0

       
        return {
            "bboxes": bboxes.tolist(),
            "labels": labels_names,
            "count": count,
            "area": area,
        }

    @torch.no_grad()
    def infer(self,
              images: list[str | Image | torch.Tensor],
              output_dir: str,
              nms_thresh: float,
              score_thresh: float,
              save_bboxes: bool = False,
              num_of_annotated_images_to_save: int = 0) -> None:


        if num_of_annotated_images_to_save == -1:
            num_of_annotated_images_to_save = len(images)
        
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        stats_file_path = os.path.join(output_dir, "stats.csv")

        bboxes_dir = os.path.join(output_dir, "bboxes") if save_bboxes else None
        if bboxes_dir:
            os.makedirs(bboxes_dir, exist_ok=True)

        annotated_dir = os.path.join(output_dir, "annotated") if num_of_annotated_images_to_save > 0 else None
        if annotated_dir:
            os.makedirs(annotated_dir, exist_ok=True)

        with open(stats_file_path, "w", newline='') as stats_file:
            stats_writer = csv.writer(stats_file, delimiter=",")
            headers = ["Image"]
            headers.extend([f"{class_name} area" for class_name in self.classes])
            headers.extend([f"{class_name} count" for class_name in self.classes])
            stats_writer.writerow(headers)
            
            predictions = self.get_predictions(images, nms_thresh, score_thresh)
            for idx, pred in predictions.items():
                image = images[idx]
                image_name = os.path.basename(image) if isinstance(image, str) else str(idx)
                stats = self.analyze(pred)

                area = stats["area"]
                count = stats["count"]
                labels = stats["labels"]
                bboxes = stats["bboxes"]
                
                row = [image_name]
                row.extend([int(area[class_name]) for class_name in self.classes])
                row.extend([int(count[class_name]) for class_name in self.classes])
                stats_writer.writerow(row)

                if save_bboxes:
                    bboxes_file_path = os.path.join(bboxes_dir, f"{image_name}.csv")
                    with open(bboxes_file_path, "w", newline='') as bboxes_file:
                        bboxes_writer = csv.writer(bboxes_file, delimiter=",")
                        headers = ["xmin", "ymin", "xmax", "ymax", "class name"]
                        bboxes_writer.writerow(headers)
                        for (x1, y1, x2, y2), class_name in zip(bboxes, labels):
                            bboxes_writer.writerow([int(x1), int(y1), int(x2), int(y2), class_name])

                
                if num_of_annotated_images_to_save > 0:
                    num_of_annotated_images_to_save -= 1
                    annotated_image_path = os.path.join(annotated_dir, f"{image_name}_annotated.jpeg")
                    if isinstance(image, str):
                        image = PIL.Image.open(image)
                    if isinstance(image, torch.Tensor):
                        image = F.to_pil_image(image)

                    visualize_bboxes(image, bboxes, labels, save_path=annotated_image_path)
