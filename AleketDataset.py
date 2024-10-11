# Standard Library
import os
import json
import shutil
from typing import  Optional

# Third-party Libraries
import PIL.Image
import gdown

# PyTorch
import torch
from torch.utils.data import Dataset

# Torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors

from utils import COCO_STATS_NAMES, LOSSES_NAMES

# Downloads and extracts the dataset if it doesn't exist locally.
def download_dataset(save_dir: str, patched_dataset_gdrive_id: str = ""):
    """Downloads and extracts the dataset if it doesn't exist locally.

    Args:
        save_dir: The directory to save the dataset.

    Returns:
        The path to the saved dataset directory.
    """
    if not os.path.exists(save_dir):
        gdown.download(id=patched_dataset_gdrive_id, output="_temp_.zip")
        shutil.unpack_archive("_temp_.zip", save_dir)
        os.remove("_temp_.zip")
    return save_dir

# Aleket Dataset
class AleketDataset(Dataset):
    """Custom dataset for Aleket images and annotations.

    Attributes:
        NUM_TO_CLASSES: Mapping of numerical labels to class names.
        CLASSES_TO_NUM: Mapping of class names to numerical labels.
    """

    CLASSES_TO_NUM = {"background": 0, "healthy": 1, "not healthy": 2}
    NUM_TO_CLASSES = {0: "background", 1: "healthy", 2: "not healthy"}

    def __init__(
        self,
        dataset_dir: str,
        transforms: Optional[v2.Transform] = None
    ) -> None:
        """Initializes the AleketDataset.

        Args:
            dataset_dir: Directory containing the 'imgs' folder and 'dataset.json'.
            transforms: Optional torchvision transforms to apply to the data.
        """
        self.img_dir = os.path.join(dataset_dir, "imgs")
        
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as annot_file:
            self.dataset = json.load(annot_file)
        
        self.transforms = transforms
        print(f"Dataset loaded from {dataset_dir}")

    def __len__(self):
        return len(self.dataset["imgs"])
 
    def __getitem__(self, idx: int):
        img_path = f"{self.img_dir}/{self.dataset['imgs'][idx]}.jpeg"
        img = PIL.Image.open(img_path).convert("RGB")

        annots = self.dataset["annotations"][idx]
        labels, bboxes = annots["category_id"], annots["boxes"]

        # Convert to torchvision tensors
        img = tv_tensors.Image(img, dtype=torch.uint8)

        wt, ht = img.shape[-1], img.shape[-2]  # Get width and height

        labels = torch.as_tensor(labels)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(wt, ht))
        
        if self.transforms:
            img, bboxes, labels = self.transforms(img, bboxes, labels)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": idx,
        }

        return img, target