# Standard Library
from collections import defaultdict
import os
import json
import shutil
from typing import  Iterable, Optional

# Third-party Libraries
import PIL.Image
import gdown

# PyTorch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors


def _collate_fn(batch):
    """Collates data samples into batches for the dataloader."""
    return tuple(zip(*batch))

# Downloads and extracts the dataset if it doesn't exist locally.
def download_dataset(save_dir: str, patched_dataset_gdrive_id: str = ""):
    """Downloads and extracts the dataset if it doesn't exist locally.

    Args:
        save_dir: The directory to save the dataset.
        patched_dataset_gdrive_id: GoogleDrive id to download the dataset from.

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

    Args:
        dataset_dir: Directory containing the 'imgs' folder and 'dataset.json'.
        img_size: Size to resize the image to.
        augmentation: Optional torchvision transforms to apply to the data.

    Attributes:
        NUM_TO_CLASSES: Mapping of numerical labels to class names.
        CLASSES_TO_NUM: Mapping of class names to numerical labels.
    """
    
    CLASSES_TO_NUM = {"background": 0, "healthy": 1, "not healthy": 2}
    NUM_TO_CLASSES = {0: "background", 1: "healthy", 2: "not healthy"}

    def __init__(
        self,
        dataset_dir: str,
        augmentation: Optional[v2.Transform] = None
    ) -> None:
        self.img_dir = os.path.join(dataset_dir, "imgs")
        
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as annot_file:
            self.dataset = json.load(annot_file)
        
        self.image_to_idx = {img: i for i, img in enumerate(self.dataset.keys())}
        self.idx_to_image = {i: img for i, img in enumerate(self.dataset.keys())}
        
        self.default_transforms = v2.ToDtype(torch.float32, scale=True)
        
        self.augmentation = augmentation
        print(f"Dataset loaded from {dataset_dir}")

    
    def split_by_full_images(self,
              dataset_fraction: float, 
              validation_fraction: float,
              generator: np.random.Generator
              ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Groups image indices by their corresponding full image ID into val and train datasets.
        Returns:
            dict[str, list[int]]: A dictionary mapping full image IDs to lists of 
                                  corresponding image indices in the dataset.
        """
        by_full_images = defaultdict(list)
        for name in self.idx_to_image.values():
            full_image_id = name.split('_')[0]
            by_full_images[full_image_id].append(name)

        full_images = list(by_full_images.keys())
        generator.shuffle(full_images)

        total_num_samples = max(2, int(len(self) * dataset_fraction))
        validation_num_samples = max(1, int(validation_fraction * total_num_samples))
        train_num_samples = total_num_samples - validation_num_samples
            
        train_len = 0
        train_set = {}
        validation_len = 0
        validation_set = {}
        
        for full_image in full_images:
            if validation_len < validation_num_samples:
                validation_set[full_image] = by_full_images[full_image]
                validation_len += len(by_full_images[full_image])
            elif train_len < train_num_samples:
                train_set[full_image] = by_full_images[full_image]
                train_len += len(by_full_images[full_image])
        
        return train_set, validation_set

    def get_indices(self, samples: Optional[Iterable[int | str]] = None) -> list[int]:
        if not samples:
            samples = list(range(len(self)))
            
        return [self.image_to_idx[ind] if isinstance(ind, str) else ind for ind in samples]
        
    def create_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        samples: list[int | str] = None,
    ) -> DataLoader:
        """Creates DataLoaders for split dataset.
        Args:
            dataset: The AleketDataset to divide.
            train_indices: Dataset indicies to train.
            val_indices: Dataset indicies to validate.
            batch_size: The batch size for the DataLoaders.
            num_workers: The number of worker processes for data loading.
        Returns:
            A tuple containing the training DataLoader and the validation DataLoader.
        """
        indices = self.get_indices(samples)
            
        subset = Subset(self, indices)
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=num_workers,
        )
        return dataloader

    def get_annots(self, indices: list[int | str]) -> list[dict]:
        indices = self.get_indices(indices)
        targets = [
            {
                "image_id": idx,
                "labels": self.dataset[self.idx_to_image[idx]]["category_id"],
                "boxes": self.dataset[self.idx_to_image[idx]]["boxes"]
            } for idx in indices
        ]
        return targets
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.idx_to_image)
 
    def __getitem__(self, idx: int | str):
        """
        Retrieves an image and its corresponding target annotations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image as a torchvision.tv_tensors.Image object 
                   and a dictionary of target annotations.
        """
        if isinstance(idx, str): 
            idx = self.image_to_idx[idx]
            
        img_name = self.idx_to_image[idx] 
          
        annots = self.dataset[img_name]
        labels, bboxes = annots["category_id"], annots["boxes"]
                
        img_path = os.path.join(f"{self.img_dir}",f"{img_name}.jpeg")
        img = PIL.Image.open(img_path).convert("RGB")

        img = tv_tensors.Image(img, dtype=torch.uint8)

        ht, wt = img.shape[-2:]  # Get width and height

        labels = torch.as_tensor(labels)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(ht, wt))
        
        if self.augmentation:
            img, bboxes, labels = self.augmentation(img, bboxes, labels)

        img, bboxes, labels = self.default_transforms(img, bboxes, labels)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": idx,
        }

        return img, target

