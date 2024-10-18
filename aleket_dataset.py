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
        img_size: int,
        augmentation: Optional[v2.Transform] = None
    ) -> None:
        self.img_dir = os.path.join(dataset_dir, "imgs")
        
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as annot_file:
            self.dataset = json.load(annot_file)
        
        self.images = list(self.dataset.keys())
        
        def get_key(x: str): # sort by full image id and then patch num
            s = x.split('_')
            return int(s[0]), int(s[1])
        
        self.images.sort(key=get_key)


        self.default_transforms = v2.Compose(
            [v2.Resize(size=img_size), v2.ToDtype(torch.float32, scale=True), ]
        )
        self.augmentation = augmentation
        print(f"Dataset loaded from {dataset_dir}")

    def by_full_images(self) -> dict[str, list[int]]:
        """Groups image indices by their corresponding full image ID.
        Returns:
            dict[str, list[int]]: A dictionary mapping full image IDs to lists of 
                                  corresponding image indices in the dataset.
        """
        by_full_images = {}
        for idx, name in enumerate(self.images):
            full_image_id = name.split('_')[0]
            if not full_image_id in by_full_images:
                by_full_images[full_image_id] = []
            by_full_images[full_image_id].append(idx)
        return by_full_images

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)
 
    def __getitem__(self, idx: int):
        """
        Retrieves an image and its corresponding target annotations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image as a torchvision.tv_tensors.Image object 
                   and a dictionary of target annotations.
        """
        image_id = self.images[idx]  
        
        annots = self.dataset[image_id]
        labels, bboxes = annots["category_id"], annots["boxes"]
                
        img_path = os.path.join(f"{self.img_dir}",f"{image_id}.jpeg")
        img = PIL.Image.open(img_path).convert("RGB")

        img = tv_tensors.Image(img, dtype=torch.uint8)

        wt, ht = img.shape[-1], img.shape[-2]  # Get width and height

        labels = torch.as_tensor(labels)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(wt, ht))
        
        if self.augmentation:
            img, bboxes, labels = self.augmentation(img, bboxes, labels)

        img, bboxes, labels = self.default_transforms(img, bboxes, labels)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": idx,
        }

        return img, target