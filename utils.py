# Standard Library
import csv
import os
import json
import time
import shutil
from typing import Any, Optional

# Third-party Libraries
import PIL.Image
import gdown

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# Torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors

from training_and_evaluation import COCO_STATS_NAMES, LOSSES_NAMES

# Downloads and extracts the dataset if it doesn't exist locally.
def download_dataset(save_dir: str, patched_dataset_gdrive_id: str = ""):
    """Downloads and extracts the dataset if it doesn't exist locally.

    Args:
        save_dir: The directory to save the dataset.

    Returns:
        The path to the saved dataset directory.
    """
    patched_dataset_gdrive_id = ""  #  FIXME: Replace with actual Google Drive ID fpr
    if not os.path.exists(save_dir):
        gdown.download(id=patched_dataset_gdrive_id, output="_temp_.zip")
        shutil.unpack_archive("_temp_.zip", save_dir)
        os.remove("_temp_.zip")
    print(f"Dataset loaded from {save_dir}")
    return save_dir

# Aleket Dataset
class AleketDataset(Dataset):
    """Custom dataset for Aleket images and annotations.

    Attributes:
        NUM_TO_CLASSES: Mapping of numerical labels to class names.
        CLASSES_TO_NUM: Mapping of class names to numerical labels.
    """

    CLASSES_TO_NUM = {"placeholder": 0, "healthy": 1, "not healthy": 2}
    NUM_TO_CLASSES = {0: "placeholder", 1: "healthy", 2: "not healthy"}

    def __init__(
        self,
        dataset_dir: str,
        transforms: Optional[nn.Module] = None,
        img_size: int = 1024,
    ) -> None:
        """Initializes the AleketDataset.

        Args:
            dataset_dir: Directory containing the 'imgs' folder and 'dataset.json'.
            transforms: Optional torchvision transforms to apply to the data.
        """
        self.img_dir = os.path.join(dataset_dir, "imgs")
        
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as annot_file:
            self.dataset = json.load(annot_file)
        
        
        self.default_transforms = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True), v2.Resize(img_size)]
        )
        self.train = True
        self.transforms = transforms

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

        # Apply transforms if in training mode
        if self.train and self.transforms is not None:
            img, bboxes, labels = self.transforms(img, bboxes, labels)

        # Calculate area and iscrowd
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        # Apply default transforms
        img, bboxes, labels = self.default_transforms(img, bboxes, labels)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "area": area,
            "image_id": idx,
            "iscrowd": iscrowd,
        }

        return img, target
    
# Dataset split
def split_dataset(
    dataset: AleketDataset,
    train_indicies: list[int],
    val_indicies: list[int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Divides the dataset into training and validation sets and creates DataLoaders.
    Args:
        dataset: The AleketDataset to divide.
        train_indicies: Dataset indicies to train
        test_fraction: The fraction of the used dataset to allocate for validation.
        batch_size: The batch size for the DataLoaders.
        num_workers: The number of worker processes for data loading.
    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """

    def collate_fn(batch):
        """Collates data samples into batches for the dataloader."""
        return tuple(zip(*batch))

    # Create training and validation subsets
    train_dataset = Subset(dataset, train_indicies)
    val_dataset = Subset(dataset, val_indicies)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader

class StatsTracker:
    """Tracks and logs training statistics."""

    def __init__(self) -> None:
        self.train_loss_history = []
        self.val_metrics_history = (
            []
        )  # List to store dictionaries of validation metrics
        self.best_val_metric = None  # Initialize to None or a suitable default value

    def update_train_loss(self, loss: dict[str, float]) -> None:
        """Updates the training loss history.
        Args:
            loss: The training loss value for the current epoch.
        """
        self.train_loss_history.append(loss)

    def update_val_metrics(self, metrics: dict) -> bool:
        """Updates the validation metrics history and checks for a new best.
        Args:
            metrics: A dictionary containing validation metrics (e.g., AP@.50:.05:.95).

        Returns:
            True if a new best validation metric is achieved, False otherwise.
        """
        self.val_metrics_history.append(metrics)

        # Update best validation metric if applicable
        if (
            self.best_val_metric is None
            or metrics["AP@.50:.05:.95"] > self.best_val_metric["AP@.50:.05:.95"]
        ):
            self.best_val_metric = metrics
            return True

        return False

    def plot_stats(self, save_path: Optional[str] = None) -> None:
        """Plots the training loss and validation AP@.50:.05:.95.
        Args:
            save_path: Optional path to save the plot. If provided, the plot will be saved to this location.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(20, 15))
        fig.suptitle("Training Statistics")
        
        loss_values = [loss_dict["loss"] for loss_dict in self.train_loss_history]
        ax1.plot(loss_values, label="Train Loss", color="blue")
        ax1.set_ylabel("Mean Training Loss")
        ax1.legend()
        
        ap_values = [ep["AP@.50:.05:.95"] for ep in self.val_metrics_history]
        ax2.plot(ap_values, label="Validation AP@.50:.05:.95", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AP@.50:.05:.95")
        ax2.legend()

        if save_path:
            fig.savefig(save_path)
            
        plt.close(fig)

class TrainingLogger:
    """A logger that uses Python's logging module."""

    def __init__(self, csv_epochs_file: Optional[str] = None) -> None:
        """
        Initializes the TrainingLogger.
        Args:
            log_file: Optional path to a log file. If provided, logs will be written to this file.
        """
        self.log_file = csv_epochs_file
        if self.log_file:
            self.log_file = os.path.abspath(csv_epochs_file)
            self.log_file = os.path.abspath(csv_epochs_file)
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COCO_STATS_NAMES + LOSSES_NAMES)
            
            
        self.time_elapsed = 0
        self.best_val_metric = None

    def log_epoch_start(self, epoch: int, total_epochs: int, lr: float) -> None:
        self.time_elapsed = time.time()
        print(f"\nEpoch {epoch}/{total_epochs}; Learning rate: {lr}")


    def log_eval_start(self) -> None:
        print("Evaluating...")

    def log_epoch_end(self, epoch: int, train_losses: dict[str, float], eval_coco_metrics: dict[str, float]) -> None:
        if self.time_elapsed != 0:
            time_elapsed = time.time() - self.time_elapsed
            self.time_elapsed = 0
            
        print(f"Time: {time_elapsed}s; Epoch {epoch} Summary: ")
        print(f"\tTrain Mean Loss: {train_losses[LOSSES_NAMES[0]]:.4f}")
        for metric_name, metric_value in eval_coco_metrics.items():
            print(f"\tValidation {metric_name}: {metric_value:.3f}")

        if self.best_val_metric is None or eval_coco_metrics[COCO_STATS_NAMES[0]] > self.best_val_metric:
            self.best_val_metric = eval_coco_metrics[COCO_STATS_NAMES[0]]
            print(f"\tNew Best Validation {COCO_STATS_NAMES[0]}: {self.best_val_metric:.3f}")
        else:
            print(f"\tBest Validation {COCO_STATS_NAMES[0]}: {self.best_val_metric:.3f}")
        
        if self.log_file:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow(list(eval_coco_metrics.values()) + list(train_losses.values()))
        