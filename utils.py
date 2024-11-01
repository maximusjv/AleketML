# Standard Library
import math
import csv
from datetime import timedelta
import os
import time
from typing import Optional

# Third-Party Libraries
from PIL import Image
import numpy as np


# PyTorch
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from aleket_dataset import AleketDataset
from metrics import VALIDATION_METRICS, LOSSES_NAMES


def make_patches(
          img: Image.Image,
          patch_size: int,
          overlap: float,
):
    overlap_size = int(patch_size * overlap)
    no_overlap_size = patch_size - overlap_size

    imgs_per_width = math.ceil(float(img.width) / no_overlap_size)
    imgs_per_height = math.ceil(float(img.height) / no_overlap_size)

    padded_height = imgs_per_width * no_overlap_size + overlap_size
    padded_width = imgs_per_width * no_overlap_size + overlap_size

    padded_img = Image.new("RGB", (padded_width, padded_height))
    padded_img.paste(img)

    patched_images = []
    patch_boxes = []

    for row in range(imgs_per_height):
        for col in range(imgs_per_width):
            xmin, ymin = col * no_overlap_size, row * no_overlap_size
            xmax, ymax = xmin + patch_size, ymin + patch_size
            patch_box = (xmin, ymin, xmax, ymax)
            patched_image = padded_img.crop(patch_box)

            patch_boxes.append(patch_box)
            patched_images.append(patched_image)

    return patched_images, patch_boxes


def split_dataset(dataset: AleketDataset,
                  dataset_fraction: float, 
                  validation_fraction: float,
                  generator: np.random.Generator,
                  ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Splits the dataset into train and validation sets.

    Splits the dataset into train and validation sets, ensuring that all patches 
    from the same full image are kept together in the same set.

    Args:
        dataset (AleketDataset): The dataset to split.
        dataset_fraction (float): The fraction of the dataset to use (for debugging/testing).
        validation_fraction (float): The fraction of the used dataset to allocate for validation.
        generator (np.random.Generator): A NumPy random generator for reproducible splitting.

    Returns:
        tuple[dict[str, list[int]], dict[str, list[int]]]: A tuple containing two dictionaries:
            - The first dictionary maps full image IDs to lists of patch indices for the training set.
            - The second dictionary maps full image IDs to lists of patch indices for the validation set.
    """
    by_full_images = dataset.by_full_images()
    
    full_images = list(by_full_images.keys())
    full_images = generator.permutation(full_images)
    
    total_num_samples = max(2,int(len(dataset.images) * dataset_fraction))
    validation_num_samples = max(1,int(validation_fraction * total_num_samples))
    train_num_samples = total_num_samples - validation_num_samples
    
    train_len = 0
    train_set = {}
    validation_len = 0
    validation_set = {}
    
    for images in full_images:
        if validation_len < validation_num_samples:
            validation_set[images] = by_full_images[images]
            validation_len += len(by_full_images[images])
        elif train_len < train_num_samples:
            train_set[images] = by_full_images[images]
            train_len += len(by_full_images[images])
    
    
    return train_set, validation_set
    
    
def collate_fn(batch):
    """Collates data samples into batches for the dataloader."""
    return tuple(zip(*batch))


# Dataset split
def create_dataloaders(
    dataset: AleketDataset,
    train_indices: list[int],
    val_indices: list[int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
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
    
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    
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
    """
    Tracks and logs training statistics, including losses and evaluation metrics.

    This class keeps track of training losses and validation metrics during the training process.
    It can also save these statistics to a CSV file and plot the training loss and a specified
    validation metric (e.g., AP@.50:.05:.95) over epochs.

    Args:
        stats_file (Optional[str]): Path to the CSV file where the statistics will be saved. 
                                    If None, the statistics are not saved to a file.
    """

    def __init__(self, stats_file: Optional[str] = None) -> None:
        self.train_loss_history = []
        self.val_metrics_history = []
        self.best_val_metric = None

        self.stats_file = stats_file
        if self.stats_file:
            self.stats_file = os.path.abspath(stats_file)
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            if not os.path.exists(self.stats_file):
                with open(self.stats_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(VALIDATION_METRICS + LOSSES_NAMES)

    def update_stats(self, train_losses: dict[str, float],
                     eval_coco_metrics: dict[str, float]):
        """
        Updates the training statistics with the latest loss and metrics.

        Appends the provided training losses and validation metrics to their respective history lists.
        Also, updates the best validation metric if the current metric is better than the previous best.
        If a `stats_file` is provided, the updated statistics are written to the CSV file.

        Args:
            train_losses (dict[str, float]): A dictionary of training loss values.
            eval_coco_metrics (dict[str, float]): A dictionary of COCO evaluation metrics.
        """

        self.train_loss_history.append(train_losses)
        self.val_metrics_history.append(eval_coco_metrics)

        is_best = False
        
        if (self.best_val_metric is None
                or eval_coco_metrics["AP@.50:.05:.95"] > self.best_val_metric
                ["AP@.50:.05:.95"]):
            is_best = True
            self.best_val_metric = eval_coco_metrics

        if self.stats_file:
            with open(self.stats_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(eval_coco_metrics.values()) +
                                list(train_losses.values()))
        return is_best

    def plot_stats(self, save_path: str) -> None:
        """
        Plots the training loss and validation AP@.50:.05:.95 over epochs.

        Generates a plot showing the trend of the training loss and the specified validation metric
        (e.g., AP@.50:.05:.95) over the training epochs. The plot is save to specified path.

        Args:
            save_path (str): The path to save the plot to.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle("Training Statistics")

        loss_values = [
            loss_dict["loss"] for loss_dict in self.train_loss_history
        ]
        ax1.plot(loss_values, label="Train Loss", color="blue")
        ax1.set_ylabel("Mean Training Loss")
        ax1.legend()

        ap_values = [
            ep["AP@.50:.05:.95"] for ep in self.val_metrics_history
        ]
        ax2.plot(ap_values,
                 label="Validation AP@.50:.05:.95",
                 color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AP@.50:.05:.95")
        ax2.legend()

        if save_path:
            fig.savefig(save_path)

        plt.close(fig)


class TrainingLogger:
    """Logs training progress, including epoch time, training losses, and validation metrics."""

    def __init__(self) -> None:
        self.time_elapsed = 0
        self.best_val_metric = None

    def log_epoch_start(self, epoch: int, total_epochs: int,
                         lr: float) -> None:
        """Logs the start of a new epoch, including the current learning rate.

        Args:
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs.
            lr (float): The learning rate for the current epoch.
        """
        self.time_elapsed = time.time()
        print(f"\nEpoch {epoch}/{total_epochs}; Learning rate: {lr}")

    def log_epoch_end(self, epoch: int, train_losses: dict[str, float],
                       eval_coco_metrics: dict[str, float]) -> None:
        """
        Logs the end of an epoch, including training losses, validation metrics, and the best validation metric so far.

        Args:
            epoch (int): The current epoch number.
            train_losses (dict[str, float]): A dictionary of training loss values.
            eval_coco_metrics (dict[str, float]): A dictionary of COCO evaluation metrics.
        """

        if self.time_elapsed != 0:
            time_elapsed = time.time() - self.time_elapsed
            self.time_elapsed = 0

            print(f"Time: {str(timedelta(seconds=int(time_elapsed)))}; Epoch {epoch} Summary: ")
            print(f"\tTrain Mean Loss: {train_losses[LOSSES_NAMES[0]]:.4f}")
            for metric_name, metric_value in eval_coco_metrics.items():
                print(f"\tValidation {metric_name}: {metric_value:.3f}")

            if (self.best_val_metric is None
                    or eval_coco_metrics[VALIDATION_METRICS[0]] >
                    self.best_val_metric):
                self.best_val_metric = eval_coco_metrics[VALIDATION_METRICS[0]]
                print(
                    f"\tNew Best Validation {VALIDATION_METRICS[0]}: {self.best_val_metric:.3f}"
                )
            else:
                print(
                    f"\tBest Validation {VALIDATION_METRICS[0]}: {self.best_val_metric:.3f}"
                )


def get_model(device: torch.device, trainable_backbone_layers: int = 3) -> FasterRCNN:
    """
    Loads a pretrained Faster R-CNN ResNet-50 FPN model and modifies the classification head 
    to accommodate the specified number of classes in dataset (3 - including background).
    Args:
        device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').

    Returns:
        FasterRCNN: The Faster R-CNN model with the modified classification head.
    """
    model = fasterrcnn_resnet50_fpn(
        weights = "DEFAULT",
        trainable_backbone_layers = trainable_backbone_layers
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features, 3
        )
    )
    return model.to(device)


def get_optimizer(model: FasterRCNN, params: dict = None) -> SGD:
    mparams = [p for p in model.parameters() if p.requires_grad]
    if params:
          return SGD(params=mparams, **params)
    else:
          return SGD(params=mparams)
  

def get_lr_scheduler(optimizer: torch.optim.Optimizer, params: dict = None) -> ReduceLROnPlateau:
    if params:
        return ReduceLROnPlateau(optimizer=optimizer, mode="max", **params)  
    else:
        return ReduceLROnPlateau(optimizer=optimizer, mode="max")
    
    
def save_checkpoint(model: FasterRCNN,
                    optimizer: SGD,
                    lr_scheduler: ReduceLROnPlateau,
                    epoch_trained: int,
                    checkpoint_path: str) -> None:
    """Saves the model's training checkpoint.

    Saves the current state of the model, optimizer, and learning rate 
    scheduler to a file for later resumption of training.

    Args:
        model (FasterRCNN): The Faster R-CNN model to save.
        optimizer (SGD): The optimizer used for training the model.
        lr_scheduler (LinearLR): The learning rate scheduler used for training.
        epoch_trained (int): The number of epochs the model has been trained for.
        checkpoint_path (str): The path to save the checkpoint file.
    """
    save_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch_trained": epoch_trained,
    }
    torch.save(save_state, checkpoint_path)


def load_checkpoint(model: FasterRCNN, checkpoint_path: str) -> tuple[FasterRCNN, SGD, ReduceLROnPlateau, int]:
    """Loads a model checkpoint.

    Loads a previously saved model checkpoint from the specified path, 
    including the model state, optimizer state, and learning rate scheduler state.

    Args:
        model (FasterRCNN): The Faster R-CNN model to load the state into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the loaded model, optimizer, 
               learning rate scheduler, and the number of epochs trained.
    """
    save_state = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(save_state["model_state_dict"])
    optimizer = get_optimizer(model)
    optimizer.load_state_dict(save_state["optimizer_state_dict"])

    epoch_trained = save_state["epoch_trained"]

    lr_scheduler = get_lr_scheduler(optimizer)
    lr_scheduler.load_state_dict(save_state["lr_scheduler_state_dict"])

    return model, optimizer, lr_scheduler, epoch_trained

