# Standard Library
import os
import json
from typing import Any, Optional
import logging

# Third-party Libraries
import PIL.Image

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset

# Torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors


# AleketDataset
def load_annotations(file: str) -> dict[str, Any]:
    """Loads annotations from a JSON file.

    Args:
        file: Path to the JSON annotation file.

    Returns:
        A dictionary containing the loaded annotations.
    """
    with open(file, "r") as annot_file:
        dataset = json.load(annot_file)
    return dataset


class AleketDataset(Dataset):
    """Custom dataset for Aleket images and annotations.

    Attributes:
        NUM_TO_CLASSES: Mapping of numerical labels to class names.
        CLASSES_TO_NUM: Mapping of class names to numerical labels.
    """

    NUM_TO_CLASSES = {"placeholder": 0, "healthy": 1, "nec": 2}
    CLASSES_TO_NUM = {0: "placeholder", 1: "healthy", 2: "nec"}

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
        self.dataset = load_annotations(os.path.join(dataset_dir, "dataset.json"))
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

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 8))
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
            
        return fig


class TrainingLogger:
    """A logger that uses Python's logging module."""

    def __init__(self, name: str, log_file: Optional[str] = None, batch_print=True) -> None:
        """
        Initializes the TrainingLogger.
        Args:
            log_file: Optional path to a log file. If provided, logs will be written to this file.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)  # Set the default log level
        self.batch_print = batch_print

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)

        # Add file handler if log_file is provided
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(name)s - %(message)s"))
            self.logger.addHandler(fh)

        self.best_val_metric = None

    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.logger.info(f"Epoch {epoch}/{total_epochs}")

    def log_batch(
        self,
        batch: int,
        total_batches: int,
        time_elapsed: Optional[float] = None,
        losses_dict: Optional[dict] = None,
        total_loss: Optional[float] = None,
    ) -> None:
        """Logs training or evaluation metrics to the logger.
        Args:
            batch: Current batch number.
            total_batches: Total number of batches.
            time_elapsed: Time elapsed for the current batch (optional).
            losses_dict: Dictionary of individual losses (optional).
            total_loss: Total loss value (optional).
        """
        if self.batch_print:
            time_str = f" time: {time_elapsed:.2f}s" if time_elapsed is not None else ""
            loss_msg = f"Loss: {total_loss:.4f} " if total_loss is not None else ""
            loss_dict_str = (
                f'loss_classifier: {losses_dict["loss_classifier"]:.4f} '
                f'loss_box_reg: {losses_dict["loss_box_reg"]:.4f} '
                f'loss_objectness: {losses_dict["loss_objectness"]:.4f} '
                f'loss_rpn_box_reg: {losses_dict["loss_rpn_box_reg"]:.4f}'
                if losses_dict is not None
                else ""
            )

            self.logger.info(
                f"[{batch}/{total_batches}] {loss_msg}{loss_dict_str}{time_str}"
            )

    def log_eval_start(self) -> None:
        self.logger.info("Evaluating...")

    def log_epoch_end(self, epoch: int, train_loss: float, eval_metrics: dict) -> None:
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Train Mean Loss: {train_loss:.4f}")
        for metric_name, metric_value in eval_metrics.items():
            self.logger.info(f"  Validation {metric_name}: {metric_value:.3f}")

        if self.best_val_metric is None or eval_metrics["AP@.50:.05:.95"] > self.best_val_metric:
            self.best_val_metric = eval_metrics["AP@.50:.05:.95"]
            self.logger.info(f"  New Best Validation AP@.50:.05:.95: {self.best_val_metric:.3f}")
        else:
            self.logger.info(f"  Best Validation AP@.50:.05:.95: {self.best_val_metric:.3f}")


# TRAIN AND EVAL UTILS
