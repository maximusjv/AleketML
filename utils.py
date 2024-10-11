# Standard Library
import csv
import os
import time

# Third-Party Libraries
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR

# PyTorch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN


from aleket_dataset import AleketDataset
from metrics import COCO_STATS_NAMES, LOSSES_NAMES



def split_dataset(dataset: AleketDataset,
                  dataset_fraction: float, 
                  validation_fraction: float,
                  generator: np.random.Generator,
                  ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    
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
    
    def collate_fn(batch):
        """Collates data samples into batches for the dataloader."""
        return tuple(zip(*batch))

    # Create training and validation subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

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

    def __init__(self, stats_file: str = None) -> None:
        self.train_loss_history = []
        self.val_metrics_history = (
            []
        )  
        self.best_val_metric = None

        self.stats_file = stats_file
        if self.stats_file:
            self.stats_file = os.path.abspath(stats_file)
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            if not os.path.exists(self.stats_file):
                with open(self.stats_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COCO_STATS_NAMES + LOSSES_NAMES)

    def update_stats(self, train_losses: dict[str, float], eval_coco_metrics: dict[str, float]):

        self.train_loss_history.append(train_losses)
        self.val_metrics_history.append(eval_coco_metrics)

        is_best = False
        if (
                self.best_val_metric is None
                or eval_coco_metrics["AP@.50:.05:.95"] > self.best_val_metric["AP@.50:.05:.95"]
        ):
            self.best_val_metric = eval_coco_metrics
            is_best = True

        if self.stats_file:
            with open(self.stats_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(eval_coco_metrics.values()) + list(train_losses.values()))


    def plot_stats(self, save_path: str = None) -> None:
        """Plots the training loss and validation AP@.50:.05:.95.
        Args:
            save_path: Path to save the plot.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))
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

    def __init__(self) -> None:
        """
        Initializes the TrainingLogger.
        """

        self.time_elapsed = 0
        self.best_val_metric = None

    def log_epoch_start(self, epoch: int, total_epochs: int, lr: float) -> None:
        self.time_elapsed = time.time()
        print(f"\nEpoch {epoch}/{total_epochs}; Learning rate: {lr}")

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
        


# Save training state
def save_checkpoint(model: FasterRCNN,
                    optimizer: SGD,
                    lr_scheduler: LinearLR,
                    epoch_trained: int,
                    checkpoint_path: str) -> None:
    save_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch_trained": epoch_trained,
    }
    torch.save(save_state, checkpoint_path)


# Load training state
def load_checkpoint(
        model: FasterRCNN,
        checkpoint_path: str) -> tuple[FasterRCNN, SGD, LinearLR, int]:
    save_state = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(save_state["model_state_dict"])
    optimizer = SGD(model.parameters())
    optimizer.load_state_dict(save_state["optimizer_state_dict"])

    epoch_trained = save_state["epoch_trained"]

    lr_scheduler = LinearLR(optimizer, last_epoch=epoch_trained)
    lr_scheduler.load_state_dict(save_state["lr_scheduler_state_dict"])

    return model, optimizer, lr_scheduler, epoch_trained

