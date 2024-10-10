# Standard Library
import csv
import os
import time

from typing import  Optional

# PyTorch
from torch.utils.data import DataLoader, Subset

from aleket_dataset import AleketDataset
from training_and_evaluation import COCO_STATS_NAMES, LOSSES_NAMES
    
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
        train_indicies: Dataset indicies to train.
        val_indicies: Dataset indicies to validate.
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
        )  
        self.best_val_metric = None  

    def update_train_loss(self, loss: dict[str, float]) -> None:
        """Updates the training loss history.
        Args:
            loss: The training losses value for the current epoch.
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

    def __init__(self, stats_file: Optional[str] = None) -> None:
        """
        Initializes the TrainingLogger.
        Args:
            stats_file: Path to a write stats to.
        """
        self.stats_file = stats_file
        if self.stats_file:
            self.stats_file = os.path.abspath(stats_file)
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            if not os.path.exists(self.stats_file):
                with open(self.stats_file, 'w', newline='') as csvfile:
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
        
        if self.stats_file:
            with open(self.stats_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(eval_coco_metrics.values()) + list(train_losses.values()))
        