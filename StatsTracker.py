import csv
import os
from typing import Optional

from matplotlib import pyplot as plt

from metrics import VALIDATION_METRICS, LOSSES_NAMES, PRIMARY_VALIDATION_METRIC


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

    def __init__(self, stats_file: Optional[str] = None, ) -> None:
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
                or eval_coco_metrics[PRIMARY_VALIDATION_METRIC] < self.best_val_metric[PRIMARY_VALIDATION_METRIC]):
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


        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle("Training Statistics")

        loss_values = [
            loss_dict["loss"] for loss_dict in self.train_loss_history
        ]
        ax1.plot(loss_values, label="Train Loss", color="blue")
        ax1.set_ylabel("Mean Training Loss")
        ax1.legend()

        ap_values = [
            ep["AP@0.50:0.95"] for ep in self.val_metrics_history
        ]
        aad_values = [
            ep["AAD"] for ep in self.val_metrics_history
        ]
        acd_values = [
            ep["ACD"] for ep in self.val_metrics_history
        ]

        ax2.plot(ap_values,
                 label="AP@0.50:0.95",
                 color="red")
        ax2.plot(aad_values,
            label="AAD",
            color="green")
        ax2.plot(acd_values,
            label="ACD",
            color="orange")

        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Validation")
        ax2.legend()

        if save_path:
            fig.savefig(save_path)

        plt.close(fig)
