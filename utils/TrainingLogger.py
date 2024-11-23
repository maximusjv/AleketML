import time
from datetime import timedelta

from utils.consts import LOSSES_NAMES, PRIMARY_VALIDATION_METRIC


class TrainingLogger:
    """Logs training progress, including epoch time, training losses, and validation metrics."""

    def __init__(self):
        self.time_elapsed = 0
        self.best_val_metric = None

    def log_epoch_start(self, epoch, total_epochs, lr):
        """Logs the start of a new epoch, including the current learning rate.

        Args:
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs.
            lr (float): The learning rate for the current epoch.
        """
        self.time_elapsed = time.time()
        print(f"\nEpoch {epoch}/{total_epochs}; Learning rate: {lr}")

    def log_epoch_end(self, epoch, train_losses, eval_coco_metrics):
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

            print(
                f"Time: {str(timedelta(seconds=int(time_elapsed)))}; Epoch {epoch} Summary: "
            )
            print(f"\tTrain Mean Loss: {train_losses[LOSSES_NAMES[0]]:.4f}")
            for metric_name, metric_value in eval_coco_metrics.items():
                print(f"\tValidation {metric_name}: {metric_value:.3f}")

            if (
                self.best_val_metric is None
                or eval_coco_metrics[PRIMARY_VALIDATION_METRIC] < self.best_val_metric
            ):
                self.best_val_metric = eval_coco_metrics[PRIMARY_VALIDATION_METRIC]
                print(
                    f"\tNew Best Validation {PRIMARY_VALIDATION_METRIC}: {self.best_val_metric:.3f}"
                )
            else:
                print(
                    f"\tBest Validation {PRIMARY_VALIDATION_METRIC}: {self.best_val_metric:.3f}"
                )
