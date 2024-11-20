import json
import os
from typing import Any, Literal

import torch
from torch import GradScaler
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection import FasterRCNN
# Torchvision
from torchvision.transforms import v2

from StatsTracker import StatsTracker
from aleket_dataset import AleketDataset


def default_optimizer(model: FasterRCNN, params: dict = None) -> SGD:
    model_params = [p for p in model.parameters() if p.requires_grad]
    if params:
        return SGD(params=model_params, **params)
    else:
        return SGD(params=model_params)


def default_lr_scheduler(optimizer: torch.optim.Optimizer, params: dict = None) -> ReduceLROnPlateau:
    if params:
        return ReduceLROnPlateau(optimizer=optimizer, mode="min", **params)
    else:
        return ReduceLROnPlateau(optimizer=optimizer, mode="min")


def default_augmentation() -> dict:
    """
    Returns a dictionary containing default augmentation settings.

    The augmentation settings include random horizontal flip, vertical flip, perspective
    transformation, rotation, color jitter, and sharpness adjustment. Each augmentation
    has a probability (`p`) of being applied, and some have additional parameters
    like `distortion_scale`, `degrees`, etc.

    Returns:
        dict: A dictionary with augmentation settings.
    """
    return {
        "horizontal_flip": {
            "p": 0.5
        },
        "vertical_flip": {
            "p": 0.5
        },
        "scale_jitter": {
            "target_size": (1024, 1024),
            "scale_range": (0.8, 1.2)
        },
        "perspective": {
            "distortion_scale": 0.1,
            "p": 0.5
        },
        "rotation": {
            "degrees": 30,
            "expand": True
        },
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.05
        }
    }


def default_optimizer_params() -> dict:
    return {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001
    }


def default_lr_scheduler_params() -> dict:
    return {
        "factor": 0.5,
        "patience": 10,
        "min_lr": 0.0001
    }


class RunParams:
    """
    Stores and manages training parameters.

    Attributes:
        run_name (str): Name of the training run.
        train_set (dict): Dictionary specifying the training set indices.
        validation_set (dict): Dictionary specifying the validation set indices.
        augmentation (dict): Augmentation settings.
        batch_size (int): Batch size for training.
        dataloader_workers (int): Number of workers for the dataloader.
        total_epochs (int): Total number of training epochs.


    Methods:
        load(path): Loads parameters from a JSON file.
        save(path): Saves parameters to a JSON file.
    """

    def __init__(self,
                 run_name: str = "default",
                 train_set: dict[str, list[str]] = None,
                 validation_set: dict[str, list[str]] = None,

                 batch_size: int = 16,
                 dataloader_workers: int = 16,
                 total_epochs: int = 100,

                 augmentation: dict[str, Any] | Literal["DEFAULT"] = None,
                 optimizer: dict[str, Any] = None,
                 lr_scheduler: dict[str, Any] = None

                 ):

        if augmentation == "DEFAULT":
            augmentation = default_augmentation()
        if train_set is None:
            train_set = {}
        if validation_set is None:
            validation_set = {}
        if optimizer is None:
            optimizer = default_optimizer_params()
        if lr_scheduler is None:
            lr_scheduler = default_lr_scheduler_params()

        self.run_name = run_name
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.augmentation = augmentation
        self.train_set = train_set
        self.validation_set = validation_set

    def load(self, path: str):
        """Loads parameters from a JSON file."""
        with open(path, 'r') as file:
            state = json.load(file)
        self.__dict__.update(**state)

    def save(self, path: str):
        """Saves parameters to a JSON file."""
        with open(path, 'w') as file:
            json.dump(self.__dict__, file, indent=1)

    def parse(self, model: FasterRCNN, dataset: AleketDataset):
        """
        Parses training parameters and sets up training components.

        Args:
            params (TrainParams): Training parameters.
            model (FasterRCNN): The Faster R-CNN model.
            dataset (AleketDataset): The dataset.

        Returns:
            dict: A dictionary containing the training dataloader, validation dataloader,
                  optimizer, learning rate scheduler, augmentation pipeline, total epochs,
                  and run path.
        """
        train_names = []
        val_names = []

        for indices in self.train_set.values():
            train_names.extend(indices)
        for indices in self.validation_set.values():
            val_names.extend(indices)

        train_dataloader, val_dataloader = dataset.create_dataloaders(
            train_names,
            val_names,
            self.batch_size,
            self.dataloader_workers)

        optimizer = default_optimizer(model, self.optimizer)
        lr_scheduler = default_lr_scheduler(optimizer, self.lr_scheduler)
        run_path = os.path.join("results", self.run_name)
        total_epochs = self.total_epochs

        augmentation_list = []
        if self.augmentation is None:
            augmentation = None
        else:
            if "horizontal_flip" in self.augmentation:
                augmentation_list.append(v2.RandomHorizontalFlip(**self.augmentation["horizontal_flip"]))

            if "vertical_flip" in self.augmentation:
                augmentation_list.append(v2.RandomHorizontalFlip(**self.augmentation["vertical_flip"]))

            if "scale_jitter" in self.augmentation:
                augmentation_list.append(v2.ScaleJitter(**self.augmentation["scale_jitter"]))

            if "perspective" in self.augmentation:
                augmentation_list.append(v2.RandomPerspective(**self.augmentation["perspective"]))

            if "rotation" in self.augmentation:
                augmentation_list.append(v2.RandomRotation(**self.augmentation["rotation"]))

            if "color_jitter" in self.augmentation:
                augmentation_list.append(v2.ColorJitter(**self.augmentation["color_jitter"]))

        augmentation = v2.Compose(augmentation_list) if augmentation_list else None

        return {
            "train_loader": train_dataloader,
            "val_loader": val_dataloader,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "augmentation": augmentation,
            "total_epochs": total_epochs,
            "run_path": run_path,
        }


def save_checkpoint(model: FasterRCNN,
                    optimizer: SGD,
                    lr_scheduler: ReduceLROnPlateau,
                    epoch_trained: int,
                    checkpoint_path: str,
                    scaler: GradScaler,
                    stats_tracker: StatsTracker) -> None:
    """Saves the model's training checkpoint.

    Saves the current state of the model, optimizer, learning rate
    scheduler, and GradScaler to a file for later resumption of training.

    Args:
        model (FasterRCNN): The Faster R-CNN model to save.
        optimizer (SGD): The optimizer used for training the model.
        lr_scheduler (LinearLR): The learning rate scheduler used for training.
        epoch_trained (int): The number of epochs the model has been trained for.
        checkpoint_path (str): The path to save the checkpoint file.
        scaler (GradScaler): The GradScaler instance used for mixed precision training.
        stats_tracker (StatsTracker):
    """
    save_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "stats_tracker": stats_tracker,
        "epoch_trained": epoch_trained,
        "scaler_state_dict": scaler.state_dict()
    }
    torch.save(save_state, checkpoint_path)


def load_checkpoint(model: FasterRCNN,
                    checkpoint_path: str
                    ) -> tuple[FasterRCNN, SGD, ReduceLROnPlateau, int, GradScaler, StatsTracker]:
    """Loads a model checkpoint.

    Loads a previously saved model checkpoint from the specified path,
    including the model state, optimizer state, learning rate scheduler state,
    and GradScaler state.

    Args:
        model (FasterRCNN): The Faster R-CNN model to load the state into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the loaded model, optimizer,
               learning rate scheduler, the number of epochs trained, and the GradScaler instance.
    """
    save_state = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(save_state["model_state_dict"])
    optimizer = default_optimizer(model)  # Assuming you have a get_optimizer function
    lr_scheduler = default_lr_scheduler(optimizer)  # Assuming you have a get_lr_scheduler function
    epoch_trained = save_state["epoch_trained"]
    stats_tracker = save_state["stats_tracker"]
    scaler = GradScaler()  # Initialize a new GradScaler

    try:
        optimizer.load_state_dict(save_state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(save_state["lr_scheduler_state_dict"])
        scaler.load_state_dict(save_state["scaler_state_dict"])
    finally:
        return model, optimizer, lr_scheduler, epoch_trained, scaler, stats_tracker
