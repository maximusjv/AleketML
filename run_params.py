
import json
import os
from typing import Any

# Torchvision
from torchvision.transforms import v2
from torchvision.models.detection import FasterRCNN

from aleket_dataset import AleketDataset
from utils import get_lr_scheduler, get_optimizer




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

     
def default_optimizer() -> dict:
    return {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001
    }
    

def default_lr_scheduler() -> dict:
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
                 
                 augmentation: dict[str, Any] = None,
                 optimizer: dict[str, Any] = None,
                 lr_scheduler: dict[str, Any] = None
                 
                ):

        if augmentation is None:
            augmentation = default_augmentation()
        if train_set is None:
            train_set = {}    
        if validation_set is None:
            validation_set = {}    
        if  optimizer is None:
            optimizer = default_optimizer()
        if lr_scheduler is None:
           lr_scheduler = default_lr_scheduler()
           
           
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




def parse_params(params: RunParams, model:FasterRCNN, dataset: AleketDataset):
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

    for indices in params.train_set.values():
        train_names.extend(indices)
    for indices in params.validation_set.values():
        val_names.extend(indices)

    train_dataloader, val_dataloader = dataset.create_dataloaders(
                                                                train_names,
                                                                val_names,
                                                                params.batch_size,
                                                                params.dataloader_workers)

    optimizer = get_optimizer(model, params.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, params.lr_scheduler)
    run_path = os.path.join("results", params.run_name)
    total_epochs = params.total_epochs

    augmentation_list = []

    if "horizontal_flip" in params.augmentation:
        augmentation_list.append(v2.RandomHorizontalFlip(**params.augmentation["horizontal_flip"]))

    if "vertical_flip" in params.augmentation:
        augmentation_list.append(v2.RandomHorizontalFlip(**params.augmentation["vertical_flip"]))
        
    if "scale_jitter" in params.augmentation:
        augmentation_list.append(v2.ScaleJitter(**params.augmentation["scale_jitter"]))

    if "perspective" in params.augmentation:
        augmentation_list.append(v2.RandomPerspective(**params.augmentation["perspective"]))

    if "rotation" in params.augmentation:
        augmentation_list.append(v2.RandomRotation(**params.augmentation["rotation"]))

    if "color_jitter" in params.augmentation:
        augmentation_list.append(v2.ColorJitter(**params.augmentation["color_jitter"]))

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
    