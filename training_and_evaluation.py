# Standard Library
import json
import os
import math
import shutil
from typing import Any

# Third-party Libraries
from torchvision.transforms import v2
from tqdm import tqdm

# PyTorch
import torch
from torch import optim
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Torchvision
import torchvision.models.detection as tv_detection
from torchvision.models.detection import FasterRCNN

from aleket_dataset import AleketDataset
from metrics import LOSSES_NAMES, Evaluator
from utils import StatsTracker, TrainingLogger, load_checkpoint, save_checkpoint, create_dataloaders


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
        "perspective": {
            "distortion_scale": 0.2,
            "p": 0.5
        },
        "rotation": {
            "degrees": 15,
            "expand": True
        },
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.1,
            "saturation": 0.05
        }
    }
    


class TrainParams:
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
        lr (float): Learning rate.
        lr_decay_factor (float): Factor for learning rate decay.
        lr_decay_milestones (list): Epochs at which to decay the learning rate.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.

    Methods:
        load(path): Loads parameters from a JSON file.
        save(path): Saves parameters to a JSON file.
    """
    
    def __init__(self,
                 run_name: str = "default",
                 train_set: dict[str, list[int]] = {},
                 validation_set: dict[str, list[int]] = {},

                 augmentation: dict[str, Any] = default_augmentation(),
                 batch_size: int = 16,
                 dataloader_workers: int = 16,
                 total_epochs: int = 100,

                 lr: float = 1e-3,
                 lr_decay_factor: float = 0.1,
                 lr_decay_milestones: list[int] = [50, 90],
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4):

        self.run_name = run_name
        self.augmentation = augmentation

        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.total_epochs = total_epochs
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_milestones = lr_decay_milestones
        self.momentum = momentum
        self.weight_decay = weight_decay
        
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



def parse_params(params: TrainParams, model:FasterRCNN, dataset: AleketDataset):
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
    train_indices = []
    val_indices = []

    for indices in params.train_set.values():
        train_indices.extend(indices)
    for indices in params.validation_set.values():
        val_indices.extend(indices)

    train_dataloader, val_dataloader = create_dataloaders(dataset,
                                                          train_indices,
                                                          val_indices,
                                                          params.batch_size,
                                                          params.dataloader_workers)

    optimizer = SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    lr_scheduler = MultiStepLR(
        optimizer, milestones=params.lr_decay_milestones, gamma=params.lr_decay_factor
    )
    run_path = os.path.join("results", params.run_name)
    total_epochs = params.total_epochs

    augmentation_list = []

    if "horizontal_flip" in params.augmentation:
        augmentation_list.append(v2.RandomHorizontalFlip(**params.augmentation["horizontal_flip"]))

    if "vertical_flip" in params.augmentation:
        augmentation_list.append(v2.RandomHorizontalFlip(**params.augmentation["vertical_flip"]))

    if "perspective" in params.augmentation:
        augmentation_list.append(v2.RandomPerspective(**params.augmentation["perspective"]))

    if "rotation" in params.augmentation:
        augmentation_list.append(v2.RandomRotation(**params.augmentation["rotation"]))

    if "color_jitter" in params.augmentation:
        augmentation_list.append(v2.ColorJitter(**params.augmentation["color_jitter"]))

    augmentation = v2.Compose(augmentation_list)

    return {
        "train_loader": train_dataloader,
        "val_loader": val_dataloader,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "augmentation": augmentation,
        "total_epochs": total_epochs,
        "run_path": run_path,
    }



def train(model:FasterRCNN,
          dataset: AleketDataset,
          params: TrainParams,
          device: torch.device,

          resume: bool = False,
          checkpoints: bool = False,
          verbose: bool = True,
          ):
    """
    Trains the Faster R-CNN model.

    Args:
        model (FasterRCNN): The Faster R-CNN model.
        dataset (AleketDataset): The dataset.
        params (TrainParams): Training parameters.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        resume (bool): Whether to resume training from a checkpoint.
        checkpoints (bool): Whether to save checkpoints during training.
        verbose (bool): Whether to print training progress.
    """
    parsed_params = parse_params(params, model, dataset)
    train_dataloader = parsed_params["train_loader"]
    val_dataloader = parsed_params["val_loader"]
    optimizer = parsed_params["optimizer"]
    lr_scheduler = parsed_params["lr_scheduler"]
    augmentation = parsed_params["augmentation"]
    total_epochs = parsed_params["total_epochs"]
    result_path = parsed_params["run_path"]


    last_checkpoint_path = os.path.join(result_path, "checkpoints", "last.pth")
    best_checkpoint_bath = os.path.join(result_path, "checkpoints", "best.pth")
    params_path = os.path.join(result_path, "params.json")
    validation_graph = os.path.join(result_path, "validation_graph")
    validation_log = os.path.join(result_path, "validation_log.csv")
    

    evaluator = Evaluator(val_dataloader.dataset)

    if resume:
        print(f"Resuming from  {last_checkpoint_path}...")
        model, optimizer, lr_scheduler, epoch_trained = load_checkpoint(model, last_checkpoint_path)
    else:
        if os.path.exists(result_path):
            shutil.rmtree(result_path)

    os.makedirs(os.path.join(result_path, "checkpoints"), exist_ok=True)
    
    params.save(params_path)
    
    epoch_trained = 0
    stats_tracker = StatsTracker(validation_log)
    logger = TrainingLogger()

    while epoch_trained < total_epochs:

        epoch = epoch_trained + 1

        if verbose:
            logger.log_epoch_start(epoch, total_epochs, lr_scheduler.get_last_lr()[0])

        dataset.augmentation = augmentation
        losses = train_one_epoch(
            model, optimizer, train_dataloader, device
        )

        dataset.augmentation = None
        eval_stats = evaluate(
            model, val_dataloader, evaluator, device
        )

        is_best = stats_tracker.update_stats(losses, eval_stats)
        stats_tracker.plot_stats(validation_graph)

        if verbose:
            logger.log_epoch_end(epoch, losses, eval_stats)


        lr_scheduler.step()

        epoch_trained = epoch
        if checkpoints:
            save_checkpoint(model, optimizer, lr_scheduler, epoch_trained, last_checkpoint_path)
            if is_best:
                save_checkpoint(model, optimizer, lr_scheduler, epoch_trained, best_checkpoint_bath)

    save_checkpoint(model, optimizer, lr_scheduler, epoch_trained, last_checkpoint_path)



def train_one_epoch(
    model: tv_detection.FasterRCNN,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Trains the model for one epoch.

    Args:
        model (tv_detection.FasterRCNN): The Faster R-CNN model.
        optimizer (optim.Optimizer): The optimizer for training.
        dataloader (DataLoader): The training dataloader.
        device (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').

    Returns:
        dict[str, float]: A dictionary containing the average losses for the epoch.
                          The keys are the loss names (e.g., 'loss', 'loss_classifier', etc.)
                          and the values are the corresponding average loss values.
    """
    model.train()
    size = len(dataloader)
    
    loss_values = { 
        key: 0 for key in LOSSES_NAMES
    }

    for batch_num, (images, targets) in tqdm(enumerate(dataloader), desc="Training batches", total=size):
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        
        loss_values['loss'] += loss.item()
        for loss_name, value in losses.items():
            loss_values[loss_name] += value.item()
            
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise Exception("Loss is infinite")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for loss_name, value in loss_values.items():
        loss_values[loss_name] = value/size
            
    return loss_values


def evaluate(
    model: tv_detection.FasterRCNN,
    dataloader: DataLoader,
    evaluator: Evaluator,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluates the model on the given dataloader using metrics.

    Args:
        model (tv_detection.FasterRCNN): The Faster R-CNN model to evaluate.
        dataloader (DataLoader): The dataloader containing the evaluation data.
        evaluator (Evaluator): The evaluator object for calculating metrics.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict[str, float]: A dictionary containing the evaluation statistics.
    """
    size = len(dataloader)
    model.eval()
    dts = {}

    for batch_num, (images, targets) in tqdm(enumerate(dataloader), desc="Evaluating batches", total=size):
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            res = {
                target["image_id"]: output
                for target, output in zip(targets, predictions)
            }
            dts.update(res)

    
    return evaluator.eval(dts)
