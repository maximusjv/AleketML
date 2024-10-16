# Standard Library
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
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

# Torchvision
import torchvision.models.detection as tv_detection
from torchvision.models.detection import FasterRCNN

from aleket_dataset import AleketDataset
from metrics import LOSSES_NAMES, Evaluator
from utils import StatsTracker, TrainingLogger, load_checkpoint, save_checkpoint, create_dataloaders


class TrainParams:
    def __init__(self,
                 run_name: str,
                 augmentation: dict[str, Any],
                 train_set: dict[str, list[int]],
                 validation_set: dict[str, list[int]],
                 batch_size: int,
                 dataloader_workers: int,
                 total_epochs: int,

                 lr: float,
                 lr_decay_factor: float,
                 lr_decay_milestones: list[int],
                 momentum: float,
                 weight_decay: float):

        self.run_name = run_name
        self.augmentation = augmentation
        self.train_set = train_set
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.total_epochs = total_epochs
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_milestones = lr_decay_milestones
        self.momentum = momentum
        self.weight_decay = weight_decay


def default_params(name: str, train_set: dict[str, list[int]], validation_set: dict[str, list[int]]) -> TrainParams:
    augmentation = {
        "horizontal_flip": {
            "p": 0.5
        },
        "vertical_flip": {
            "p": 0.5
        },
        "perspective": {
            "distortion_scale": 0.1,
            "p": 0.5
        },
        "affine": {
            "degrees": 10,
            "translate": (0,0),
            "scale": (1,1),
        },
        "color_jitter": {
            "brightness": 0,
            "contrast": 0,
            "saturation": 0,
            "hue": 0,
        },
        "sharpness":{
          "p": 0.2,
          "sharpness_factor": 0.2
        }
    }
    return TrainParams(
        run_name=name,
        augmentation=augmentation,
        train_set=train_set,
        validation_set=validation_set,
        batch_size=32,
        dataloader_workers=4,
        total_epochs=100,
        lr=1e-3,
        lr_decay_factor=0.1,
        lr_decay_milestones=[50],
        momentum=0.9,
        weight_decay=1e-4,
    )



def parse_params(params: TrainParams, model:FasterRCNN, dataset: AleketDataset):

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

    if "affine" in params.augmentation:
        augmentation_list.append(v2.RandomAffine(**params.augmentation["affine"]))

    if "color_jitter" in params.augmentation:
        augmentation_list.append(v2.ColorJitter(**params.augmentation["color_jitter"]))

    if "sharpness" in params.augmentation:
        augmentation_list.append(v2.RandomAdjustSharpness(**params.augmentation["sharpness"]))

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
    epoch_trained = 0
    stats_tracker = StatsTracker(validation_log)
    logger = TrainingLogger()

    while epoch_trained < total_epochs:

        epoch = epoch_trained + 1

        if verbose:
            logger.log_epoch_start(epoch, total_epochs, lr_scheduler.get_last_lr()[0])

        dataset.augmentation = None
        losses = train_one_epoch(
            model, optimizer, train_dataloader, device
        )

        dataset.augmentation = augmentation
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
    """Trains the model for one epoch.
    Args:
        model: The Faster R-CNN model.
        optimizer: The optimizer for training.
        dataloader: The training dataloader.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
    Returns:
        The average loss for the epoch.
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
    """Evaluates the model on the given dataloader using COCO metrics.
    Args:
        model: The Faster R-CNN model to evaluate.
        dataloader: The dataloader containing the evaluation data.
        coco_eval: The COCO evaluator object for calculating metrics.
        device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the COCO evaluation statistics.
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
