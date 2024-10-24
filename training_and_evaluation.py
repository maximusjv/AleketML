# Standard Library
import os
import math
import shutil

# Third-party Libraries
from tqdm import tqdm

# PyTorch
import torch
from torch import optim
from torch.utils.data import DataLoader

# Torchvision
import torchvision.models.detection as tv_detection
from torchvision.models.detection import FasterRCNN

from aleket_dataset import AleketDataset
from metrics import LOSSES_NAMES, VALIDATION_METRICS, Evaluator
from run_params import RunParams, parse_params
from utils import StatsTracker, TrainingLogger, load_checkpoint, save_checkpoint, create_dataloaders


def train(model:FasterRCNN,
          dataset: AleketDataset,
          params: RunParams,
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
    
    epoch_trained = 0
    evaluator = Evaluator(dataset, val_dataloader.dataset.indices)

    if resume:
        print(f"Resuming from  {last_checkpoint_path}...")
        model, optimizer, lr_scheduler, epoch_trained = load_checkpoint(model, last_checkpoint_path)
    else:
        if os.path.exists(result_path):
            shutil.rmtree(result_path)

    os.makedirs(os.path.join(result_path, "checkpoints"), exist_ok=True)
    
    params.save(params_path)
    
    
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


        lr_scheduler.step(stats_tracker.val_metrics_history[-1][VALIDATION_METRICS[0]])

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
