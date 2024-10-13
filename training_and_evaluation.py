# Standard Library
import os
import math

# Third-party Libraries
from torchvision.transforms import v2
from tqdm import tqdm

# PyTorch
import torch
from torch import optim
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

# Torchvision
import torchvision.models.detection as tv_detection
from torchvision.models.detection import FasterRCNN

from aleket_dataset import AleketDataset
from metrics import LOSSES_NAMES, CocoEvaluator, Evaluator, COCO_STATS_NAMES
from utils import StatsTracker, TrainingLogger, load_checkpoint, save_checkpoint


def train(run_name: str,
          model:FasterRCNN,

          dataset: AleketDataset,
          default_transforms: v2.Transform, training_transforms: v2.Transform,
          train_dataloader: DataLoader, val_dataloader: DataLoader,

          total_epochs: int,
          warmup_epochs: int,

          learning_rate: float,
          momentum: float,
          weight_decay: float,

          evaluator: CocoEvaluator,

          device: torch.device,

          resume: bool = False,
          checkpoints: bool = False,
          verbose: bool = True,
          ):


    optimizer = SGD(model.parameters(), lr=learning_rate*10, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=warmup_epochs
    )

    result_path = os.path.abspath(f"result_{run_name}")
    os.makedirs(os.path.join(run_name, "checkpoints"), exist_ok=True)
    last_checkpoint_path = os.path.join(result_path, "run", "last.pth")
    best_checkpoint_bath = os.path.join(result_path, "run", "best.pth")
    validation_graph = os.path.join(result_path, "validation_graph")
    validation_log = os.path.join(result_path, "validation_log.csv")

    epoch_trained = 0
    stats_tracker = StatsTracker(validation_log)
    logger = TrainingLogger()

    if resume:
        print(f"Resuming from  {last_checkpoint_path}...")
        model, optimizer, lr_scheduler, epoch_trained = load_checkpoint(model, last_checkpoint_path)

    while epoch_trained < total_epochs:

        epoch = epoch_trained + 1

        if verbose:
            logger.log_epoch_start(epoch, total_epochs, lr_scheduler.get_last_lr()[0])

        dataset.transforms = training_transforms
        losses = train_one_epoch(
            model, optimizer, train_dataloader, device
        )

        dataset.transforms = default_transforms
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
    coco_eval: CocoEvaluator,
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

    e = Evaluator(dataloader.dataset)
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
            coco_eval.append(res)


    stats = coco_eval.eval()
    Evaluator.COCO = coco_eval
    my_stats = e.eval(dts)
    for name in COCO_STATS_NAMES:
        if name in my_stats:
            assert abs(my_stats[name] - stats[name]) < 1e-4

    coco_eval.clear_detections()
    
    return stats
