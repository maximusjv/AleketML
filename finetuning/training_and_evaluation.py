import math
import os

from tqdm import tqdm
import torch
from torch import GradScaler

from utils.StatsTracker import StatsTracker
from utils.TrainingLogger import TrainingLogger
from finetuning.checkpoints import save_checkpoint, load_checkpoint
from utils.consts import LOSSES_NAMES, PRIMARY_VALIDATION_METRIC
from finetuning.metrics import Evaluator


def train(
    model, dataset, params, device, resume=False, checkpoints=False, verbose=True
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
    parsed_params = params.parse(model, dataset)
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

    if not resume:
        os.makedirs(os.path.join(result_path, "checkpoints"), exist_ok=False)

    epoch_trained = 0
    scaler = GradScaler()
    dataset.augmentation = None
    evaluator = Evaluator(dataset.get_annots(val_dataloader.dataset.indices))
    stats_tracker = StatsTracker(validation_log)
    logger = TrainingLogger()

    if resume:
        print(f"Resuming from  {last_checkpoint_path}...")
        (
            model,
            optimizer,
            lr_scheduler,
            epoch_trained,
            scaler,
            loaded_stats_tracker,
        ) = load_checkpoint(model, last_checkpoint_path)
        logger.best_val_metric = loaded_stats_tracker.best_val_metric[
            PRIMARY_VALIDATION_METRIC
        ]
        stats_tracker.best_val_metric = loaded_stats_tracker.best_val_metric
        stats_tracker.train_loss_history = loaded_stats_tracker.train_loss_history
        stats_tracker.val_metrics_history = loaded_stats_tracker.val_metrics_history
        print(
            f"Last epoch:\n {stats_tracker.train_loss_history[-1]}, {stats_tracker.val_metrics_history[-1]}"
        )

    params.save(params_path)

    while epoch_trained < total_epochs:

        epoch = epoch_trained + 1

        if verbose:
            logger.log_epoch_start(epoch, total_epochs, lr_scheduler.get_last_lr()[0])

        dataset.augmentation = augmentation
        losses = train_one_epoch(model, optimizer, train_dataloader, device, scaler)

        dataset.augmentation = None
        eval_stats = evaluate(model, val_dataloader, evaluator, device)

        is_best = stats_tracker.update_stats(losses, eval_stats)
        stats_tracker.plot_stats(validation_graph)

        lr_scheduler.step(eval_stats[PRIMARY_VALIDATION_METRIC])

        epoch_trained = epoch
        if checkpoints:
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                epoch_trained,
                last_checkpoint_path,
                scaler,
                stats_tracker,
            )
            if is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_trained,
                    best_checkpoint_bath,
                    scaler,
                    stats_tracker,
                )
        if verbose:
            logger.log_epoch_end(epoch, losses, eval_stats)

    save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        epoch_trained,
        last_checkpoint_path,
        scaler,
        stats_tracker,
    )


def train_one_epoch(model, optimizer, dataloader, device, scaler):
    """
    Trains the model for one epoch using mixed precision training.

    Args:
        model (FasterRCNN): The Faster R-CNN model.
        optimizer (optim.Optimizer): The optimizer for training.
        dataloader (DataLoader): The training dataloader.
        device (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').
        scaler (GradScaler): The GradScaler instance for mixed precision training.

    Returns:
        dict[str, float]: A dictionary containing the average losses for the epoch.
                         The keys are the loss names (e.g., 'loss', 'loss_classifier', etc.)
                         and the values are the corresponding average loss values.
    """
    model.train()
    size = len(dataloader)

    loss_values = {key: 0 for key in LOSSES_NAMES}

    for images, targets in tqdm(dataloader, desc="Training batches"):

        images = [img.to(device) for img in images]
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())

        loss_values["loss"] += loss.item()
        for loss_name, value in losses.items():
            loss_values[loss_name] += value.item()

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise Exception("Loss is infinite")

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # Scale the loss before backpropagation
        scaler.step(optimizer)
        scaler.update()

    for loss_name, value in loss_values.items():
        loss_values[loss_name] = value / size

    return loss_values


def evaluate(model, dataloader, evaluator, device):
    """
    Evaluates the model on the given dataloader using metrics.

    Args:
        model (FasterRCNN): The Faster R-CNN model to evaluate.
        dataloader (DataLoader): The dataloader containing the evaluation data.
        evaluator (Evaluator): The evaluator object for calculating metrics.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict[str, float]: A dictionary containing the evaluation statistics.
    """
    model.eval()
    dts = {}

    for images, targets in tqdm(dataloader, desc="Evaluating batches"):
        images = [img.to(device) for img in images]
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16
        ):
            predictions = model(images)
            res = {
                target["image_id"]: output
                for target, output in zip(targets, predictions)
            }
            dts.update(res)

    return evaluator.eval(dts)
