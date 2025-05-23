import argparse
import csv
import math
import os

import PIL
import numpy as np

import torch
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from data.checkpoints import get_default_model
from utils.visualize import visualize_bboxes

def stats_count(classes, prediction):
    """
    Calculate statistics for detected objects in an image.

    This function processes the prediction results from an object detection model
    and calculates various statistics such as count and area for each class of object.

    Parameters:
        classes (dict[int, str]): A dictionary mapping class IDs to class names.
        prediction (dict[str, np.array]): A dictionary containing prediction results.
            Must include 'boxes' and 'labels' keys with corresponding numpy arrays.

    Returns:
        dict: A dictionary containing the following keys:
            - 'bboxes': List of bounding boxes for all detected objects.
            - 'labels': List of class names for all detected objects.
            - 'count': Dictionary with class names as keys and object counts as values.
            - 'area': Dictionary with class names as keys and total object areas as values.

    Note:
        - Area is calculated as an ellipse (pi * width/2 * height/2) for each bounding box.
    """
    bboxes = np.asarray(prediction["boxes"])
    labels = np.asarray(prediction["labels"])

    labels_names = [classes[i] for i in labels]
    count = {}
    area = {}
    for class_id, class_name in classes.items():
        if class_name == "background":
            continue  # skip background
        bboxes_by_class = bboxes[np.where(labels == class_id)]
        count[class_name] = len(bboxes_by_class)
        area[class_name] = (
            np.sum(
                (bboxes_by_class[:, 2] - bboxes_by_class[:, 0])
                / 2.0
                * (bboxes_by_class[:, 3] - bboxes_by_class[:, 1])
                / 2.0
            )
            * math.pi
            if len(bboxes_by_class) > 0
            else 0
        )

    return {
        "bboxes": bboxes.tolist(),
        "labels": labels_names,
        "count": count,
        "area": area,
    }


def _save_statistics(stats_writer, image_name, stats, classes):
    """
    Saves object detection statistics to a CSV file.

    Args:
        stats_writer (csv.writer): CSV writer object.
        image_name (str): Name of the image.
        stats (dict): Dictionary containing area, count, labels, and bboxes.
        classes (dict[int, str]): Dictionary mapping class indices to class names.
    """
    area = stats["area"]
    count = stats["count"]

    row = [image_name]
    row.extend(
        [
            int(area[class_name])
            for class_name in classes.values()
            if class_name != "background"
        ]
    )
    row.extend(
        [
            int(count[class_name])
            for class_name in classes.values()
            if class_name != "background"
        ]
    )
    stats_writer.writerow(row)
    


def _save_annotated_image(
    image,
    image_name,
    bboxes,
    labels,
    annotated_dir,
):
    annotated_image_path = os.path.join(annotated_dir, f"{image_name}_annotated.jpeg")
    if isinstance(image, str):
        image = PIL.Image.open(image)
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
    visualize_bboxes(image, bboxes, labels, save_path=annotated_image_path)


def _save_annotations(
    image_name,
    bboxes,
    labels,
    bboxes_dir,
):
    """
    Saves annotations (bounding boxes and optionally annotated images) to files.

    Args:
        image (PIL.Image.Image or torch.Tensor or str): Image object, tensor, or path.
        image_name (str): Name of the image.
        bboxes (list): List of bounding boxes.
        labels (list): List of class labels.
        bboxes_dir (str): Directory to save bounding boxes.
    """

    bboxes_file_path = os.path.join(bboxes_dir, f"{image_name}.csv")
    with open(bboxes_file_path, "w", newline="") as bboxes_file:
        bboxes_writer = csv.writer(bboxes_file, delimiter=",")
        headers = ["xmin", "ymin", "xmax", "ymax", "class name"]
        bboxes_writer.writerow(headers)
        for (x1, y1, x2, y2), class_name in zip(bboxes, labels):
            bboxes_writer.writerow([int(x1), int(y1), int(x2), int(y2), class_name])


def infer(
    predictor,
    images,
    classes,
    output_dir,
    iou_thresh,
    score_thresh,
    save_annots=False,
    save_images=False,
    verbose=False,
):
    """
    Performs inference on a list of images and saves the results.

    Args:
        predictor (Predictor): The predictor object to use for inference.
        images (list[str | Image | torch.Tensor]): List of image paths, PIL Images, or tensors.
        classes (dict[int, str]): Dictionary mapping class indices to class names.
        output_dir (str): Directory to save the results.
        iou_thresh (float): IoU threshold for non-maximum suppression.
        score_thresh (float): Score threshold for object detection.
        save_annots (bool, optional): Where to save annotations. Defaults to False.
            Defaults to 0.
        save_images (bool, optional): Whether to save annotated images. Defaults to False.
        verbose (bool, optional): Whether to print infer progress or not. Defaults to False.
    """

    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stats_file_path = os.path.join(output_dir, "stats.csv")

    bboxes_dir = (
        os.path.join(output_dir, "bboxes") if save_annots else None
    )
    if bboxes_dir:
        os.makedirs(bboxes_dir, exist_ok=True)

    annotated_dir = (
        os.path.join(output_dir, "annotated")
        if save_images else None
    )
    if annotated_dir:
        os.makedirs(annotated_dir, exist_ok=True)

    with open(stats_file_path, "w", newline="") as stats_file:
        stats_writer = csv.writer(stats_file, delimiter=",")
        headers = ["Image"]
        headers.extend(
            [
                f"{class_name} area"
                for class_name in classes.values()
                if class_name != "background"
            ]
        )
        headers.extend(
            [
                f"{class_name} count"
                for class_name in classes.values()
                if class_name != "background"
            ]
        )
        stats_writer.writerow(headers)
        
        pb = tqdm(enumerate(images), total=len(images),desc="Predicting: ") if verbose else None
        
        for idx, image in enumerate(images):
            image = images[idx]
            image_name = os.path.basename(image) if isinstance(image, str) else str(idx)
            try: 
                pred = predictor.predict(
                    image,
                    iou_thresh,
                    score_thresh,
                )
    
                stats = stats_count(classes, pred)

                _save_statistics(stats_writer, image_name, stats, classes)
                stats_file.flush()
                if save_annots:
                    _save_annotations(
                        image_name,
                        stats["bboxes"],
                        stats["labels"],
                        bboxes_dir,
                    )
                if save_images:
                    _save_annotated_image(
                        image,
                        image_name,
                        stats["bboxes"],
                        stats["labels"],
                        annotated_dir,
                    )
            except Exception as e:
                print("Sorry unexcpeted error occured:")
                print(e)
                print(f"Skipping: {image_name} \n")
                raise e 
            except BaseException as e:
                stats_file.close()
                if pb:
                    pb.close()
                raise e 
            if pb:
                pb.update(1)      
        if pb:
            pb.close()
                

def load_model(model_path, device):
    """
    Loads a FasterRCNN_ResNet50_FPN_V2 model with the specified number of classes.

    Args:
        model_path (str): Path to the .pth model file.

    Returns:
        FasterRCNN: A loaded FasterRCNN model.
    """
    model = get_default_model(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def load_pathes(path):
    """
    Loads image paths from a file or a directory.

    Args:
        path (str): Path to the image directory or file.

    Returns:
        list[str]: List of image paths.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    
    if os.path.isfile(path):
        with open(path, "r") as file:
            return [os.path.join(parent,line.strip()) for line in file]
    elif os.path.isdir(path):
        return [
            os.path.join(path, file)
            for file in os.listdir(path)
            if file.endswith((".jpeg", ".jpg", ".png"))
        ]
    else:
        raise ValueError(f"Invalid path: {path}")