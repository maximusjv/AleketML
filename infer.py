import csv
import math
import os
from time import time

import PIL
import numpy as np
from PIL.Image import Image
import torch
from torchvision.transforms.v2 import functional as F

from visualize import visualize_bboxes
from predictor import Predictor


def stats_count(
    classes: dict[int, str],
    prediction: dict[str, np.array],
) -> dict:
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


def infer(
    predictor: Predictor,
    images: list[str | Image | torch.Tensor],
    classes: dict[int, str],
    output_dir: str,
    iou_thresh: float,
    score_thresh: float,
    use_merge: bool = True,
    num_of_annotated_images_to_save: int = 0,
) -> None:

    if num_of_annotated_images_to_save == -1:
        num_of_annotated_images_to_save = len(images)

    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stats_file_path = os.path.join(output_dir, "stats.csv")

    bboxes_dir = (
        os.path.join(output_dir, "bboxes")
        if num_of_annotated_images_to_save > 0
        else None
    )
    if bboxes_dir:
        os.makedirs(bboxes_dir, exist_ok=True)

    annotated_dir = (
        os.path.join(output_dir, "annotated")
        if num_of_annotated_images_to_save > 0
        else None
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
        
        timeit = time()
        predictions = predictor.get_predictions(images, iou_thresh, score_thresh, use_merge)
        print(time() - timeit)
        
        for idx, pred in predictions.items():
            image = images[idx]
            image_name = os.path.basename(image) if isinstance(image, str) else str(idx)
            stats = stats_count(classes, pred)

            area = stats["area"]
            count = stats["count"]
            labels = stats["labels"]
            bboxes = stats["bboxes"]

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

            if num_of_annotated_images_to_save > 0:
                num_of_annotated_images_to_save -= 1
                annotated_image_path = os.path.join(
                    annotated_dir, f"{image_name}_annotated.jpeg"
                )
                if isinstance(image, str):
                    image = PIL.Image.open(image)
                if isinstance(image, torch.Tensor):
                    image = F.to_pil_image(image)

                visualize_bboxes(image, bboxes, labels, save_path=annotated_image_path)
                bboxes_file_path = os.path.join(bboxes_dir, f"{image_name}.csv")
                with open(bboxes_file_path, "w", newline="") as bboxes_file:
                    bboxes_writer = csv.writer(bboxes_file, delimiter=",")
                    headers = ["xmin", "ymin", "xmax", "ymax", "class name"]
                    bboxes_writer.writerow(headers)
                    for (x1, y1, x2, y2), class_name in zip(bboxes, labels):
                        bboxes_writer.writerow(
                            [int(x1), int(y1), int(x2), int(y2), class_name]
                        )
