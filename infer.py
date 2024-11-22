import argparse
import csv
import math
import os

import PIL
import numpy as np
from PIL.Image import Image
import torch
from torchvision.transforms.v2 import functional as F

from consts import NUM_TO_CLASSES
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
    num_of_annotations_to_save: int = 0,
    save_annotated_images: bool = False,
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
        predictions = predictor.get_predictions(images, iou_thresh, score_thresh, use_merge)
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

            if num_of_annotations_to_save > 0:
                num_of_annotations_to_save -= 1
                
                if save_annotated_images:
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
                        
def load_model(model_path: str):
    import torchvision
    """
    Loads a FasterRCNN_ResNet50_FPN_V2 model with the specified number of classes.

    Args:
        model_path (str): Path to the .pth model file.
        num_classes (int, optional): Number of classes in the model (including background). Defaults to 3.

    Returns:
        Predictor: A Predictor instance with the loaded FasterRCNN model.
    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 3) 

    model.load_state_dict(torch.load(model_path))
    return Predictor(model)  # Create and return a Predictor instance

def load_pathes(path: str) -> list[str]:
    """
    Loads image paths from a file or a directory.

    Args:
        path (str): Path to the image directory or file.

    Returns:
        list[str]: List of image paths.
    """
    if os.path.isfile(path):
        with open(path, "r") as file:
            return [line.strip() for line in file]
    elif os.path.isdir(path):
        return [os.path.join(path, file) for file in os.listdir(path) if file.endswith([
            ".jpeg", ".jpg", ".png"
            ])]
    else:
        raise ValueError(f"Invalid path: {path}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Run inference and generate statistics."
        )
    parser.add_argument(
        "model-path", 
        type=str, 
        help="Path to model weights (fasterrcnn_resnet50_fpn_v2)"
    )
    parser.add_argument(
        "images_path",  
        type=str, 
        help="path to image dir or a file contatining list of images to infer"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="output directory (default: output)"
    )
    parser.add_argument(
        "--iou-thresh", 
        type=float, 
        default=0.5, 
        help="IOU threshold for postproccessing (default: 0.5)"
    )
    parser.add_argument(
        "--score-thresh", 
        type=float, 
        default=0.5, 
        help="score threshold for object detection (default: 0.5)"
    )
    parser.add_argument(
        "--num_of_annotations_to_save", 
        type=int, 
        default=0, 
        help="number of annotated images to save (default: 0, -1 for all)"
    )
    parser.add_argument(
        "--save-annotated-images", 
        action="store_true", 
        help="save annotated images"
    )
    parser.add_argument(
        "--use-merge", 
        action="store_true", 
        help="use merge postprocessing"
    )
    parser.add_argument(
        "--images-per-batch",
        type=int,
        default=4,
        help="number of images to process in a batch (default: 4)",
    )
    parser.add_argument(
        "--image-size-factor",
        type=float,
        default=1.0,
        help="factor to resize input images (default: 1.0)",
    )
    parser.add_argument(
        "--detections-per-image",
        type=int,
        default=300,
        help="maximum number of detections per image (default: 300)",
    )
    parser.add_argument(
        "--detections-per-patch",
        type=int,
        default=100,
        help="maximum number of detections per patch (default: 100)",
    )
    parser.add_argument(
        "--patches-per-batch",
        type=int,
        default=4,
        help="number of patches to process in a batch (default: 4)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1024,
        help="size of each image patch (default: 1024)",
    )
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=0.2,
        help="overlap between adjacent patches (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")
    
    model = load_model(args.model_path)
    pathes = load_pathes(args.images_path)
      
    predictor = Predictor(
        model,
        device,
        images_per_batch=args.images_per_batch,
        image_size_factor=args.image_size_factor,
        detections_per_image=args.detections_per_image,
        detections_per_patch=args.detections_per_patch,
        patches_per_batch=args.patches_per_batch,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
    )
    
    infer(
        predictor,
        pathes,
        NUM_TO_CLASSES,
        args.output_dir,
        args.iou_thresh,
        args.score_thresh,
        args.use_merge,
        args.num_of_annotations_to_save,
        args.save_annotated_images
    )

if __name__ == "__main__":
    main()