import os
from collections import defaultdict
import random
import shutil
from PIL import Image
from tqdm import tqdm
from statistics import mean

from . import load_yolo_annotations, xyxy_to_xywh_center


def box_area(box: list[int]) -> float:
    """Calculate area of a bounding box in XYXY format."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two boxes.
    Boxes must be in (x1, y1, x2, y2) format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    areaA = box_area(boxA)
    areaB = box_area(boxB)
    union = areaA + areaB - inter_area

    return inter_area / union


def load_split_list(split_path):
    """Load image paths from a split file."""
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_crop(
    image: Image.Image, box_coords, dest_dir, class_label: str, base_name, crop_index
):
    """
    Crop the image using box_coords and save it in a folder named after the class.
    """
    crop = image.crop(box_coords)
    target_folder = os.path.join(dest_dir, str(class_label))
    os.makedirs(target_folder, exist_ok=True)
    save_path = os.path.join(target_folder, f"{base_name}_{crop_index}.jpeg")
    crop.save(save_path, quality=100)


def process_detection_crops(
    image: Image.Image,
    base_name: str,
    annotations: dict,
    dest_dir: str,
    offset: float,
    classes: dict,
):
    """
    Process detection boxes from a given label file:
      - Apply an optional offset.
      - Convert normalized YOLO coordinates to absolute pixel coordinates.
      - Save detection crops.

    Returns:
        det_boxes (list): List of detection boxes in absolute coordinates.
        class_counts (defaultdict): Count of detections per class.
    """
    img_width, img_height = image.size
    det_boxes = []
    class_counts = defaultdict(int)

    for idx, (class_id, box) in enumerate(
        zip(annotations["category_id"], annotations["boxes"])
    ):

        # YOLO normalized values: x_center, y_center, width, height
        x_center, y_center, box_width, box_height = xyxy_to_xywh_center(box)
        # Apply offset
        box_width *= 1 + offset
        box_height *= 1 + offset
        # Convert normalized coordinates to absolute pixel coordinates
        x1 = int((x_center - box_width / 2))
        y1 = int((y_center - box_height / 2))
        x2 = int((x_center + box_width / 2))
        y2 = int((y_center + box_height / 2))
        # Clamp coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        box_abs = (x1, y1, x2, y2)
        det_boxes.append(box_abs)

        class_counts[class_id] += 1

        save_crop(
            image, box_abs, dest_dir, classes.get(class_id, class_id), base_name, idx
        )

    return det_boxes, class_counts


def generate_background_crops(
    image,
    det_boxes,
    dest_dir,
    base_name,
    img_size,
    class_counts,
    bg_iou_threshold,
    bg_attempts_multiplier,
):
    """
    Generate background crops based on the mean size of detections. For each image, the number
    of background crops is set to the mean number of detections per class. Random candidate crops
    are generated and accepted if their maximum IoU with any detection box is below the threshold.

    Args:
        image (PIL.Image): The source image.
        det_boxes (list): List of detection boxes in absolute coordinates.
        dest_dir (str): Destination directory for saving crops.
        base_name (str): Base name for generating filenames.
        img_size (tuple): Size of image (width, height).
        class_counts (dict): Count of detections per class.
        bg_iou_threshold (float): IoU threshold for a candidate background crop.
        bg_attempts_multiplier (int): Attempts multiplier.
    """
    img_width, img_height = img_size

    # Calculate the average width and height for detection boxes.
    widths = [box[2] - box[0] for box in det_boxes]
    heights = [box[3] - box[1] for box in det_boxes]
    avg_width = int(mean(widths))
    avg_height = int(mean(heights))

    # Determine desired number of background crops as the mean count across detection classes.
    desired_bg = int(round(mean(list(class_counts.values())))) if class_counts else 0
    bg_saved = 0
    attempts = 0
    max_attempts = bg_attempts_multiplier * desired_bg if desired_bg > 0 else 0

    while bg_saved < desired_bg and attempts < max_attempts:
        attempts += 1
        if img_width - avg_width <= 0 or img_height - avg_height <= 0:
            break  # Image too small for background crop of this size.

        candidate_width = int(random.gauss(avg_width, 1))
        candidate_height = int(random.gauss(avg_height, 1))

        candidate_width = max(100, min(candidate_width, img_width))
        candidate_height = max(100, min(candidate_height, img_height))

        rand_x = random.randint(0, img_width - candidate_width)
        rand_y = random.randint(0, img_height - candidate_height)
        candidate_box = (
            rand_x,
            rand_y,
            rand_x + candidate_width,
            rand_y + candidate_height,
        )

        # Check candidate IoU with each detection box.
        max_iou = max(compute_iou(candidate_box, det_box) for det_box in det_boxes)
        if max_iou < bg_iou_threshold:
            save_crop(
                image,
                candidate_box,
                dest_dir,
                "background",
                base_name,
                f"bg_{bg_saved}",
            )
            bg_saved += 1

    if desired_bg > 0 and bg_saved < desired_bg:
        print(
            f"Image {base_name}: generated only {bg_saved} background crops (desired {desired_bg})."
        )


def prepare_classification_dataset(config: dict):
    """
    Converts a YOLO detection dataset into a classification dataset:
      - Processes detection crops with an optional offset.
      - Generates background crops if their maximum IoU with any detection box is below a threshold.

    The dataset is created for each split (e.g. "train", "val") as specified in split_files.

    Args:
        config (dict): Configuration dictionary containing various settings
        source_dir (str): Optional source directory. If None, uses config["destination"]
    """

    source_dir = config["source"]
    classes = config.get("classes", {})

    offset: float = config["classify_offset"]
    split_files = {
        "train": os.path.join(source_dir, "autosplit_train.txt"),
        "val": os.path.join(source_dir, "autosplit_val.txt"),
    }
    output_dir = os.path.join(source_dir, "classification")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    bg_iou_threshold: float = config["bg_iou_threshold"]
    bg_attempts_multiplier: int = 10

    annotations = load_yolo_annotations(source_dir)

    for split_name, split_file in split_files.items():
        image_paths = load_split_list(split_file)
        print(f"Processing {split_name} split with {len(image_paths)} images.")

        for rel_img_path in tqdm(image_paths, desc=f"Processing {split_name}"):

            image_path = os.path.normpath(os.path.join(source_dir, rel_img_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            if not annotations[image_name]["category_id"]:
                continue

            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Error opening {image_path}: {e}")
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            split_dest_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dest_dir, exist_ok=True)

            # Process detection crops: save crops and record bounding boxes.
            det_boxes, class_counts = process_detection_crops(
                image,
                base_name,
                annotations[image_name],
                split_dest_dir,
                offset,
                classes,
            )

            # Generate background crops, if there are detections.
            if det_boxes:
                generate_background_crops(
                    image=image,
                    det_boxes=det_boxes,
                    dest_dir=split_dest_dir,
                    base_name=base_name,
                    img_size=image.size,
                    class_counts=class_counts,
                    bg_iou_threshold=bg_iou_threshold,
                    bg_attempts_multiplier=bg_attempts_multiplier,
                )
    print(f"\nClassification dataset created at {output_dir}")
