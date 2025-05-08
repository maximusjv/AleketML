# PHASE 1: Create YOLO annotations from original dataset
import json
import os
import random
import shutil
from collections import defaultdict
from statistics import mean

from PIL import Image, ImageDraw
from tqdm import tqdm

from . import autosplit_detect, setup_directories
from .load import load_split_list, load_yolo
from utils.boxes import Patch, crop_patches, make_patches


def prepare_yolo_dataset(config: dict):
    """
    Create a YOLO dataset from original annotations without patching.
    Just copy images and convert annotations to YOLO format.
    """
    random.seed(config["seed"])
    
    # Setup directories
    setup_directories(config)
    
    # Load dataset
    dataset_path = os.path.join(config["source"], "dataset.json")
    dataset = json.load(open(dataset_path))
    
    # Process each image
    for image_name, annotations in tqdm(dataset.items(), desc="Processing images"):
        # Source paths
        image_filename = f"{image_name}.jpeg"
        image_path = os.path.join(config["source"], "imgs", image_filename)
        
        # Destination paths
        dest_image_path = os.path.join(config["destination"], "images", image_filename)
        dest_label_path = os.path.join(config["destination"], "labels", f"{image_name}.txt")
        
        # Copy image if it doesn't exist already
        if not os.path.exists(dest_image_path):
            shutil.copy(image_path, dest_image_path)
        
        # Create YOLO format labels
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        with open(dest_label_path, "w") as f:
            wrote = False
            for cat, bbox in zip(annotations["category_id"], annotations["boxes"]):
                patch = Patch(*bbox)
                x, y, w, h = patch.xywh
                f.write(f"{cat} {x / img_width} {y / img_height} {w / img_width} {h / img_height}\n")
                wrote = True
            
            if not wrote:
                os.remove(dest_label_path)
    
    # Create autosplit files
    autosplit_detect(
        image_dir=os.path.join(config["destination"], "images"),
        output_dir=config["destination"],
        train_ratio=0.8,
    )
    
    # Save configuration
    config_path = os.path.join(config["destination"], "conversion_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"YOLO dataset created at {config['destination']}")


def compute_iou(boxA: Patch, boxB: Patch):
    """
    Compute the Intersection over Union (IoU) of two boxes.
    Boxes must be in (x1, y1, x2, y2) format.
    """
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    areaA = boxA.area
    areaB = boxB.area
    union = areaA + areaB - inter_area

    return inter_area / union


def save_crop(
    image: Image.Image, box_coords: Patch, dest_dir, class_label: str, base_name, crop_index
):
    """
    Crop the image using box_coords and save it in a folder named after the class.
    """
    crop = image.crop(box_coords.xyxy)
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

    for idx, row in enumerate(
        annotations
    ):
        class_id = row[-1]
        patch = Patch(*(row[:-1]))
        patch.expand(offset)
        x1, y1, x2, y2 = patch.xyxy    
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        box_abs = Patch(x1, y1, x2, y2)
        det_boxes.append(box_abs)
        class_counts[class_id] += 1
        save_crop(
            image, box_abs, dest_dir, classes.get(class_id, class_id), base_name, idx
        )

    return det_boxes, class_counts


def generate_background_crops(
    image: Image.Image,
    det_boxes: list[Patch],
    dest_dir: str,
    base_name: str,
    img_size: tuple[int,int],
    class_counts: dict,
    bg_iou_threshold: float,
    bg_attempts_multiplier: int,
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
    widths = [box.xmax - box.xmin for box in det_boxes]
    heights = [box.ymax - box.ymin for box in det_boxes]
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
        candidate_box = Patch(
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

    for split_name, split_file in split_files.items():
        image_paths = load_split_list(split_file)
        annotations = load_yolo(image_paths)
        
        print(f"Processing {split_name} split with {len(image_paths)} images.")

        for img_path in tqdm(image_paths, desc=f"Processing {split_name}"):
            
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            if not annotations[image_name]:
                continue
            try:
                image = Image.open(img_path)
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
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


def remove_background_images(root_dir: str, removal_percentage: float):
    """Remove a percentage of images that have no annotations."""
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")

    background_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if not os.path.exists(os.path.join(label_dir, f.replace(".jpeg", ".txt")))
    ]

    to_remove = random.sample(
        background_images, int(len(background_images) * removal_percentage)
    )

    for image_path in to_remove:
        os.remove(image_path)
        label_file = os.path.join(
            label_dir, os.path.basename(image_path).replace(".jpeg", ".txt")
        )
        if os.path.exists(label_file):
            os.remove(label_file)

    print(f"Removed {len(to_remove)} background images.")
    

def save_patch_annotations(f, annotations: list, patch: Patch, config: dict, draw_context: ImageDraw.ImageDraw, image_move: bool) -> bool:
    """Writes YOLO annotations to file for one patch."""
    wrote = False
    for row in annotations:
        cat = row[-1]
        bbox_patch = Patch(*(row[:-1]))
        relative_bbox = patch.clamp(bbox_patch)
        cropped_ratio = 1 - relative_bbox.area / bbox_patch.area if bbox_patch.area else 0

        if cropped_ratio > config["crop_tolerance"]:
            if cropped_ratio < 1 and config["erase_cropped"] and image_move:
                draw_context.rectangle(relative_bbox.xyxy, fill="black")
            continue

        x, y, w, h = [n / config["patch_size"] for n in relative_bbox.xywh]
        f.write(f"{cat} {x} {y} {w} {h}\n")
        wrote = True

    return wrote


def process_image_for_patching(image_name: str, annotations: list, config: dict):
    """Process a single image and generate its patches and annotations."""
    image_filename = f"{image_name}.jpeg"
    image_path = os.path.join(config["source"], "images", image_filename)
    image = Image.open(image_path)

    _, _, patch_boxes = make_patches(
        image.width, image.height, config["patch_size"], config["patch_overlap"]
    )
    
    patches = crop_patches(image, patch_boxes) if config["image_move"] else [None] * len(patch_boxes)
    
    for i, (patch, patched_image) in enumerate(zip(patch_boxes, patches)):
        patch_name = f"{image_name}_{i}.jpeg"
        patch_label_name = patch_name.replace(".jpeg", ".txt")
        patch_image_path = os.path.join(config["destination"], "images", patch_name)
        patch_label_path = os.path.join(
            config["destination"], "labels", patch_label_name
        )

        if patched_image is not None:
            patched_image.save(patch_image_path, quality=100)
            draw_context = ImageDraw.Draw(patched_image)
        else:
            draw_context = None

        with open(patch_label_path, "w") as f:
            wrote = save_patch_annotations(
                f, annotations, patch, config, draw_context, config["image_move"]
            )

        if not wrote:
            os.remove(patch_label_path)

def patch_yolo_dataset(config: dict):
    """
    Create a patched YOLO dataset from an existing YOLO dataset.
    """
    random.seed(config["seed"])
    
    # Setup directories
    setup_directories(config)
    
    annotations = load_yolo(config["source"])
    
    for image_name, annots in tqdm(annotations.items(), desc="Processing images for patching"):      
        # Process the image for patching
        process_image_for_patching(image_name, annots, config)
    
    if config["image_move"]:
        remove_background_images(config["destination"], config["background_removal"])
    
    # Create autosplit files
    autosplit_detect(
        image_dir=os.path.join(config["destination"], "images"),
        output_dir=config["destination"],
        train_ratio=0.8,
    )
    
    # Save configuration
    config_path = os.path.join(config["destination"], "patching_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Patched YOLO dataset created at {config['destination']}")
    
    return config["destination"]