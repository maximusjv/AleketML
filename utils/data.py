import json
from collections import defaultdict
import os
from PIL import Image
from tqdm import tqdm
import shutil
import random
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm
from statistics import mean
import gdown
import zipfile

from utils.patches import make_patches, crop_patches

def gdownload(url: str):
    """
    Downloads a zip file from the given URL using gdown and decompresses it.

    Args:
        url: The URL of the zip file to download.
    """
    try:
        # Get the filename from the URL
        filename = os.path.basename(url)

        # Download the zip file using gdown
        print(f"Downloading '{filename}' from '{url}'...")
        gdown.download(url, output=filename, quiet=False)
        print(f"Successfully downloaded '{filename}'.")

        # Decompress the zip file
        print(f"Decompressing '{filename}'...")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall()
        print(f"Successfully decompressed '{filename}'.")

        os.remove(filename)
        print(f"Removed downloaded zip file '{filename}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


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


def setup_directories(config: dict):
    """Prepare destination directories based on config."""
    dst = config["destination"]
    if os.path.exists(dst):
        if config["image_move"]:
            shutil.rmtree(dst)
        else:
            shutil.rmtree(os.path.join(dst, "labels"))

    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)


def save_patch_annotations(
    f, annotations, patch, config, draw_context, image_move: bool
) -> bool:
    """Writes YOLO annotations to file for one patch."""
    wrote = False
    for cat, bbox in zip(annotations["category_id"], annotations["boxes"]):
        relative_bbox = patch.clamp_box(bbox)
        cropped_ratio = 1 - box_area(relative_bbox) / box_area(bbox)

        if cropped_ratio > config["crop_tolerance"]:
            if cropped_ratio < 1 and config["erase_cropped"] and image_move:
                draw_context.rectangle(relative_bbox, fill="black")
            continue

        x_center = (relative_bbox[0] + relative_bbox[2]) / 2 / config["patch_size"]
        y_center = (relative_bbox[1] + relative_bbox[3]) / 2 / config["patch_size"]
        width = (relative_bbox[2] - relative_bbox[0]) / config["patch_size"]
        height = (relative_bbox[3] - relative_bbox[1]) / config["patch_size"]

        f.write(f"{cat} {x_center} {y_center} {width} {height}\n")
        wrote = True

    return wrote


def process_image(image_name: str, annotations: dict, config: dict):
    """Process a single image and generate its patches and annotations."""
    image_filename = f"{image_name}.jpeg"
    image_path = os.path.join(config["source"], "imgs", image_filename)
    image = Image.open(image_path)

    _, _, patch_boxes = make_patches(
        image.width, image.height, config["patch_size"], config["patch_overlap"]
    )
    
    patches = crop_patches(image,patch_boxes) if config["image_move"] else [None] * len(patch_boxes)
    
    for i, (patch, patched_image) in enumerate(zip(patch_boxes, patches)):
        patch_name = f"{image_name}_{i}.jpg"
        patch_label_name = patch_name.replace(".jpg", ".txt")
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


def remove_background_images(root_dir: str, removal_percentage: float):
    """Remove a percentage of images that have no annotations."""
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")

    background_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if not os.path.exists(os.path.join(label_dir, f.replace(".jpg", ".txt")))
    ]

    to_remove = random.sample(
        background_images, int(len(background_images) * removal_percentage)
    )

    for image_path in to_remove:
        os.remove(image_path)
        label_file = os.path.join(
            label_dir, os.path.basename(image_path).replace(".jpg", ".txt")
        )
        if os.path.exists(label_file):
            os.remove(label_file)

    print(f"Removed {len(to_remove)} background images.")


def autosplit_detect(
    image_dir: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42
) -> tuple[str, str]:
    """
    Auto-splits dataset patches into train and val splits ensuring all patches from
    the same original image are in the same split.

    Args:
        image_dir (str): Path to the "images" folder containing patches.
        output_dir (str): Directory to save autosplit text files.
        train_ratio (float): Proportion of the dataset to assign to training.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[str, str]: Paths to autosplit_train.txt and autosplit_val.txt.
    """

    # Step 1: Group patch images by their origin image
    grouped_patches = defaultdict(list)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            origin_image = "_".join(filename.split("_")[:-1])  # Remove the patch number
            grouped_patches[origin_image].append("./" + os.path.join("images", filename).replace("\\", "/"))

    origin_keys = list(grouped_patches.keys())
    random.shuffle(origin_keys)

    train_cutoff = int(len(origin_keys) * train_ratio)
    train_keys = set(origin_keys[:train_cutoff])
    val_keys = set(origin_keys[train_cutoff:])

    train_list = []
    val_list = []

    for key, patches in grouped_patches.items():
        if key in train_keys:
            train_list.extend(patches)
        else:
            val_list.extend(patches)

    train_txt = os.path.join(output_dir, "autosplit_train.txt")
    val_txt = os.path.join(output_dir, "autosplit_val.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_list))

    with open(val_txt, "w") as f:
        f.write("\n".join(val_list))

    print(f"Auto-split completed: {len(train_list)} train, {len(val_list)} val")
    return train_txt, val_txt


def prepare_detection_dataset(config: dict):
    random.seed(config["seed"])

    setup_directories(config)

    dataset_path = os.path.join(config["source"], "dataset.json")
    dataset = json.load(open(dataset_path))

    for image_name, annotations in tqdm(dataset.items(), desc="Processing images"):
        process_image(image_name, annotations, config)

    if(config["image_move"]):
        remove_background_images(config["destination"], config["background_removal"])

    autosplit_detect(
        image_dir=os.path.join(config["destination"], "images"),
        output_dir=config["destination"],
        train_ratio=0.8,
    )

    config_path = os.path.join(config["destination"], "convertation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)




def load_split_list(split_path):
    """Load image paths from a split file."""
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def save_crop(image, box_coords, dest_dir, class_label, base_name, crop_index):
    """
    Crop the image using box_coords and save it in a folder named after the class.
    """
    crop = image.crop(box_coords)
    target_folder = os.path.join(dest_dir, str(class_label))
    os.makedirs(target_folder, exist_ok=True)
    save_path = os.path.join(target_folder, f"{base_name}_{crop_index}.jpg")
    crop.save(save_path)

def process_detection_crops(image, label_path, dest_dir, offset):
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
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = parts[0]
        # YOLO normalized values: x_center, y_center, width, height
        x_center, y_center, box_width, box_height = map(float, parts[1:])
        # Apply offset
        box_width *= (1 + offset)
        box_height *= (1 + offset)
        # Convert normalized coordinates to absolute pixel coordinates
        x1 = int((x_center - box_width / 2) * img_width)
        y1 = int((y_center - box_height / 2) * img_height)
        x2 = int((x_center + box_width / 2) * img_width)
        y2 = int((y_center + box_height / 2) * img_height)
        # Clamp coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        box_abs = (x1, y1, x2, y2)
        det_boxes.append(box_abs)
        class_counts[class_id] += 1

        base_name = os.path.splitext(os.path.basename(label_path))[0]
        save_crop(image, box_abs, dest_dir, class_id, base_name, idx)

    return det_boxes, class_counts

def generate_background_crops(image, det_boxes, dest_dir, base_name, img_size,
                              class_counts, bg_iou_threshold, bg_attempts_multiplier):
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
        candidate_box = (rand_x, rand_y, rand_x + candidate_width, rand_y + candidate_height)

        # Check candidate IoU with each detection box.
        max_iou = max(compute_iou(candidate_box, det_box) for det_box in det_boxes)
        if max_iou < bg_iou_threshold:
            save_crop(image, candidate_box, dest_dir, "background", base_name, f"bg_{bg_saved}")
            bg_saved += 1
            
    if desired_bg > 0 and bg_saved < desired_bg:
        print(f"Image {base_name}: generated only {bg_saved} background crops (desired {desired_bg}).")

# --- Main Function --- #

def prepare_classification_dataset(
    config,
  
):
    """
    Converts a YOLO detection dataset into a classification dataset:
      - Processes detection crops with an optional offset.
      - Generates background crops if their maximum IoU with any detection box is below a threshold.
      
    The dataset is created for each split (e.g. "train", "val") as specified in split_files.
    
    Args:
        source_dir (str): Directory containing "images" and "labels" subfolders.
        output_dir (str): Target directory where classification dataset will be built.
        split_files (dict): Dictionary with split names and corresponding text files.
        offset (float): Fractional expansion of detection boxes.
        bg_iou_threshold (float): IoU threshold for accepting a candidate background crop.
        bg_attempts_multiplier (int): Multiplier for the maximum number of background candidate attempts.
    """
    
    offset: float = config["classify_offset"]
    split_files = {
        "train": os.path.join(config["destination"], "autosplit_train.txt"),
        "val": os.path.join(config["destination"], "autosplit_val.txt"),
    }
    source_dir = config["destination"]
    output_dir = os.path.join(config["destination"], "classification")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    bg_iou_threshold: float = config["bg_iou_threshold"]
    bg_attempts_multiplier: int = 10
    
    for split_name, split_file in split_files.items():
        image_paths = load_split_list(split_file)
        print(f"Processing {split_name} split with {len(image_paths)} images.")

        for rel_img_path in tqdm(image_paths, desc=f"Processing {split_name}"):
            image_path = os.path.join(source_dir, rel_img_path)
            # Assume the label file shares the same base name in a "labels" folder.
            label_filename = os.path.splitext(os.path.basename(rel_img_path))[0] + ".txt"
            label_path = os.path.join(source_dir, "labels", label_filename)

            if not os.path.exists(label_path):
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
            det_boxes, class_counts = process_detection_crops(image, label_path, split_dest_dir, offset)

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
                    bg_attempts_multiplier=bg_attempts_multiplier
                )
    print(f"\nClassification dataset created at {output_dir}")
   
