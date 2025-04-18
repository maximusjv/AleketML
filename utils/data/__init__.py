from collections import defaultdict
import os
import random
import shutil
import zipfile
from PIL import Image
import gdown

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

def setup_directories(config: dict):
    """Prepare destination directories based on config."""
    dst = config["destination"]
    if os.path.exists(os.path.join(dst, "labels")):
        shutil.rmtree(os.path.join(dst, "labels"))
    if config.get("image_move", False) and os.path.exists(os.path.join(dst, "images")):
            shutil.rmtree(os.path.join(dst, "images"))

    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)

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
    random.seed(seed)
    
    # Step 1: Group patch images by their origin image
    grouped_patches = defaultdict(list)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpeg")):
            # For patched images
            if "_" in filename:
                origin_image = "_".join(filename.split("_")[:-1])  # Remove the patch number
            # For non-patched images
            else:
                origin_image = os.path.splitext(filename)[0]
                
            grouped_patches[origin_image].append(os.path.join(".","images", filename))

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

def load_split_list(split_path):
    """Load image paths from a split file."""
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_simple_yolo(root_dir: str) -> dict:
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpeg'))]
    
    annotations = {}
    
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        
        # Load image to get dimensions
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        img_width, img_height = image.size
     
        annotations[image_name] = []
        # Only process images that have corresponding label files
        if not os.path.exists(label_file):
            continue
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cat = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Convert YOLO format back to absolute coordinates (x1, y1, x2, y2)
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            x2 = (x_center + width/2) * img_width
                            y2 = (y_center + height/2) * img_height
                            
                            annotations[image_name].append([x1, y1, x2, y2, cat])
                            
    return annotations
    
    
def load_yolo_annotations(root_dir: str) -> dict:
    
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpeg'))]
    
    annotations = {}
       
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        
        # Load image to get dimensions
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        img_width, img_height = image.size
     
        annotations[image_name] = { "category_id": [], "boxes": []}
        
        # Only process images that have corresponding label files
        if not os.path.exists(label_file):
            continue
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cat = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Convert YOLO format back to absolute coordinates (x1, y1, x2, y2)
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            x2 = (x_center + width/2) * img_width
                            y2 = (y_center + height/2) * img_height
                            
                            annotations[image_name]["category_id"].append(cat)
                            annotations[image_name]["boxes"].append([x1, y1, x2, y2])
                            
    return annotations


    

from .converting_to_yolo import prepare_yolo_dataset
from .detection_to_classification import prepare_classification_dataset
from .patch_yolo import patch_yolo_dataset
from .patches import Patch, crop_patches, make_patches

__all__ = ["patch_yolo_dataset", "prepare_classification_dataset", "prepare_yolo_dataset",
           "autosplit_detect", "remove_background_images", "setup_directories", "gdownload",
           "Patch", "crop_patches", "make_patches", "expand_patch"
           ]