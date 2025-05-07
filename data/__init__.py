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
                
            grouped_patches[origin_image].append(os.path.join(".","images", filename).replace("\\", "/"))

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

    

from .converting_to_yolo import prepare_yolo_dataset
from .detection_to_classification import prepare_classification_dataset
from .patch_yolo import patch_yolo_dataset

__all__ = ["patch_yolo_dataset", "prepare_classification_dataset", "prepare_yolo_dataset",
           "autosplit_detect", "remove_background_images", "setup_directories", "gdownload"
           ]