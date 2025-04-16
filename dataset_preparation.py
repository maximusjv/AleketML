
import os
from utils.data import prepare_detection_dataset, prepare_classification_dataset

# === CONFIGURATION === #
CONFIG = {
    "source": os.path.normpath("C:/Users/maksi/Documents/dataset_full_images"),
    "destination": os.path.normpath("C:/Users/maksi/Documents/datasets/orobanche_cummana"),
    "image_move": False,
    "patch_size": 1024,
    "patch_overlap": 0.2,
    "crop_tolerance": 0.7,
    "erase_cropped": True,
    "background_removal": 0.8,
    "classify_offset": 0.5,
    "seed": 0,
    "bg_iou_threshold": 0.2
}

if __name__ == "__main__":
    prepare_detection_dataset(CONFIG)
    prepare_classification_dataset(CONFIG)
    