
import os
from utils.data import prepare_yolo_dataset, prepare_classification_dataset, patch_yolo_dataset

# === CONFIGURATION === #
CONFIG = {
    "source": os.path.normpath("C:/Users/maksi/Documents/dataset_full_images"),
    "destination": os.path.normpath("C:/Users/maksi/Documents/datasets/orobanche_cummana"),
    "seed": 42,                             # Random seed for reproducibility
    "image_move": True,                     # Whether to move images or just create annotations
    "patch_size": 1024,                      # Size of patches
    "patch_overlap": 0.5,                   # Overlap between patches
    "crop_tolerance": 0.5,                  # Maximum allowed cropping ratio
    "erase_cropped": True,                  # Whether to erase cropped objects
    "background_removal": 0.5,              # Percentage of background images to remove
    "classify_offset": 0.5,                 # Offset for classification crops
    "bg_iou_threshold": 0.1,                # IoU threshold for background crops,
    "classes": {1: "healthy", 2: "necrotic"}, # Classes names for classification
}

def main():
    # Example configuration

    # Phase 1: Create YOLO dataset without patching
    print("Phase 1: Creating YOLO dataset from source annotations...")
    prepare_yolo_dataset(CONFIG)
    
    CONFIG["source"] = CONFIG["destination"]
    CONFIG["destination"] = os.path.join(CONFIG["destination"], "patched")
    
    
    # Phase 2: Create patched YOLO dataset
    print("Phase 2: Creating patched YOLO dataset...")
    patch_yolo_dataset(CONFIG)
    
    CONFIG["destination"] = CONFIG["source"] 
    # Phase 3: Create classification dataset from patched YOLO dataset
    print("Phase 3: Creating classification dataset from YOLO dataset...")
    prepare_classification_dataset(CONFIG)
    
    
    print("All processing completed!")


if __name__ == "__main__":
    main()