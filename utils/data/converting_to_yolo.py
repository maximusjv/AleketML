# PHASE 1: Create YOLO annotations from original dataset
import json
import os
import random
import shutil
from tqdm import tqdm
from . import setup_directories, autosplit_detect
from PIL import Image
 
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
                x1, y1, x2, y2 = bbox
                
                # Convert to YOLO format (x_center, y_center, width, height) - normalized
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                f.write(f"{cat} {x_center} {y_center} {width} {height}\n")
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
