
import json
import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from utils.data import autosplit_detect, load_yolo_annotations, remove_background_images, setup_directories
from utils.patches import make_patches, crop_patches

def box_area(box: list[int]) -> float: 
    """Calculate area of a bounding box in XYXY format."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

# PHASE 2: Create patched dataset from YOLO dataset
def save_patch_annotations(f, annotations, patch, config, draw_context, image_move: bool) -> bool:
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


def process_image_for_patching(image_name: str, annotations: dict, config: dict):
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
    
    annotations = load_yolo_annotations(config["source"])
    
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