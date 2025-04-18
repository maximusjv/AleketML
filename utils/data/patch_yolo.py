
import json
import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from . import autosplit_detect, load_simple_yolo, setup_directories
from .patches import Patch, make_patches, crop_patches

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
    
    annotations = load_simple_yolo(config["source"])
    
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