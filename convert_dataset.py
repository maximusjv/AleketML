"""
This script converts a dataset of images and their corresponding annotations from one format to another.

It reads the images and annotations from a source directory,
processes them to create patches of a specified size with a specified overlap,
and saves the patches and their annotations in a destination directory.

The final annotation format is yolo format in which:
[image_name_no_extension].txt file contains the annotations for the corresponding image in the following format:
[class_id, x_center, y_center, width, height]
where x_center, y_center, width, and height are normalized values between 0 and 1.
"""

import json
import os
import shutil

from tqdm import tqdm

from utils.patches import make_patches
from PIL import Image, ImageDraw


SOURCE = os.path.normpath("C:\\Users\\maksi\\Documents\\dataset_full_images")
DESTINATION = os.path.normpath("C:\\Users\\maksi\\Documents\\datasets\\orobanche_cummana")
IMAGE_MOVE = True
PATCH_SIZE = 1024
PATCH_OVERLAP = 0.2
CROP_TOLERANCE = 0.7
ERASE_CROPPED = True  # If True, the cropped area will be erased in the image
BACKGROUND_REMOVAL = 0.8


# Create the destination directory if it doesn't exist
if os.path.exists(DESTINATION):
    if IMAGE_MOVE:
        shutil.rmtree(DESTINATION)
    else: 
        shutil.rmtree(os.path.join(DESTINATION, "labels"))  # Remove only the labels folder

os.makedirs(DESTINATION, exist_ok=True)
os.makedirs(os.path.join(DESTINATION, "images"), exist_ok=True)
os.makedirs(os.path.join(DESTINATION, "labels"), exist_ok=True)

convertation_config = {
    "source": SOURCE,
    "destination": DESTINATION,
    "patch_size": PATCH_SIZE,
    "patch_overlap": PATCH_OVERLAP,
    "crop_tolerance": CROP_TOLERANCE,
    "erase_cropped": ERASE_CROPPED,
    "background_removal": BACKGROUND_REMOVAL,
}

def box_area(box):
    """Calculates the box area

    Args:
        box: The box in XYXY format.

    Returns:
        float: The area of the box.
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

dataset = json.load(open(os.path.join(SOURCE, "dataset.json")))
for image_name, annotations in tqdm(dataset.items()):
    # Create the destination directory for the image and label
    image_name = image_name + ".jpeg"
    original_image_path = os.path.join(SOURCE, "imgs", image_name)
    original_image = Image.open(original_image_path)
    # Create patches from the image and save them to the destination directory
    padded_width, padded_height, patch_boxes = make_patches(original_image.width, original_image.height, PATCH_SIZE, PATCH_OVERLAP)

    # Save the patches and their annotations in the destination directory
    for i, patch in enumerate(patch_boxes):
        patch_name = f"{image_name.replace('.jpeg', '')}_{i}.jpg"
        patch_label = patch_name.replace(".jpg", ".txt")
        
        if IMAGE_MOVE:
            patched_image = patch.crop(original_image)
            # Save the patch image
            patched_image.save(os.path.join(DESTINATION, "images", patch_name), quality=100)
            draw_context = ImageDraw.Draw(patched_image)  # Create a drawing context for the cropped image
        
        # Save the annotations for the patch in YOLO format
        wrote = False
        with open(os.path.join(DESTINATION, "labels", patch_label), "w") as f:
            for cat, bbox  in zip(annotations["category_id"], annotations["boxes"]):
                # Calculate the coordinates of the bounding box in the patch
                relative_bbox = patch.clamp_box(bbox)

                cropped = 1 - box_area(relative_bbox) / box_area(bbox)  # Calculate the fraction of the bounding box that is cropped
                if(cropped > CROP_TOLERANCE):
                    if cropped < 1 and ERASE_CROPPED and IMAGE_MOVE:
                        draw_context.rectangle(relative_bbox, fill="black")
                    continue
            
                x_center = (relative_bbox[0] + relative_bbox[2]) / 2 / PATCH_SIZE
                y_center = (relative_bbox[1] + relative_bbox[3]) / 2 / PATCH_SIZE
                width = (relative_bbox[2] - relative_bbox[0]) / PATCH_SIZE
                height = (relative_bbox[3] - relative_bbox[1]) / PATCH_SIZE
                # Write the annotation in YOLO format
                wrote = True
                f.write(f"{cat} {x_center} {y_center} {width} {height}\n")
                
        if not wrote:
            os.remove(os.path.join(DESTINATION, "labels", patch_label))

import os
import random
import shutil

def remove_percentage_background_images(root_dir: str, removal_percentage=0.8):
    """
    Removes a specified percentage of background images from a dataset.

    Args:
        removal_percentage (float): Percentage of background images to remove (e.g., 0.8 for 80%).
    """
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "label")
    
    image_files = os.listdir(image_dir)
    background_images = []

    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_dir, image_name + ".txt")

        # Check if a corresponding label file exists
        if not os.path.exists(label_file):
            background_images.append(os.path.join(image_dir, image_file))

    num_to_remove = int(len(background_images) * removal_percentage)
    images_to_remove = random.sample(background_images, num_to_remove)

    for image_path in images_to_remove:
        os.remove(image_path)
        #if there is a corresponding image in another directory, that needs to be deleted also.
        label_file_name = os.path.split(image_path)[1].split('.')[0] + ".txt"
        label_file_path = os.path.join(label_dir, label_file_name)
        if os.path.exists(label_file_path):
            os.remove(label_file_path)

    print(f"Removed {num_to_remove} background images.")


remove_percentage_background_images(DESTINATION, BACKGROUND_REMOVAL)
json.dump(convertation_config, open(os.path.join(DESTINATION, "convertation_config.json"), "w"), indent=4)