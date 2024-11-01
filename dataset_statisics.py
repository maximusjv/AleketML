# Standard Libraries
import csv
from io import BytesIO
import os
from typing import Optional

# Pytorch
import PIL
from PIL.Image import Image
from PIL import ImageDraw
from matplotlib import patches, pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2

from aleket_dataset import AleketDataset


def count_analyze(dataset: Dataset, folder_name: Optional[str] = None) -> tuple[dict, dict]:
    """
    Analyzes the dataset to count the occurrences of each class and the number of objects in each size range.

    Args:
        dataset (Dataset): The dataset to analyze.
        folder_name (Optional[str]): The name of the folder to save the statistics to as CSV files. 
                                     If None, the statistics are not saved.

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
                            - The first dictionary maps class labels to their counts.
                            - The second dictionary maps size ranges ('small', 'medium', 'large') to their counts.
    """
    class_counts = {}
    size_counts = {'small': 0, 'medium': 0, 'large': 0}

    for img, target in dataset:
        areas = ((target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])).tolist()
        for box, label, area in zip(target["boxes"], target["labels"], areas):
            label = label.item()
            label = AleketDataset.NUM_TO_CLASSES[label]

            class_counts[label] = class_counts.get(label, 0) + 1

            if area <= 32 ** 2:
                size_counts['small'] += 1
            elif area <= 96 ** 2:
                size_counts['medium'] += 1
            else:
                size_counts['large'] += 1

    if folder_name is not None:
        with open(os.path.join(folder_name, 'number_of_objects_by_class.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class Name', 'Number of Objects'])
            for class_id, count in class_counts.items():
                writer.writerow([class_id, count])

        with open(os.path.join(folder_name, 'number_of_objects_by_size.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Size', 'Number of Objects'])
            for size, count in size_counts.items():
                writer.writerow([size, count])

    return class_counts, size_counts


def visualize_bboxes(img: Image,
                     bboxes: list[list[int]],
                     labels: list[str],
                     linewidth = 3,
                     fontsize = 16,
                     save_path: Optional[str] = None
                     ) -> Image:
    """
    Visualizes bounding boxes on selected images from the dataset and optionally saves them.

    Args:
        img (Image): A PIL Image object.
        bboxes (list[list[int]]): A list of bounding boxes, each represented as a list of integers [xmin, ymin, xmax, ymax].
        labels (list[str]): A list of labels corresponding to the bounding boxes.

    Returns:
        Image: PIL Image with visualized bounding boxes.
    """

    class_colors = {
        'healthy': 'green',
        'not healthy': 'red'
    }


    fig, ax = plt.subplots(1)
    ax.margins(x=0, y=0)
    ax.set_frame_on(False)
    ax.set_axis_off()
    fig.set_size_inches(img.width//100, img.height//100)
    ax.imshow(img)

    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        color = class_colors.get(label, 'blue')
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin - linewidth, ymin - linewidth, label, color=color, fontsize=fontsize)

    fig.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory

    # Create a PIL Image from the BytesIO object
    pil_image = PIL.Image.open(buf)

    return pil_image


def visualize_samples(dataset: Dataset,
                      folder_name: Optional[str] = None,
                      image_ids_to_visualize: Optional[list[int]] = None) -> list:
    """
    Visualizes bounding boxes on selected images from the dataset and optionally saves them.

    Args:
        dataset (Dataset): The dataset containing the images and bounding box annotations.
        folder_name (Optional[str]): The name of the folder to save the visualized images to. 
                                     If None, the images are not saved.
        image_ids_to_visualize (Optional[list[int]]): A list of image IDs to visualize. 
                                                      If None, a default list is used.

    Returns:
        list: A list of visualized images as PIL Image objects.
    """

    if image_ids_to_visualize is None:
        image_ids_to_visualize = [1, 2, 3]


    visualized_images = []

    for img_id in image_ids_to_visualize:
        img, target = dataset[img_id]
        img = v2.functional.to_pil_image(img)

        bboxes = target["boxes"].cpu().tolist()
        labels = [AleketDataset.NUM_TO_CLASSES[label.item()] for label in target['labels']]

        img_with_boxes = visualize_bboxes(img, bboxes, labels)
        visualized_images.append(img_with_boxes)

        if folder_name is not None:
            save_path = os.path.join(folder_name, f"image_{img_id}_with_boxes.jpeg")
            img_with_boxes.save(save_path)

    return visualized_images
