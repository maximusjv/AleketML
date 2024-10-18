# Standard Libraries
import csv
import os
from typing import Optional

# Pytorch
import torchvision
from torch.utils.data import Dataset

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



def visualize_example_bboxes(dataset: Dataset,
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

    class_colors = {
        'healthy': 'green',
        'not healthy': 'red'
    }

    visualized_images = []

    for img_id in image_ids_to_visualize:
        img, target = dataset[img_id]
        labels = [AleketDataset.NUM_TO_CLASSES[label.item()] for label in target['labels']]

        colors = [class_colors[label] for label in labels]

        img_with_boxes = torchvision.utils.draw_bounding_boxes(
            img, target['boxes'], colors=colors, width=5
        )
        img_with_boxes = torchvision.transforms.ToPILImage()(img_with_boxes)
        visualized_images.append(img_with_boxes)

        if folder_name is not None:
            save_path = os.path.join(folder_name, f"image_{img_id}_with_boxes.jpeg")
            img_with_boxes.save(save_path)

    return visualized_images