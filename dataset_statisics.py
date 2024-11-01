# Standard Libraries
import csv
from io import BytesIO
import os
from typing import Optional


import PIL
from PIL.Image import Image
from matplotlib import patches, pyplot as plt
import numpy as np

# Pytorch
import torch
from torchvision import ops
from torch.utils.data import Dataset
from torchvision.transforms import v2

from aleket_dataset import AleketDataset


def count_analyze(dataset: AleketDataset,
                  indices: list[int] = None,
                  save_folder: Optional[str] = None) -> tuple[dict, dict]:
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
    

    by_class_count = {}
    
    count_thrs = np.linspace(10, 1000, (1000-10)//10+1)
    area_thrs = np.array([32**2, 96**2]) # according to coco
    iou_thrs = np.linspace(0.1, 0.9, 9)
    
    by_img_count = np.zeros(len(count_thrs) + 1, dtype=np.int32)
    by_area = np.zeros(len(area_thrs) + 1, dtype=np.int32)
    by_iou = np.zeros(len(iou_thrs) + 1, dtype=np.int32)
    
    with torch.no_grad():
        if not indices:
            indices = list(range(len(dataset)))
            
        for target in dataset.get_annots(indices):
            boxes = torch.as_tensor(target["boxes"])
            labels = torch.as_tensor(target["labels"])
            areas = ops.box_area(boxes)
            uq_labels = torch.unique(labels).tolist()
            
            for label in uq_labels:
                label_name = AleketDataset.NUM_TO_CLASSES[label]
                by_class_count[label_name] = by_class_count.get(label_name, 0) + (labels == label).sum().item()
            
            count = len(labels)
            ious = ops.box_iou(boxes, boxes).unsqueeze(0)
            ious = ious[torch.where(ious <= 0.999)] # remove iou of same boxes
            
            count_inds = np.searchsorted(count_thrs, [count], side='left')[0]
            by_img_count[count_inds] += 1
        
            areas_inds = np.searchsorted(area_thrs, areas.numpy(), side='left').tolist()
            for i in areas_inds:
                by_area[i] += 1
                
            ious_inds = np.searchsorted(iou_thrs, ious.numpy(), side='left').tolist()
            for i in ious_inds:
                by_iou[i] += 1

                
    by_iou //= 2
    if save_folder is not None:
        
        os.makedirs(save_folder, exist_ok=True)
        
        with open(os.path.join(save_folder, 'class_historgram.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class Name', 'Number of Objects'])
            for class_id, count in by_class_count.items():
                writer.writerow([class_id, count])
                
        with open(os.path.join(save_folder, 'area_histogram.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Area Range', 'Number of Objects']) 
            for i, count in enumerate(by_area): 
                if i == 0:
                    writer.writerow(['small', count])  # First bin is 'small'
                elif i == 1:
                    writer.writerow(['medium', count])  # Second bin is 'medium'
                elif i == 2:
                    writer.writerow(['large', count])  # Third bin is 'large'

        with open(os.path.join(save_folder, 'ious_histogram.csv'), 'w', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['IoU Range', 'Number of intersections'])  
            for i, count in enumerate(by_iou):  
                writer.writerow([f'{iou_thrs[i-1] if i > 0 else 0:.2f}-{iou_thrs[i] if i < len(iou_thrs) else 1:.2f}', count])
                
        with open(os.path.join(save_folder, 'count_histogram.csv'), 'w', newline='') as csvfile:  # New file for by_img_count
            writer = csv.writer(csvfile)
            writer.writerow(['Objects Count Range', 'Number of Images'])
            for i, count in enumerate(by_img_count): 
                writer.writerow([f'{count_thrs[i-1] if i > 0 else 0:.0f}-{round(count_thrs[i]) if i < len(count_thrs) else "inf"}', count]) 

    return by_class_count, by_img_count, by_area, by_iou


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
