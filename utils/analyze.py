import csv
import os

import numpy as np

from utils.box_utils import box_area, box_iou
from utils.consts import NUM_TO_CLASSES


def count_analyze(annots, save_folder=None):
    """
    Analyzes dataset annotations to count class occurrences, object sizes, and IoU distributions.

    Processes annotation dictionaries, computes statistics, and optionally saves results to CSV files.

    Args:
        annots (list[dict]): List of annotation dictionaries with 'boxes' and 'labels' keys.
        save_folder (str, optional): Path to save CSV files with statistics. If None, no files are saved.

    Returns:
        tuple[dict, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - by_class_count (dict): Maps class labels to their counts.
            - by_img_count (np.ndarray): Histogram of images by number of objects.
            - by_area (np.ndarray): Histogram of objects by area size (small, medium, large).
            - by_iou (np.ndarray): Histogram of object pairs by IoU value.
    """

    by_class_count = {}

    count_thrs = np.arange(10, 1000, 10)
    area_thrs = np.array([32**2, 96**2])  # according to coco: small, medium, large
    iou_thrs = np.round(np.arange(0.1, 0.9 + 1e-2, 0.1), 1)

    by_img_count = np.zeros(len(count_thrs) + 1, dtype=np.int32)
    by_area = np.zeros(len(area_thrs) + 1, dtype=np.int32)
    by_iou = np.zeros(len(iou_thrs) + 1, dtype=np.int32)

    for target in annots:
        boxes = np.asarray(target["boxes"])
        labels = np.asarray(target["labels"])
        areas = box_area(boxes)
        uq_labels = np.unique(labels).tolist()

        for label in uq_labels:
            label_name = NUM_TO_CLASSES[label]
            by_class_count[label_name] = (
                by_class_count.get(label_name, 0) + (labels == label).sum().item()
            )

        count = len(labels)
        if len(boxes) > 0:
            ious = box_iou(boxes, boxes)
            ious = ious[np.where(ious <= 0.999)]  # remove iou of same boxes

            count_inds = np.searchsorted(count_thrs, [count], side="left")[0]
            by_img_count[count_inds] += 1

            areas_inds = np.searchsorted(area_thrs, areas, side="left").tolist()
            for i in areas_inds:
                by_area[i] += 1

            ious_inds = np.searchsorted(iou_thrs, ious, side="left").tolist()
            for i in ious_inds:
                by_iou[i] += 1

    by_iou //= 2  # every IoU value is counted twice

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

        with open(
            os.path.join(save_folder, "class_historgram.csv"), "w", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Class Name", "Number of Objects"])
            for class_id, count in by_class_count.items():
                writer.writerow([class_id, count])

        with open(
            os.path.join(save_folder, "area_histogram.csv"), "w", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Area Range", "Number of Objects"])
            for i, count in enumerate(by_area):
                if i == 0:
                    writer.writerow(["small", count])  # First bin is 'small'
                elif i == 1:
                    writer.writerow(["medium", count])  # Second bin is 'medium'
                elif i == 2:
                    writer.writerow(["large", count])  # Third bin is 'large'

        with open(
            os.path.join(save_folder, "ious_histogram.csv"), "w", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["IoU Range", "Number of intersections"])
            for i, count in enumerate(by_iou):
                writer.writerow(
                    [
                        f"{iou_thrs[i - 1] if i > 0 else 0:.2f}-{iou_thrs[i] if i < len(iou_thrs) else 1:.2f}",
                        count,
                    ]
                )

        with open(
            os.path.join(save_folder, "count_histogram.csv"), "w", newline=""
        ) as csvfile:  # New file for by_img_count
            writer = csv.writer(csvfile)
            writer.writerow(["Objects Count Range", "Number of Images"])
            for i, count in enumerate(by_img_count):
                writer.writerow(
                    [
                        f'{count_thrs[i - 1] if i > 0 else 0:.0f}-{round(count_thrs[i]) if i < len(count_thrs) else "inf"}',
                        count,
                    ]
                )

    return by_class_count, by_img_count, by_area, by_iou
