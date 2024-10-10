# Standard Libraries
import csv
import os

# Pytorch
import torch
import torchvision

from aleket_dataset import AleketDataset

if __name__ == '__main__':

    if __name__ == '__main__':
        dataset = AleketDataset("dataset_patched")

        # Initialize dictionaries to store statistics
        class_counts = {}
        size_counts = {'small': 0, 'medium': 0, 'large': 0}

        # Iterate over dataset and collect statistics
        for img, target in dataset:
            areas = ((target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])).tolist()
            for box, label, area in zip(target["boxes"], target["labels"], areas):
                label = label.item()
                label = AleketDataset.NUM_TO_CLASSES[label]
                # Update class counts

                class_counts[label] = class_counts.get(label, 0) + 1
                # Update size counts based on COCO area ranges
                if area <= 32 ** 2:
                    size_counts['small'] += 1
                elif area <= 96 ** 2:
                    size_counts['medium'] += 1
                else:
                    size_counts['large'] += 1

        os.makedirs("statistics/", exist_ok=True)
        # Write statistics to CSV
        with open(os.path.join('statistics','number_of_objects_by_class.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class Name', 'Number of Objects'])
            for class_id, count in class_counts.items():
                writer.writerow([class_id, count])

        with open(os.path.join('statistics','number_of_objects_by_size.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Size', 'Number of Objects'])
            for size, count in size_counts.items():
                writer.writerow([size, count])

        image_ids_to_visualize = [2]

        # Colors for each class
        class_colors = {
            'healthy': 'green',
            'not healthy': 'red'
        }

        # Visualize and save bounding boxes
        for img_id in image_ids_to_visualize:
            img, target = dataset[img_id]
            img_pil = torchvision.transforms.ToPILImage()(img)
            labels = [AleketDataset.NUM_TO_CLASSES[label.item()] for label in target['labels']]
            # Get colors for each bounding box based on class
            colors = [class_colors[label] for label in labels]

            # Draw bounding boxes with class-specific colors
            img_with_boxes = torchvision.utils.draw_bounding_boxes(
                img, target['boxes'], colors=colors, width=5
            )
            img_with_boxes = torchvision.transforms.ToPILImage()(img_with_boxes)
            # Save the image with bounding boxes
            save_path = os.path.join("statistics", f"image_{img_id}_with_boxes.jpeg")
            img_with_boxes.save(save_path)

            print(f"Image with bounding boxes saved to {save_path}")
        print("Dataset statistics saved to 'statistics'")
