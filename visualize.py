# Standard Libraries
from typing import Optional

import PIL
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from typing import Optional
# Pytorch

from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes



def visualize_bboxes(img: Image.Image,
                    bboxes: list[list[int]],
                    labels: list[str],
                    linewidth: int = 4,
                    fontsize: int = 20,
                    save_path: Optional[str] = None
                    ) -> Image.Image:
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

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize) 
    except IOError:
        font = ImageFont.load_default(fontsize)

    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        color = class_colors.get(label, 'blue')
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=linewidth)
        draw.text((xmin , ymin-fontsize), label, fill=color, font=font)

    if save_path:
        img.save(save_path)

    return img


def draw_heat_map(
        name: str, x_label: str, y_label: str, values: np.ndarray, ax: Axes, x_ticks: np.ndarray, y_ticks: np.ndarray
):
    masked_results = np.ma.masked_where(values == -1, values)
    ax.imshow(masked_results, cmap="viridis", vmin=0, interpolation="nearest")

    X = len(x_ticks)
    Y = len(y_ticks)

    ax.set_title(name)
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Iou Threshold")
    ax.set_xticks(np.arange(X))
    ax.set_yticks(np.arange(Y))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    max_val = values.max()

    for i in range(Y):
        for j in range(X):
            value = values[i, j]
            color = "black" if value > max_val / 2 else "white"
            text = ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color)
