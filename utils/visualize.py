import numpy as np
from PIL import ImageDraw, ImageFont


def visualize_bboxes(img, bboxes, labels, linewidth=4, fontsize=20, save_path=None):
    """
    Visualizes bounding boxes on an image and optionally saves it.

    Args:
        img (Image.Image): A PIL Image object.
        bboxes (list[list[int]]): A list of bounding boxes, each represented as a list
                                   of integers [xmin, ymin, xmax, ymax].
        labels (list[str]): A list of labels corresponding to the bounding boxes.
        linewidth (int, optional): Width of the bounding box lines. Defaults to 4.
        fontsize (int, optional): Font size for the labels. Defaults to 20.
        save_path (str, optional): Path to save the image. Defaults to None.

    Returns:
        Image.Image: PIL Image with visualized bounding boxes.
    """

    class_colors = {"healthy": "green", "necrosed": "red"}

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default(fontsize)

    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        color = class_colors.get(label, "blue")
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=linewidth)
        draw.text((xmin, ymin - fontsize), label, fill=color, font=font)

    if save_path:
        img.save(save_path)

    return img


def draw_heat_map(name, x_label, y_label, values, ax, x_ticks, y_ticks):
    """
    Draws a heatmap on the given axes.

    Args:
        name (str): Title of the heatmap.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        values (np.ndarray): 2D array of values to represent in the heatmap.
        ax (Axes): Matplotlib Axes object to draw the heatmap on.
        x_ticks (np.ndarray): Array of tick labels for the x-axis.
        y_ticks (np.ndarray): Array of tick labels for the y-axis.
    """
    masked_results = np.ma.masked_where(values == -1, values)
    ax.imshow(masked_results, cmap="viridis", vmin=0, interpolation="nearest")

    X = len(x_ticks)
    Y = len(y_ticks)

    ax.set_title(name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
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
