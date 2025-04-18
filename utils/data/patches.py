import math
import numpy as np
from PIL.Image import Image


class Patch:
    """
    A class representing a patch of an image.

    Attributes:
        xmin (int): The x-coordinate of the top-left corner of the patch.
        ymin (int): The y-coordinate of the top-left corner of the patch.
        xmax (int): The x-coordinate of the bottom-right corner of the patch.
        ymax (int): The y-coordinate of the bottom-right corner of the patch.
    """
        
    def __init__(
        self, xmin: int, ymin: int, xmax: int, ymax: int,
    ):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = (ymax)

        
    @property
    def xyxy(self) -> tuple[int,int,int,int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    @property
    def xywh(self) -> tuple[float, float, float, float]:
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        xc = self.xmin + w / 2
        yc = self.ymin + h / 2
        return (xc, yc, w, h)
    
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    def clamp(self, other):
        relative_bbox = (
            max(self.xmin, other.xmin) - self.xmin,
            max(self.ymin, other.ymin) - self.ymin,
            min(self.xmax, other.xmax) - self.xmin,
            min(self.ymax, other.ymax) - self.ymin,
        )
        return Patch(*relative_bbox)
    
    def expand(self, offset: float):
        x1, y1, x2, y2 = self.xyxy
        w, h = x2 - x1, y2 - y1
        o_w, o_h = w * offset, h * offset
        return Patch(x1 - o_w, y1 - o_h, x2 + o_w, y2 + o_h)




def crop_patches(
    image: Image | np.ndarray, patches: list[Patch]
) -> list[np.ndarray | Image]:
    if isinstance(image, Image):
        return [image.crop(patch.xyxy) for patch in patches]
    else:
        return [
            image[patch.ymin : patch.ymax, patch.xmin : patch.xmax, :]
            for patch in patches
        ]

def make_patches(
    width: int, height: int, patch_size: int, overlap: float
) -> tuple[int, int, list[Patch]]:
    """
    Create a grid of overlapping patches for an image.

    This function divides an image into a grid of overlapping patches based on the
    specified dimensions and overlap. It calculates the necessary padding and returns
    the dimensions of the padded image along with the coordinates of each patch.

    Args:
        width (int): The width of the original image.
        height (int): The height of the original image.
        patch_size (int): The size of each square patch.
        overlap (float): The fraction of overlap between adjacent patches (0.0 to 1.0).

    Returns:
        tuple: A tuple containing three elements:
            - padded_width (int): The width of the padded image.
            - padded_height (int): The height of the padded image.
            - patch_boxes (list): A list of tuples, where each tuple contains four integers
              (xmin, ymin, xmax, ymax) representing the coordinates of a patch.

    """
    overlap_size = int(patch_size * overlap)
    no_overlap_size = patch_size - overlap_size

    imgs_per_width = math.ceil(float(width) / no_overlap_size)
    imgs_per_height = math.ceil(float(height) / no_overlap_size)

    padded_height = imgs_per_width * no_overlap_size + overlap_size
    padded_width = imgs_per_width * no_overlap_size + overlap_size

    patches = []

    for row in range(imgs_per_height):
        for col in range(imgs_per_width):
            xmin, ymin = col * no_overlap_size, row * no_overlap_size
            xmax, ymax = xmin + patch_size, ymin + patch_size
            patch_box = (xmin, ymin, xmax, ymax)

            patches.append(Patch(*patch_box))

    return padded_width, padded_height, patches
