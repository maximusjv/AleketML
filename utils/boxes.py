import math
import numpy as np 
import PIL.Image as Image

def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Calculate the area of bounding boxes.

    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format.

    Returns:
        Array of shape (N,) with the area of each box.
    """
    width = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0, a_max=None)
    height = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0, a_max=None)
    return width * height

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between boxes1 and boxes2.
    
    Args:
        boxes1: array of shape (N, 4) with [x1, y1, x2, y2] format
        boxes2: array of shape (M, 4) with [x1, y1, x2, y2] format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expand dimensions to compute IoU matrix
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    
    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / np.maximum(union, 1e-10)

def expand(boxes: np.ndarray, offset: float) -> np.ndarray:
    """
    Expand the bounding boxes by a given offset.
    boxes: array of shape (N, 4) with [x1, y1, x2, y2] format
    """
    # Compute width and height
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # Expand widths and heights
    o_w = w * (1 + offset)
    o_h = h * (1 + offset)

    # Expand patches
    x1 = boxes[:, 0] - o_w
    y1 = boxes[:, 1] - o_h
    x2 = boxes[:, 2] + o_w
    y2 = boxes[:, 3] + o_h

    expanded_patches = np.stack([x1, y1, x2, y2], axis=1)
    return expanded_patches

def clamp(boxes: np.ndarray, clamp: tuple) -> np.ndarray:
    clamped = boxes.copy()
    clamped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], clamp[0], clamp[2])
    clamped[:, [0, 3]] = np.clip(boxes[:, [0, 3]], clamp[1], clamp[3])
    clamped[:, [0, 2]] -= [clamp[0], clamp[0]]
    clamped[:, [1, 3]] -= [clamp[1], clamp[1]]
    return clamped

def vectorized_make_patches(
    width: int, height: int, patch_size: int, overlap: float
) -> tuple[int, int, np.ndarray]:
    overlap_size = int(patch_size * overlap)
    stride = patch_size - overlap_size

    # Generate grid of top-left (xmin, ymin) coordinates
    x_starts = np.arange(0, width, stride)
    y_starts = np.arange(0, height, stride)

    # Clip so patches don't go beyond width/height
    x_starts = x_starts[x_starts + patch_size <= width + stride]
    y_starts = y_starts[y_starts + patch_size <= height + stride]

    # Create grid of (xmin, ymin)
    xv, yv = np.meshgrid(x_starts, y_starts)
    xv = xv.flatten()
    yv = yv.flatten()

    # Compute (xmax, ymax)
    xmax = xv + patch_size
    ymax = yv + patch_size

    # Create list of Patch objects
    patches = np.concatenate((xv, yv, xmax, ymax), axis=0).T

    padded_width = max(xmax)
    padded_height = max(ymax)

    return padded_width, padded_height, patches

def vectorized_crop_patches(
    image: Image.Image | np.ndarray, patches: np.ndarray
) -> list[np.ndarray | Image.Image]:
    """
    Crop patches from an image based on patch coordinates using optimized methods.
    
    Args:
        image: Input image, either PIL Image or numpy array
        patches: Numpy array of shape [N, 4] with each row containing [xmin, ymin, xmax, ymax]
    
    Returns:
        List of cropped patches
    """
    if isinstance(image, Image.Image):
        # For PIL images, we still need to iterate but can convert coordinates directly
        patches_int = patches.astype(int)
        return [image.crop((x1, y1, x2, y2)) for x1, y1, x2, y2 in patches_int]
    else:
        # For numpy arrays, pre-convert to int to avoid repeated conversions
        patches_int = patches.astype(int)
        
        # Create output array of appropriate size to hold all patches
        n_patches = patches.shape[0]
        patch_heights = patches_int[:, 3] - patches_int[:, 1]
        patch_widths = patches_int[:, 2] - patches_int[:, 0]
        
        # Convert single loop to vectorized operation if all patches are the same size
        if np.all(patch_heights == patch_heights[0]) and np.all(patch_widths == patch_widths[0]):
            # If all patches are the same size, we can use a more efficient approach
            h, w = patch_heights[0], patch_widths[0]
            result = np.zeros((n_patches, h, w, image.shape[2]), dtype=image.dtype)
            
            for i in range(n_patches):
                y1, x1, y2, x2 = patches_int[i, 1], patches_int[i, 0], patches_int[i, 3], patches_int[i, 2]
                result[i] = image[y1:y2, x1:x2, :]
                
            return result
        else:
            # If patches have different sizes, we still need to process individually
            return [
                image[y1:y2, x1:x2, :] 
                for x1, y1, x2, y2 in zip(patches_int[:, 0], patches_int[:, 1], 
                                          patches_int[:, 2], patches_int[:, 3])
            ]

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
        self.ymax = int(ymax)

        
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
        o_w, o_h = w * (1 + offset), h * (1 + offset)
        return Patch(x1 - o_w, y1 - o_h, x2 + o_w, y2 + o_h)

def crop_patches(
    image: Image.Image | np.ndarray, patches: list[Patch]
) -> list[np.ndarray | Image.Image]:
    if isinstance(image, Image.Image):
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
    stride = patch_size - overlap_size
    
    ymin = 0

    patches = []
    
    while ymin <= height:
        xmin = 0
        ymax = ymin + patch_size
        while xmin <= width:
            xmax = xmin + patch_size
            patch_box = (xmin, ymin, xmax, ymax)
            patches.append(Patch(*patch_box))
            xmin += stride
            if  xmax >= width:
                break
        ymin += stride
        if  ymax >= height:
            break
            

    return xmax, ymax, patches
