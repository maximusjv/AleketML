import math

import PIL
import torch
import torchvision.transforms.v2.functional as F
from PIL.Image import Image
from torch.utils.data import Dataset


def make_patches(width, height, patch_size, overlap):
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

    patch_boxes = []

    for row in range(imgs_per_height):
        for col in range(imgs_per_width):
            xmin, ymin = col * no_overlap_size, row * no_overlap_size
            xmax, ymax = xmin + patch_size, ymin + patch_size
            patch_box = (xmin, ymin, xmax, ymax)

            patch_boxes.append(patch_box)

    return padded_width, padded_height, patch_boxes


class Patcher(Dataset):

    def __init__(self, images, size_factor, patch_size, patch_overlap):
        """
        Initialize the Patcher class.

        This constructor sets up the Patcher object with the given parameters for image processing.

        Args:
            images (list[str | Image | torch.Tensor] | Dataset): A list of image paths, PIL Images,
                PyTorch tensors, or a PyTorch Dataset containing the images to be processed.
            size_factor (float): A scaling factor to adjust the size of the input images.
            patch_size (int): The size of each square patch to be extracted from the images.
            patch_overlap (float): The fraction of overlap between adjacent patches (0.0 to 1.0).

        """
        self.images = images
        self.size_factor = size_factor
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    @torch.no_grad()
    def preprocess(self, image, idx):
        """
        Preprocess an input image by resizing, padding, and splitting it into patches.

        This method takes an input image, resizes it according to the size factor,
        pads it to fit the patch grid, and then splits it into overlapping patches.

        Args:
            image (Image | torch.Tensor | str): The input image. Can be a PIL Image,
                PyTorch tensor, or a string path to an image file.
            idx (int): An index or identifier for the image.

        Returns:
            tuple: A tuple containing three elements:
                - patches (list[tuple[int, int, int, int]]): A list of tuples, each
                 containing the coordinates (x1, y1, x2, y2) of a patch.
                - patched_images (Tensor): A tensor containing all the image patches.
                - idx (int): The input index, passed through.

        """
        if isinstance(image, str):
            image = PIL.Image.open(image)

        if isinstance(image, Image):
            image = F.to_image(image)

        ht, wd = image.shape[-2:]
        ht = int(ht * self.size_factor)
        wd = int(wd * self.size_factor)

        padded_width, padded_height, patches = make_patches(
            wd, ht, self.patch_size, self.patch_overlap
        )

        image = F.resize(image, size=[ht, wd])
        image = F.pad(
            image, padding=[0, 0, padded_width - wd, padded_height - ht], fill=0.0
        )
        image = F.to_dtype(image, dtype=torch.float32, scale=True)

        patched_images = torch.stack(
            [F.crop(image, y1, x1, y2 - y1, x2 - x1) for (x1, y1, x2, y2) in patches]
        )

        return patches, patched_images, idx

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, idx):
        """
        Retrieve and preprocess an item from the dataset.

        This method fetches an image from the dataset, preprocesses it, and returns
        the processed data. It handles both cases where self.images is a Dataset
        or a list of images.

        Args:
            idx (int): The index of the item to retrieve from the dataset.

        Returns:
            tuple: A tuple containing three elements:
                - list[tuple[int, int, int, int]]: A list of patch coordinates.
                - Tensor: A tensor containing the preprocessed image patches.
                - int: The image index or ID.
        """
        if isinstance(self.images, Dataset):
            image, target = self.images[idx]
            return self.preprocess(image, target["image_id"])
        return self.preprocess(self.images[idx], idx)
