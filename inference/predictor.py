import PIL
from PIL.Image import Image


import numpy as np
import torch

from inference.models import Model
from utils.boxes import Patch, box_iou, make_patches

import torchvision.transforms.v2.functional as F
from PIL.Image import Image


def wbf(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """
    Merges object detections based on Weighted Boxes Fusion (WBF).

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 4) representing bounding boxes.
        scores (np.ndarray): A numpy array of shape (N,) representing confidence scores.
        iou_threshold (float): The IoU threshold for merging boxes.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
            - merged_scores: A numpy array of shape (M,) representing merged scores.
    """
    # Assumed that detections are sorted
    if iou_threshold >= 1:  # no neeed in wbf
        return boxes, scores

    merged_boxes, merged_scores, cluster_boxes, cluster_scores = [], [], [], []

    # 1. Iterate through predictions
    for current_box, current_score in zip(boxes, scores):
        found_cluster = False
        # 2. Find cluster
        for i, merged_box in enumerate(merged_boxes):
            # Calculate IoU between current box and merged box
            iou = box_iou(current_box[None, ...], merged_box[None, ...])[
                0, 0
            ]
            if iou > iou_threshold:  # 3. Cluster Found
                found_cluster = True

                # Add current box to the cluster
                cluster_boxes[i].append(current_box)
                cluster_scores[i].append(current_score)

                # Get all boxes and scores in the cluster
                matched_boxes = np.stack(cluster_boxes[i])
                matched_scores = np.stack(cluster_scores[i])

                # Merge boxes using weighted average based on scores
                merged_boxes[i] = (matched_boxes * matched_scores[:, None]).sum(
                    axis=0
                ) / matched_scores.sum()
                merged_scores[i] = matched_scores.mean()  # Average the scores
                break  # Move to the next box

        # 4. Cluster not found
        if not found_cluster:
            # If no overlap, add the current box as a new merged box
            merged_boxes.append(current_box)
            merged_scores.append(current_score)

            # Create a new cluster for this box
            cluster_boxes.append([current_box])
            cluster_scores.append([current_score])

    # 5. Return merged boxes, scores, and labels
    return np.stack(merged_boxes), np.stack(merged_scores)


def batched_wbf(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float):
    """
    Merges object detections by classes based on Weighted Boxes Fusion (WBF).

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 4) representing bounding boxes.
        scores (np.ndarray): A numpy array of shape (N,) representing confidence scores.
        classes (np.ndarray): A numpy array of shape (N,) representing class IDs.
        iou_threshold (float): The IoU threshold for merging boxes.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
            - merged_scores: A numpy array of shape (M,) representing merged scores.
            - merged_classes: A numpy array of shape (M,) representing merged classes.
    """
    classes = np.unique(labels)

    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for class_id in classes:
        keep = np.where(labels == class_id)
        wbf_boxes, wbf_scores = wbf(boxes[keep], scores[keep], iou_threshold)
        merged_boxes.append(wbf_boxes)
        merged_scores.append(wbf_scores)
        merged_labels.append(np.full_like(wbf_scores, class_id))

    merged_boxes = np.concatenate(merged_boxes)
    merged_scores = np.concatenate(merged_scores)
    merged_labels = np.concatenate(merged_labels)

    indices = np.argsort(-merged_scores)
    merged_boxes = merged_boxes[indices]
    merged_scores = merged_scores[indices]
    merged_labels = merged_labels[indices]

    return merged_boxes, merged_scores, merged_labels


@torch.no_grad()
def preprocess(size_factor, patch_size, patch_overlap, image):
    """
    Preprocesses an image by resizing, padding, and dividing it into patches.

    This function takes an image, resizes it by a given factor, pads it to ensure
    it can be divided into patches of a specified size with a given overlap,
    and then extracts the patches.

    Args:
        size_factor (float): The factor by which to resize the image.
        patch_size (int): The size of the patches (width and height).
        patch_overlap (float): The overlap between adjacent patches (as a fraction).
        image (str, PIL.Image.Image, or torch.Tensor): The input image.
                                                       Can be a file path,
                                                       a PIL Image, or a PyTorch tensor.

    Returns:
        tuple: A tuple containing:
            - patches (list): A list of bounding boxes (x1, y1, x2, y2) for each patch.
            - patched_images (torch.Tensor): A PyTorch tensor containing the extracted patches.
    """
    # 1. Load image if it's a file path
    if isinstance(image, str):
        image = PIL.Image.open(image)

    # 2. Convert PIL Image to torchvision.Image
    if isinstance(image, Image):
        image = F.to_image(image)

    # 3. Resize the image
    ht, wd = image.shape[-2:]
    ht = int(ht * size_factor)
    wd = int(wd * size_factor)
    image = F.resize(image, size=[ht, wd])

    # 4. Calculate padding and patch coordinates
    padded_width, padded_height, patches = make_patches(
        wd, ht, patch_size, patch_overlap
    )

    # 5. Pad the image
    image = F.pad(
        image, padding=[0, 0, padded_width - wd, padded_height - ht], fill=0.0
    )

    # 6. Convert image to float32 tensor
    image = F.to_dtype(image, dtype=torch.float32, scale=True)

    # 7. Extract patches
    patched_images = torch.stack(
        [F.crop(image, patch.ymin, patch.xmin, patch.height, patch.width) for patch in patches]
    )
    return patches, patched_images


@torch.no_grad()
def merge_patches(size_factor, patches: list[Patch], predictions):
    """
    Merges predictions from image patches into a single prediction for the original image.

    This function takes the predictions generated for individual image patches and combines them
    to create a unified prediction for the whole image. It adjusts the bounding boxes from patch-local
    coordinates to the original image coordinates, and then rescales the boxes based on the
    original image size.

    Args:
        size_factor (float): The factor by which the original image was resized before patching.
        patches (list): A list of bounding boxes (x1, y1, x2, y2) representing the coordinates of each patch
                        in the original image.
        predictions (list): A list of dictionaries, where each dictionary contains the predictions
                            for a single patch. Each dictionary should have the keys 'boxes', 'labels',
                            and 'scores'.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - boxes: A NumPy array of shape (N, 4) representing the merged bounding boxes in the original
                    image coordinates.
            - labels: A NumPy array of shape (N,) representing the corresponding labels for the merged boxes.
            - scores: A NumPy array of shape (N,) representing the corresponding confidence scores for the
                    merged boxes.
    """
    boxes = []
    labels = []
    scores = []

    # Adjust bounding boxes to original image coordinates
    for prediction, patch in zip(predictions, patches):
        prediction["boxes"][:, [0, 2]] += patch.xmin  # Adjust x-coordinates of boxes
        prediction["boxes"][:, [1, 3]] += patch.ymin  # Adjust y-coordinates of boxes
        if len(prediction["boxes"]) != 0:
            boxes.extend(prediction["boxes"].numpy(force=True))
            labels.extend(prediction["labels"].numpy(force=True))
            scores.extend(prediction["scores"].numpy(force=True))

    labels = np.stack(labels)
    boxes = np.stack(boxes)
    scores = np.stack(scores)

    boxes /= size_factor  # Rescale boxes to original image size

    return {"boxes": boxes, "labels": labels, "scores": scores}


@torch.no_grad()
def postprocess(
    predictions,
    pre_wbf_detections,
    score_thresh,
    iou_thresh,
):
    """Post-processes object detection predictions.

    This function performs post-processing on the output of an object detection model. It filters
    detections based on a score threshold, applies Weighted Boxes Fusion (WBF) for non-maximum suppression,
    and selects the top-k detections.

    Args:
        predictions (dict): A dictionary containing the 'boxes', 'scores', and 'labels' predicted by the model.
        pre_wbf_detections (int): The number of top detections to keep before applying WBF.
        score_thresh (float): The minimum score threshold for a detection to be considered.
        iou_thresh (float): The IoU threshold for WBF.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the post-processed 'boxes', 'scores', and 'labels'
                                as NumPy arrays.
    """
    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    # 1. Filter by score
    ind = np.where(scores >= score_thresh)
    labels = labels[ind]
    boxes = boxes[ind]
    scores = scores[ind]

    ind = np.argsort(-scores)
    labels = labels[ind]
    boxes = boxes[ind]
    scores = scores[ind]

    boxes = boxes[:pre_wbf_detections]  # Select top-k detections
    scores = scores[:pre_wbf_detections]
    labels = labels[:pre_wbf_detections]

    # 2. Apply WBF and select top-k detections
    if len(boxes) > 0:
        boxes, scores, labels = batched_wbf(
            boxes, scores, labels, iou_thresh
        )  # Apply WBF by class
    else:
        # Handle the case where no detections were made
        labels = np.zeros(0, dtype=np.int64)
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)

    return {"boxes": boxes, "scores": scores, "labels": labels}


class Predictor:
    """
    Performs object detection predictions using a FasterRCNN model with support for image patching.

    This class handles the entire prediction pipeline, including preprocessing images
    (potentially splitting them into patches), applying the model, and post-processing the results.
    """

    def __init__(
        self,
        model: Model,
        device: torch.device,
        image_size_factor: float = 1,
        pre_wbf_detections: int = 2500,
        detections_per_patch: int = 300,
        patches_per_batch: int = 4,
        patch_size: int = 1024,
        patch_overlap: float = 0.2,
    ):
        """
        Initialize the Predictor class.

        Args:
            model (torch.nn.Module): The Detection model to be used for predictions
                                     (input are Tensor batches of images ouput should be
                                     dictionary with tensors of boxes, scores, and labels).
            device (torch.device): The device (CPU/GPU) on which to run the model.
            image_size_factor (float, optional): Factor to resize input images. Defaults to 1.
            pre_wbf_detections (int, optional): Maximum number of detections to keep before applying
                                                Weighted Boxes Fusion (WBF). Defaults to 2500.
            detections_per_patch (int, optional): Maximum number of detections to return per image patch.
                                                Defaults to 300.
            patches_per_batch (int, optional): Number of patches to process in each batch. Defaults to 4.
            patch_size (int, optional): Size of each image patch. Defaults to 1024.
            patch_overlap (float, optional): Overlap between adjacent patches. Defaults to 0.2.
        """
        self.device = device
        self.model = model
        self.image_size_factor = image_size_factor
        self.pre_wbf_detections = pre_wbf_detections

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch
        self.patches_per_batch = patches_per_batch

    @torch.no_grad()
    def predict(self, image, iou_thresh, score_thresh):
        """
        Generate predictions for an image using the model.

        This method processes the input image, applies the object detection model,
        and post-processes the results to produce predictions.

        Args:
            image (Image.Image | torch.Tensor | str): The input image. Provided as PIL Image,
                                                        torch Tensor or a file path.
            iou_thresh (float): The Intersection over Union (IoU) threshold used for merging or filtering
                                overlapping detections.
            score_thresh (float): The confidence score threshold used to filter out low-confidence detections.

        Returns:
            dict[str, np.array]: A dictionary containing:
                - 'boxes': Bounding boxes of detected objects.
                - 'scores': Confidence scores for each detection.
                - 'labels': Predicted class labels for each detection.
        """

        predictions = []
        patches, patched_images = preprocess(
            self.image_size_factor, self.patch_size, self.patch_overlap, image
        )
        for b_imgs in torch.split(patched_images, self.patches_per_batch):
            b_imgs = b_imgs.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                predictions.extend(self.model.predict(
                    b_imgs,
                    score_thresh=0.05,
                    nms_iou_thresh=0.7,
                    max_det=self.detections_per_patch,
                    ))
        predictions = merge_patches(self.image_size_factor, patches, predictions)

        predictions = postprocess(
            predictions, self.pre_wbf_detections, score_thresh, iou_thresh
        )

        return predictions

    @torch.no_grad()
    def get_predictions(
        self,
        images,
        iou_thresh,
        score_thresh,
    ):
        """
        Generate predictions for a single image or a set of images.

        This method efficiently processes input images, applies the object detection model,
        and post-processes the results to produce a dictionary of predictions.

        Args:
            images (list[Image.Image | torch.Tensor | str] | Dataset | Image.Image | torch.Tensor | str):
                The input image(s) to process. Can be a single image or a list of images,
                provided as PIL Images, torch Tensors, file paths, or a Dataset.
            iou_thresh (float): The Intersection over Union (IoU) threshold used for merging or filtering
                                overlapping detections.
            score_thresh (float): The confidence score threshold used to filter out low-confidence detections.

        Returns:
            dict[int, dict[str, np.array]] | None: A dictionary where:
                - Keys are image ids.
                - Values are dictionaries containing:
                    - 'boxes': Bounding boxes of detected objects.
                    - 'scores': Confidence scores for each detection.
                    - 'labels': Predicted class labels for each detection.
                If a single image is provided, only a single prediction is returned (not a dictionary).
                Returns None if no predictions are generated.
        """

        single = isinstance(images, (torch.Tensor, Image.Image, str))
        images = [images] if single else images
        result = {}

        for idx, image in enumerate(images):
            if isinstance(images, torch.utils.data.Dataset):
                image, target = image
                idx = target["image_id"]

            predictions = self.predict(image, iou_thresh, score_thresh)
            result[idx] = predictions

        return result if not single else next(iter(result.values()), None)
