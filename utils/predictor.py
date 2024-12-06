import PIL
from PIL.Image import Image

import torch
import torchvision.ops as ops

from utils.patches import make_patches

import torchvision.transforms.v2.functional as F
from PIL.Image import Image
from torch.utils.data import Dataset

def wbf(boxes, scores, labels, iou_threshold):
    """
    Merges object detections based on Weighted Boxes Fusion (WBF).

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

    # 1. Sort Detections
    indices = torch.argsort(scores, descending=True)
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]

    merged_boxes = []
    merged_scores = []
    merged_labels = []

    cluster_boxes = []
    cluster_scores = []

    for current_box, current_score, current_label in zip(boxes, scores, labels):
        found = False

        for i, merged_box in enumerate(merged_boxes):
            iou = ops.box_iou(current_box[torch.newaxis, ...], merged_box[torch.newaxis, ...])[0, 0]
            if iou > iou_threshold:
                found = True

                cluster_boxes[i].append(current_box)
                cluster_scores[i].append(current_score)

                matched_boxes = torch.as_tensor(cluster_boxes[i])
                matched_scores = torch.as_tensor(cluster_scores[i])

                # Merge boxes using weighted average
                merged_boxes[i] = (matched_boxes * matched_scores[:, torch.newaxis]).sum(
                    axis=0
                ) / matched_scores.sum()
                merged_scores[i] = matched_scores.mean()
                break

        if not found:
            merged_boxes.append(current_box)
            merged_scores.append(current_score)
            merged_labels.append(current_label)

            cluster_boxes.append([current_box])
            cluster_scores.append([current_score])

    return torch.as_tensor(merged_boxes), torch.as_tensor(merged_scores), torch.as_tensor(merged_labels)


def batched_wbf(boxes, scores, labels, iou_threshold):
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


    classes = torch.unique(labels)

    merged_boxes = []
    merged_scores = []
    merged_labels = []
    
    for class_id in classes:
        keep = torch.where(labels == class_id)
        mb, ms, ml = wbf(boxes[keep], scores[keep], labels[keep], iou_threshold)
        merged_boxes.append(mb)
        merged_scores.append(ms)
        merged_labels.append(ml)

    merged_boxes = torch.cat(merged_boxes)
    merged_scores = torch.cat(merged_scores)
    merged_labels = torch.concatenate(merged_labels)

    indices = torch.argsort(merged_scores, descending=True)
    merged_boxes = merged_boxes[indices]
    merged_scores = merged_scores[indices]
    merged_labels = merged_labels[indices]
    
    return merged_boxes, merged_scores, merged_labels


@torch.no_grad()
def preprocess(
    size_factor, 
    patch_size,
    patch_overlap,
    image):

    if isinstance(image, str):
        image = PIL.Image.open(image)

    if isinstance(image, Image):
        image = F.to_image(image)

    ht, wd = image.shape[-2:]
    ht = int(ht * size_factor)
    wd = int(wd * size_factor)

    padded_width, padded_height, patches = make_patches(
        wd, ht, patch_size, patch_overlap
    )

    image = F.resize(image, size=[ht, wd])
    image = F.pad(
        image, padding=[0, 0, padded_width - wd, padded_height - ht], fill=0.0
    )
    image = F.to_dtype(image, dtype=torch.float32, scale=True)

    patched_images = torch.stack(
        [F.crop(image, y1, x1, y2 - y1, x2 - x1) for (x1, y1, x2, y2) in patches]
    )

    return patches, patched_images

@torch.no_grad()
def postprocess(
    size_factor,
    patches,
    predictions,
    post_postproces_detections,
    iou_thresh,
    use_merge=True,
):
    """
    Postprocesses object detection predictions.

    This function takes a list of predictions and corresponding patch coordinates, adjusts the bounding boxes
    to the original image coordinates, and applies either WBF or NMS to merge or filter overlapping detections.

    Args:
        size_factor (float): The factor by which the image was resized before patching.
        patches (list[list[int]]): A list of patch coordinates, where each patch is represented by
                                    a list [x1, y1, x2, y2].
        predictions (list[dict[str, torch.Tensor]]): A list of prediction dictionaries, where each dictionary
                                                    contains 'boxes', 'scores', and 'labels' as PyTorch tensors.
        post_postproces_detections (int): The maximum number of detections to keep after post-processing.
        iou_thresh (float): The IoU threshold for merging or filtering boxes.
        use_merge (bool, optional): Whether to use WBF (True) or NMS (False). Defaults to True.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the post-processed 'boxes', 'scores', and 'labels'
                                as NumPy arrays.
    """
    boxes = []
    labels = []
    scores = []

    for prediction, patch in zip(predictions, patches):
        x1, y1, _, _ = patch
        prediction["boxes"][:, [0, 2]] += x1
        prediction["boxes"][:, [1, 3]] += y1
        if len(prediction["boxes"]) != 0:
            boxes.extend(prediction["boxes"])
            labels.extend(prediction["labels"])
            scores.extend(prediction["scores"])

    if boxes:
        labels = torch.stack(labels)
        boxes = torch.stack(boxes)
        scores = torch.stack(scores)

        boxes /= size_factor
        if use_merge:
            boxes, scores, labels = batched_wbf(boxes, scores, labels, iou_thresh)
        else:
            keep = ops.batched_nms(
                boxes,
                scores,
                labels,
                iou_thresh,
            )

        boxes = boxes[:post_postproces_detections]
        scores = scores[:post_postproces_detections]
        labels = labels[:post_postproces_detections]

    else:
        labels = torch.zeros(0, dtype=torch.int64)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros(0, dtype=torch.float32)

    return {"boxes": boxes, "scores": scores, "labels": labels}


class Predictor:

    def __init__(
        self,
        model,
        device,
        image_size_factor=1,
        detections_per_image=500,
        detections_per_patch=100,
        patches_per_batch=4,
        patch_size=1024,
        patch_overlap=0.2,
    ):
        """
        Initialize the Predictor class.

        This class is designed to handle object detection predictions using a FasterRCNN model,
        with support for image patching and batch processing.

        Args:
            model (FasterRCNN): The FasterRCNN model to be used for predictions.
            device (torch.device): The device (CPU/GPU) on which to run the model.
            images_per_batch (int, optional): Number of images to process in each batch. Defaults to 4.
            image_size_factor (float, optional): Factor to resize input images. Defaults to 1.
            detections_per_image (int, optional): Maximum number of detections to return per image. Defaults to 300.
            detections_per_patch (int, optional): Maximum number of detections to return per image patch. Defaults to 100.
            patches_per_batch (int, optional): Number of patches to process in each batch. Defaults to 4.
            patch_size (int, optional): Size of each image patch. Defaults to 1024.
            patch_overlap (float, optional): Overlap between adjacent patches. Defaults to 0.2.
        """
        self.device = device
        self.model = model.to(device)
        self.image_size_factor = image_size_factor
        self.per_image_detections = detections_per_image

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch
        self.patches_per_batch = patches_per_batch

    @torch.no_grad()
    def predict(
        self, image, iou_thresh, score_thresh, use_merge=True
    ):
        """
        Generate predictions for an image or a set of images using the model.

        This method efficiently processes input image, applies the object detection model,
        and post-processes the results to produce predictions.

        Args:
            image (Image | torch.Tensor | str): 
                The input image. Provided as PIL Image, torch Tensor or a file path.
            iou_thresh (float): 
                The Intersection over Union (IoU) threshold used for merging or filtering 
                overlapping detections.
            score_thresh (float): 
                The confidence score threshold used to filter out low-confidence detections.
            use_merge (bool, optional): 
                If True, merge overlapping detections using Weighted Boxes Fusion (WBF). 
                If False, use Non-Maximum Suppression (NMS). Defaults to True.

        Returns:
            dict[str, np.array]: A dictionary where:
                - 'boxes': Bounding boxes of detected objects.
                - 'scores': Confidence scores for each detection.
                - 'labels': Predicted class labels for each detection.
        """

        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = 1 if use_merge else iou_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch
        self.model.eval()
        
        result = {}
        
        predictions = []
        patches, patched_images = preprocess(self.image_size_factor, self.patch_size, self.patch_overlap, image)
        for b_imgs in torch.split(patched_images, self.patches_per_batch):
            b_imgs = b_imgs.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                predictions.extend(self.model(b_imgs))

        predictions = postprocess(
            self.image_size_factor,
            patches,
            predictions,
            self.per_image_detections,
            iou_thresh,
            use_merge=use_merge,
        )

        return predictions
    
    
    @torch.no_grad()
    def get_predictions(
        self, images, iou_thresh, score_thresh, use_merge=True
    ):
        """
        Generate predictions for an image or a set of images using the model.

        This method efficiently processes input images, applies the object detection model,
        and post-processes the results to produce a dictionary of predictions.

        Args:
            images (list[Image | torch.Tensor | str] | Dataset | Image | torch.Tensor | str): 
                The input image(s) to process. Can be a single image or a list of images, 
                provided as PIL Images, torch Tensors, file paths, or a Dataset.
            iou_thresh (float): 
                The Intersection over Union (IoU) threshold used for merging or filtering 
                overlapping detections.
            score_thresh (float): 
                The confidence score threshold used to filter out low-confidence detections.
            use_merge (bool, optional): 
                If True, merge overlapping detections using Weighted Boxes Fusion (WBF). 
                If False, use Non-Maximum Suppression (NMS). Defaults to True.

        Returns:
            dict[int, dict[str, np.array]]: A dictionary where:
                - Keys are image ids
                - Values are dictionaries containing the following numpy arrays:
                    - 'boxes': Bounding boxes of detected objects.
                    - 'scores': Confidence scores for each detection.
                    - 'labels': Predicted class labels for each detection.
            if a single image provided only single prediction returned (not a dictionary)
            
        """

        single = isinstance(images, (torch.Tensor, Image, str))
        images = [images] if single else images
        result = {}
        
        for idx, image in enumerate(images):
            if isinstance(images, Dataset):
                image, target = image
                idx = target["image_id"]
                
            predictions = self.predict(image, iou_thresh, score_thresh, use_merge)
            result[idx] = predictions
        
        return result if not single else next(iter(result.values()), None)
