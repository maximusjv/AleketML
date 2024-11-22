import numpy as np
import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader
from tqdm import tqdm

from box_utils import box_iou
from consts import collate_fn
from patcher import Patcher


def merge_detections(boxes, scores, labels, iou_threshold):
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
    indices = np.argsort(-scores)
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
            iou = box_iou(current_box[np.newaxis, ...],
                          merged_box[np.newaxis, ...])
            if iou > iou_threshold:
                found = True

                cluster_boxes[i].append(current_box)
                cluster_scores[i].append(current_score)

                matched_boxes = np.asarray(cluster_boxes[i])
                matched_scores = np.asarray(cluster_scores[i])

                # Merge boxes using weighted average
                merged_boxes[
                    i] = (matched_boxes * matched_scores[:, np.newaxis]
                         ).sum(axis=0) / matched_scores.sum()
                merged_scores[i] = matched_scores.mean()
                break

        if not found:
            merged_boxes.append(current_box)
            merged_scores.append(current_score)
            merged_labels.append(current_label)

            cluster_boxes.append([current_box])
            cluster_scores.append([current_score])

    return np.array(merged_boxes), np.array(merged_scores), np.array(
        merged_labels)


@torch.no_grad()
def postprocess(size_factor,
                patches,
                predictions,
                post_postproces_detections,
                iou_thresh,
                use_merge=True):
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
            boxes.extend(prediction["boxes"].numpy(force=True))
            labels.extend(prediction["labels"].numpy(force=True))
            scores.extend(prediction["scores"].numpy(force=True))

    if boxes:
        labels = np.stack(labels)
        boxes = np.stack(boxes)
        scores = np.stack(scores)

        boxes /= size_factor
        if use_merge:
            boxes, scores, labels = merge_detections(boxes, scores, labels,
                                                    iou_thresh)
        else:
            keep = ops.batched_nms(torch.as_tensor(boxes),
                                    torch.as_tensor(scores),
                                    torch.as_tensor(labels),
                                    iou_thresh).numpy(force=True)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        boxes = boxes[:post_postproces_detections]
        scores = scores[:post_postproces_detections]
        labels = labels[:post_postproces_detections]

    else:
        labels = np.zeros(0, dtype=np.int64)
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)

    return {"boxes": boxes, "scores": scores, "labels": labels}


class Predictor:

    def __init__(self,
                 model,
                 device,
                 images_per_batch=4,
                 image_size_factor=1,
                 detections_per_image=300,
                 detections_per_patch=100,
                 patches_per_batch=4,
                 patch_size=1024,
                 patch_overlap=0.2):
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
        self.images_per_batch = images_per_batch
        self.image_size_factor = image_size_factor
        self.per_image_detections = detections_per_image

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.detections_per_patch = detections_per_patch
        self.patches_per_batch = patches_per_batch

    @torch.no_grad()
    def get_predictions(self,
                        images,
                        iou_thresh,
                        score_thresh,
                        use_merge=True,
                        progress_bar=False):
        """
        Generate predictions for a set of images using the model.

        This method processes the input images, applies the model to detect objects,
        and post-processes the results.

        Args:
            images (list[Image | torch.Tensor | str] | Dataset): The input images to process.
                Can be a list of PIL Images, torch Tensors, file paths, or a Dataset.
            iou_thresh (float): The Intersection over Union threshold for merging
                or filtering overlapping detections.
            score_thresh (float): The confidence score threshold for filtering detections.
            use_merge (bool, optional): If True, merge overlapping detections using WBF.
                If False, use Non-Maximum Suppression (NMS). Defaults to True.
            progress_bar (bool, optional): If True, shows tqdm progress bar. Defaults to False.

        Returns:
            dict[int, dict[str, np.array]]: A dictionary where keys are image indices and
            values are dictionaries containing 'boxes', 'scores', and 'labels' as numpy arrays.
        """
        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = 1 if use_merge else iou_thresh
        self.model.roi_heads.detections_per_img = self.detections_per_patch
        self.model.eval()

        patcher = Patcher(images, self.image_size_factor, self.patch_size,
                          self.patch_overlap)
        dataloader = DataLoader(patcher,
                                batch_size=self.images_per_batch,
                                num_workers=self.images_per_batch,
                                collate_fn=collate_fn)

        result = {}
        pbar = tqdm(total=len(images)) if progress_bar else None

        for batched_patches, batched_images, batched_idxs in dataloader:
            for patches, imgs, idx in zip(batched_patches, batched_images,
                                         batched_idxs):
                predictions = []
                for b_imgs in torch.split(imgs, self.patches_per_batch):
                    b_imgs = b_imgs.to(self.device)
                    with torch.autocast(device_type=self.device.type,
                                        dtype=torch.float16):
                        predictions.extend(self.model(b_imgs))
                predictions = postprocess(self.image_size_factor, patches,
                                          predictions,
                                          self.per_image_detections,
                                          iou_thresh,
                                          use_merge=use_merge)
                result[idx] = predictions
            if pbar:
                pbar.update(len(batched_idxs))
            
        if pbar:
            pbar.close()
        del patcher
        del dataloader
        return result