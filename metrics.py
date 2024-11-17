import time
# Third-party Libraries
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset

# Torchision
from torchvision import ops

from aleket_dataset import AleketDataset

# METRICS NAMES
VALIDATION_METRICS = ["AP@0.50:0.95", "AP@.5", "AP@.75",
                      "Recall@.50:.05:.95", "Recall@.5", "Recall@.75",
                      "ACD", "AAD"]

LOSSES_NAMES = ["loss", "loss_classifier", "loss_box_reg", 'loss_objectness', 'loss_rpn_box_reg']

PRIMARY_VALIDATION_METRIC = "AAD"

def prepare_gts(dataset: AleketDataset, indices: list[int]) -> tuple[dict[tuple[int, int], np.ndarray], list, list]:
    """Prepares ground truth data for evaluation.

    Extracts ground truth bounding boxes and labels from the dataset and organizes them by image and category.

    Args:
        dataset (Dataset): The dataset containing images and ground truth annotations.

    Returns:
        tuple[dict, list, list]: A tuple containing:
            - A dictionary mapping (image_id, category_id) to a NumPy array of ground truth bounding boxes.
            - A sorted list of image IDs.
            - A sorted list of category IDs.
    """
    gts = {}
    categories = set()
    image_ids = set()
    annots = dataset.get_annots(indices)
    for target in annots:
        img_id = target["image_id"]
        bbox = np.asarray(target["boxes"])
        labels = np.asarray(target["labels"])

        image_ids.add(img_id)
        categories.update(labels.tolist())

        for label in categories:
            ind = np.where(labels == label)
            if len(bbox[ind]) != 0:
                gts[img_id, label] = bbox[ind]

    return gts, sorted(image_ids), sorted(categories)


def prepare_dts(predictions: dict[int, dict[str, torch.Tensor]]
                ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Prepares detection results for evaluation.

    Organizes detection results by image and category, sorting them by confidence score.

    Args:
        predictions (dict): A dictionary mapping image IDs to prediction dictionaries 
                            containing 'boxes', 'scores', and 'labels'.

    Returns:
        dict: A dictionary mapping (image_id, category_id) to a tuple of:
            - A NumPy array of detected bounding boxes.
            - A NumPy array of corresponding confidence scores.
    """
    dts = dict()
    categories = set()

    for img_id, preds in predictions.items():
        labels = preds["labels"].cpu().numpy()
        bbox = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        categories.update(labels.tolist())

        for cat in categories:

            ind = np.where(labels == cat)
            bbox_filtered = bbox[ind]
            scores_filtered = scores[ind]

            ind = np.argsort(-scores_filtered, kind="mergesort")  # sort detections by score

            if len(bbox_filtered[ind]) != 0:
                dts[img_id, cat] = bbox_filtered[ind], scores_filtered[ind]

    return dts


def match_gts_dts(gts: np.ndarray,
                  dts:  np.ndarray,
                  iou_matrix: np.ndarray,
                  iou_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Matches ground truth and detected objects based on IoU.

    For each detected object, finds the best matching ground truth object 
    based on IoU, considering an IoU threshold.

    Args:
        gts (np.ndarray): A NumPy array of ground truth bounding boxes.
        dts (np.ndarray): A NumPy array of detected bounding boxes.
        iou_matrix (np.ndarray): A NumPy array representing the IoU between 
                                 each pair of ground truth and detected boxes.
        iou_thresh (float): The IoU threshold for considering a match.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A NumPy array of boolean values indicating which ground truth objects were matched.
            - A NumPy array of boolean values indicating which detected objects were matched.
    """

    # assumes that predictions already sorted
    dt_matches = np.zeros(len(dts))
    gt_matches = np.zeros(len(gts))

    for dind, _ in enumerate(dts):
        iou = iou_thresh
        match = -1
        for gind, _ in enumerate(gts):
            # if gt already matched
            if gt_matches[gind] != 0:
                continue
            # continue to next gt unless better match made
            if iou_matrix[dind, gind] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = iou_matrix[dind, gind]
            match = gind

        if match != -1:
            dt_matches[dind] = 1
            gt_matches[match] = 1

    return gt_matches, dt_matches


def pr_eval(gt_matches: np.ndarray, dt_matches: np.ndarray, dt_scores: np.ndarray, recall_thrs: np.ndarray):
    """Calculates precision-recall curve and related metrics.

    Computes the precision-recall curve, precision, and recall for the given 
    ground truth and detection matches.

    Args:
        gt_matches (np.ndarray): A NumPy array of boolean values indicating 
                                 which ground truth objects were matched.
        dt_matches (np.ndarray): A NumPy array of boolean values indicating 
                                 which detected objects were matched.
        dt_scores (np.ndarray): A NumPy array of detection confidence scores.
        recall_thrs (np.ndarray): A NumPy array of recall thresholds.

    Returns:
        dict: A dictionary containing:
            - 'recall': The final recall value.
            - 'precision': The final precision value.
            - 'pr_curve': The precision-recall curve as a NumPy array.
    """
    inds = np.argsort(-dt_scores, kind="mergesort")
    dt_matches = dt_matches[inds]

    tps = np.cumsum(dt_matches, axis=0, dtype=float)
    fps = np.cumsum(np.logical_not(dt_matches), axis=0, dtype=float)

    rc = tps / len(gt_matches)
    pr = tps / (fps + tps + np.spacing(1))

    pr = pr.tolist()
    # Interpolate precision
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    inds = np.searchsorted(rc, recall_thrs, side='left')
    pr_curve = np.zeros(len(recall_thrs)).tolist()

    for ri, pi in enumerate(inds):
        pr_curve[ri] = pr[pi] if pi < len(pr) else 0

    return {
        "recall": rc[-1] if len(rc) > 0 else 0,
        "precision": pr[-1] if len(pr) > 0 else 0,
        "pr_curve": np.array(pr_curve),
    }


def area_relative_diff(gt: np.ndarray, dt: np.ndarray) -> float:
    """Calculates the area relative difference between ground truth and detected boxes.

    Computes the absolute difference between the total area of ground truth boxes 
    and the total area of detected boxes, divided by the mean area.

    Args:
        gt (np.ndarray): A NumPy array of ground truth bounding boxes.
        dt (np.ndarray): A NumPy array of detected bounding boxes.

    Returns:
        float: The area relative difference.
    """
    gt_area = np.sum((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) if len(gt) != 0 else 0
    dt_area = np.sum((dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])) if len(dt) != 0 else 0

    mean = (dt_area + gt_area) / 2.0

    return (dt_area - gt_area) / mean if mean != 0 else 0


def count_relative_diff(gt: np.ndarray, dt: np.ndarray) -> float:
    """Calculates the count relative difference between ground truth and detected boxes.

    Computes the absolute difference between the number of ground truth boxes and 
    the number of detected boxes, divided by the mean count.

    Args:
        gt (np.ndarray): A NumPy array of ground truth bounding boxes.
        dt (np.ndarray): A NumPy array of detected bounding boxes.

    Returns:
        float: The count relative difference.
    """
    gt_count = len(gt)
    dt_count = len(dt)

    mean = (gt_count + dt_count) / 2.0
    return (dt_count - gt_count) / mean if mean != 0 else 0


class Evaluator:
    """
    Evaluator class for calculating COCO-style metrics for object detection, 
    including Average Precision (AP), Average Recall (AR), 
    Average Count Difference (ACD), and Average Area Difference (AAD).

    Args:
        ds (Dataset): The dataset containing images and ground truth annotations.
    """

    def __init__(self, ds: AleketDataset, indices: list[int]):
        (self.gts,
         self.images_id,
         self.categories) = prepare_gts(ds, indices)

        self.recall_thrs = np.linspace(.0, 1.00, 101)
        self.iou_thrs = np.linspace(.50, 0.95, 10).tolist()

        self.eval_res = {}

    def _quantitative_eval(self, dts: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]):
        """Calculates Average Area Difference (AAD) and Average Count Difference (ACD).

        Computes the AAD and ACD between ground truth and detected objects across 
        all categories and images.

        Args:
            dts (dict): A dictionary mapping (image_id, category_id) to a tuple of:
                - A NumPy array of detected bounding boxes.
                - A NumPy array of corresponding confidence scores.

        Returns:
            dict: A dictionary containing:
                - 'AAD': The Average Area Difference.
                - 'ACD': The Average Count Difference.
        """
        I = len(self.images_id)
        K = len(self.categories)

        AD = np.zeros((K, I))  # area difference over gt area
        CD = np.zeros((K, I))  # count difference over gt count

        for c, cat in enumerate(self.categories):
            for i, image_id in enumerate(self.images_id):
                gt = self.gts.get((image_id, cat), np.empty((0, 4)))
                dt, score = dts.get((image_id, cat), (np.empty((0, 4)), np.empty(0)))

                AD[c, i] = area_relative_diff(gt, dt)
                CD[c, i] = count_relative_diff(gt, dt)

        return {
            "AD": AD,
            "CD": CD
        }

    def _pr_eval_by_iou(self, dts: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], iou_thrs: list):
        """Calculates precision-recall curves and related metrics for different IoU thresholds.

        Computes precision-recall curves, precision, and recall for each category and 
        IoU threshold, enabling the calculation of Average Precision (AP).

        Args:
            dts (dict): A dictionary mapping (image_id, category_id) to a tuple of:
                - A NumPy array of detected bounding boxes.
                - A NumPy array of corresponding confidence scores.
            iou_thrs (list): A list of IoU thresholds for evaluation.

        Returns:
            dict: A dictionary containing:
                - 'pr_curve': A NumPy array of shape (T, K, R) representing the 
                              precision-recall curves for each IoU threshold (T), 
                              category (K), and recall threshold (R).
                - 'precision': A NumPy array of shape (T, K) representing the final 
                               precision values for each IoU threshold and category.
                - 'recall': A NumPy array of shape (T, K) representing the final 
                            recall values for each IoU threshold and category.
        """
        T = len(iou_thrs)
        K = len(self.categories)
        R = len(self.recall_thrs)

        pr_curve = np.zeros((T, K, R))
        precision = np.zeros((T, K, ))
        recall = np.zeros((T, K, ))

        for t, iou_thresh in enumerate(iou_thrs):
            for c, cat in enumerate(self.categories):
                gt_matches = []
                dt_matches = []
                dt_scores = []

                for image_id in self.images_id:

                    gt = self.gts.get((image_id, cat), np.empty((0,4)))
                    dt, score = dts.get((image_id, cat), (np.empty((0,4)), np.empty(0)))

                    if len(dt) == 0 and len(gt) == 0:
                        continue
                    
                    gt_match, dt_match = np.zeros(len(gt)), np.zeros(len(dt))
                    if len(gt) != 0 and len(dt) != 0:
                        # Compute IoU
                        iou_matrix = ops.box_iou(
                            torch.as_tensor(dt),
                            torch.as_tensor(gt)).numpy()
                        gt_match, dt_match = match_gts_dts(gt, dt, iou_matrix, iou_thresh)

                    gt_matches.extend(gt_match)
                    dt_matches.extend(dt_match)
                    dt_scores.extend(score)

                if not gt_matches:
                    continue

                gt_matches = np.array(gt_matches)
                dt_matches = np.array(dt_matches)
                dt_scores = np.array(dt_scores)

                pr_res = pr_eval(gt_matches, dt_matches, dt_scores, self.recall_thrs)
                pr_curve[t, c] = pr_res["pr_curve"]
                precision[t, c] = pr_res["precision"]
                recall[t, c] = pr_res["recall"]

        return {
            "pr_curve": pr_curve,
            "precision": precision,
            "recall": recall,
        }


    def eval(self, dts: dict[int, dict[str, torch.Tensor]]):
        """
        Evaluates the detection results and calculates COCO metrics.

        Computes Average Precision (AP), Average Recall (AR), Average Count Difference (ACD), 
        and Average Area Difference (AAD) based on the provided detection results.

        Args:
            dts (dict): A dictionary mapping image IDs to prediction dictionaries 
                        containing 'boxes', 'scores', and 'labels'.

        Returns:
            dict: A dictionary mapping metric names to their calculated values.
        """
        dts = prepare_dts(dts)
        pr_res = self._pr_eval_by_iou(dts, iou_thrs=self.iou_thrs)

        pr_curve = pr_res["pr_curve"]
        precision = pr_res["precision"]
        recall = pr_res["recall"]

        # Compute AP
        mAP = -1
        AP50 = -1    
        AP75 = -1
        AR = -1
        AR50 = -1
        AR75 = -1
           
        pr = pr_curve[pr_curve > -1]
        if len(pr) > 0:
            mAP = np.mean(pr)
            
        pr = pr_curve[0, :, :]
        pr = pr[pr > -1]
        if len(pr) > 0:
            AP50 = np.mean(pr)
            
        pr = pr_curve[5, :, :]
        pr = pr[pr > -1]
        if len(pr) > 0:
            AP75 = np.mean(pr)
            

        # Compute AR
        r = recall[recall > -1]
        if len(pr) > 0:
            AR = np.mean(r)
           
        r = recall[0, :]
        r = r[r > -1]
        if len(r) > 0:
            AR50 = np.mean(r)
        
        r = recall[5, :]
        r = r[r > -1]
        if len(r) > 0:
            AR75 = np.mean(r)
        
        q_results = self._quantitative_eval(dts)

        AD = q_results["AD"]
        CD = q_results["CD"]

        self.eval_res = {
            "pr_curve": pr_curve,
            "recall": recall,
            "precision": precision,
            "quatitative_results": q_results,

        }

        metrics = [mAP, AP50, AP75, AR, AR50, AR75, np.absolute(AD).mean(), np.absolute(CD).mean()]
        return dict(zip(VALIDATION_METRICS,metrics))
