import numpy as np

from torch import Tensor

from utils.boxes import box_area, box_iou
from config.consts import VALIDATION_METRICS

def match_gts_dts(gts, dts, iou_thresh):
    """
    Matches ground truth (GT) boxes to detected (DT) boxes based on IoU.

    Args:
        gts (np.ndarray): An array of shape (N, 4) representing ground truth boxes.
        dts (np.ndarray): An array of shape (M, 4) representing detected boxes.
        iou_thresh (float): The IoU threshold used for matching.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                        - gt_matches: A binary array of shape (N,) indicating
                                                        whether each GT box has a match (1) or not (0).
                                        - dt_matches: A binary array of shape (M,) indicating
                                                        whether each DT box has a match (1) or not (0).
    """
    # It is assumed that dts are already sorted by score in descending order
    iou_matrix = box_iou(dts, gts)  # Calculatate IoU of each box pair


    # Initialize matches
    dt_matches = np.zeros(len(dts))
    gt_matches = np.zeros(len(gts))
    
    for dt_index in range(len(dts)):
        iou = iou_thresh  # Init the threshold
        best_match = -1
        for gt_index in range(len(gts)):
            # Ignore if GT already matched
            if gt_matches[gt_index] != 0:
                continue
            # Continue to next GT unless better match made
            if iou_matrix[dt_index, gt_index] < iou:
                continue
            # If match successful and best so far, store appropriately
            iou = iou_matrix[dt_index, gt_index]
            best_match = gt_index

        if best_match != -1:
            dt_matches[dt_index] = 1
            gt_matches[best_match] = 1
            
    return gt_matches, dt_matches


def prepare_gts(annots, use_categories=True):
    """
    Prepares ground truth data for evaluation.

    This function processes a list of ground truth annotations and organizes them into a dictionary
    for efficient lookup during evaluation. It handles potential Tensor inputs and allows for 
    filtering by category.

    Args:
        annots (list[dict]): A list of annotation dictionaries, each containing 'image_id', 'labels', and 'boxes'.
        use_categories (bool, optional): Whether to organize ground truths by category. Defaults to True. 
                                         If False, all objects are treated as a single category.

    Returns:
        tuple[dict, list, list]: A tuple containing:
            - A dictionary mapping (image_id, category_id) to a NumPy array of ground truth bounding boxes.
            - A sorted list of image IDs.
            - A sorted list of category IDs.
    """
    gts = {}  # Initialize a dictionary to store ground truth bounding boxes
    categories = set()  # Initialize a set to store unique category IDs
    image_ids = set()  # Initialize a set to store unique image IDs
    for target in annots:  # Iterate over each annotation in the list
        img_id = target["image_id"]  # Get the image ID

        # Convert bounding boxes to a NumPy array, handling potential Tensor inputs
        bbox = (
            target["boxes"].numpy(force=True)
            if isinstance(target["boxes"], Tensor)
            else np.asarray(target["boxes"])
        )
        # Convert labels to a NumPy array, handling potential Tensor inputs and optional category filtering
        labels = (
            target["labels"].numpy(force=True)
            if isinstance(target["labels"], Tensor)
            else np.asarray(target["labels"])
        ) if use_categories else np.full(len(bbox), 1)  # If use_categories is False, assign all objects to a single category

        image_ids.add(img_id)  # Add the image ID to the set
        categories.update(labels.tolist())  # Add the category IDs to the set

        for label in categories:  # Iterate over each category
            ind = np.where(labels == label)  # Find the indices of bounding boxes corresponding to the current category
            if len(bbox[ind]) != 0:  # If there are any bounding boxes for this category
                gts[img_id, label] = bbox[ind]  # Add the bounding boxes to the dictionary

    return gts, sorted(image_ids), sorted(categories)  # Return the organized ground truths, image IDs, and category IDs


def prepare_dts(predictions, use_categories=True):
    """
    Prepares detection results for evaluation.

    This function processes a dictionary of detection results and organizes them into a dictionary
    for efficient lookup during evaluation. It handles potential Tensor inputs, sorts detections 
    by confidence score, and allows for filtering by category.

    Args:
        predictions (dict): A dictionary mapping image IDs to prediction dictionaries
                            containing 'boxes', 'scores', and 'labels'.
        use_categories (bool, optional): Whether to organize detections by category. Defaults to True.
                                         If False, all objects are treated as a single category.

    Returns:
        dict: A dictionary mapping (image_id, category_id) to a tuple of:
            - A NumPy array of detected bounding boxes.
            - A NumPy array of corresponding confidence scores.
    """
    dts = dict()  # Initialize a dictionary to store detection results
    categories = set()  # Initialize a set to store unique category IDs

    for img_id, preds in predictions.items():  # Iterate over each image ID and its predictions
        # Convert bounding boxes to a NumPy array, handling potential Tensor inputs
        bbox = (
            preds["boxes"].numpy(force=True)
            if isinstance(preds["boxes"], Tensor)
            else np.asarray(preds["boxes"])
        )
        # Convert labels to a NumPy array, handling potential Tensor inputs and optional category filtering
        labels = (
            preds["labels"].numpy(force=True)
            if isinstance(preds["labels"], Tensor)
            else np.asarray(preds["labels"])
        ) if use_categories else np.full(len(bbox), 1)  # If use_categories is False, assign all objects to a single category
        # Convert scores to a NumPy array, handling potential Tensor inputs
        scores = (
            preds["scores"].numpy(force=True)
            if isinstance(preds["scores"], Tensor)
            else np.asarray(preds["scores"])
        )

        categories.update(labels.tolist())  # Add the category IDs to the set

        for cat in categories:  # Iterate over each category
            ind = np.where(labels == cat)  # Find the indices of bounding boxes corresponding to the current category
            bbox_filtered = bbox[ind]  # Filter bounding boxes by category
            scores_filtered = scores[ind]  # Filter scores by category

            # Sort detections by score in descending order
            ind = np.argsort(
                -scores_filtered, kind="mergesort"
            )  

            if len(bbox_filtered[ind]) != 0:  # If there are any bounding boxes for this category
                # Add the sorted bounding boxes and scores to the dictionary
                dts[img_id, cat] = bbox_filtered[ind], scores_filtered[ind]  

    return dts  # Return the organized detection results


def pr_eval(gt_matches, dt_matches, dt_scores, recall_thrs):
    """
    Calculates precision-recall curve and related metrics.

    This function computes the precision-recall curve, precision, recall, and F1 score
    for the given ground truth and detection matches.

    Args:
        gt_matches (np.ndarray): A NumPy array of boolean values indicating
                                    which ground truth objects were matched.
        dt_matches (np.ndarray): A NumPy array of boolean values indicating
                                    which detected objects were matched.
        dt_scores (np.ndarray): A NumPy array of detection confidence scores.
        recall_thrs (np.ndarray): A NumPy array of recall thresholds.

    Returns:
        dict: A dictionary containing:
            - 'R': The final recall value.
            - 'P': The final precision value.
            - 'F1': The F1 score.
            - 'pr_curve': The precision-recall curve as a NumPy array.
    """
    inds = np.argsort(-dt_scores, kind="mergesort")
    dt_matches = dt_matches[inds]

    tps = np.cumsum(dt_matches, axis=0, dtype=float)
    fps = np.cumsum(np.logical_not(dt_matches), axis=0, dtype=float)

    rc = tps / len(gt_matches)
    pr = tps / (fps + tps + np.spacing(1))

    pr_interpolated = pr.tolist()
    # Interpolate precision
    for i in range(len(pr_interpolated) - 1, 0, -1):
        if pr_interpolated[i] > pr_interpolated[i - 1]:
            pr_interpolated[i - 1] = pr_interpolated[i]

    inds = np.searchsorted(rc, recall_thrs, side="left")
    pr_curve = np.zeros(len(recall_thrs)).tolist()

    for ri, pi in enumerate(inds):
        pr_curve[ri] = pr_interpolated[pi] if pi < len(pr_interpolated) else 0

    R = rc[-1] if len(rc) > 0 else 0
    P = pr[-1] if len(pr) > 0 else 0
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0
    return {"R": R, "P": P, "F1": F1, "pr_curve": np.array(pr_curve)}


def area_relative_diff(gt, dt):
    """
    Calculates the relative difference in area between ground truth and detected bounding boxes.

    Args:
        gt (np.ndarray): A NumPy array of ground truth bounding boxes.
        dt (np.ndarray): A NumPy array of detected bounding boxes.

    Returns:
        float: The relative difference in area. A positive value indicates that the detected area
                is larger than the ground truth area, while a negative value indicates the opposite.
    """
    gt_area = box_area(gt).sum()
    dt_area = box_area(dt).sum()
    mean = (gt_area + dt_area) / 2.0
    return (dt_area - gt_area) / mean if mean != 0 else 0


def count_relative_diff(gt, dt):
    """
    Calculates the relative difference in count between ground truth and detected bounding boxes.

    Args:
        gt (np.ndarray): A NumPy array of ground truth bounding boxes.
        dt (np.ndarray): A NumPy array of detected bounding boxes.

    Returns:
        float: The relative difference in count. A positive value indicates that there are more
                detections than ground truths, while a negative value indicates the opposite.
    """
    gt_count = len(gt)
    dt_count = len(dt)

    mean = (gt_count + dt_count) / 2.0
    return (dt_count - gt_count) / mean if mean != 0 else 0


class Evaluator:
    """
    Evaluates object detection models using various metrics.

    This class provides methods for calculating precision-recall curves, average precision, 
    and quantitative metrics like area difference and count difference. It supports evaluating
    models with or without category distinctions.
    """

    def __init__(self, annots, use_categories=True):
        """
        Initializes the Evaluator with ground truth annotations.

        Args:
            annots (list[dict]): A list of annotation dictionaries, each containing 
                                 'image_id', 'labels', and 'boxes'.
            use_categories (bool, optional): Whether to evaluate with category distinctions. 
                                             Defaults to True.
        """
        (self.gts, self.images_id, self.categories) = prepare_gts(
            annots, use_categories
        )
        self.recall_thrs = np.linspace(0.0, 1.00, 101)
        self.iou_thresh = 0.5
        self.use_categories = use_categories
        self.eval_res = {}

    def quantitative_eval(self, dts):
        """
        Calculates quantitative metrics (area difference and count difference).

        Args:
            dts (dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]): 
                A dictionary of detection results. Keys are (image_id, category_id) tuples, 
                values are (detected_boxes, scores) tuples.

        Returns:
            dict[str, np.ndarray]: A dictionary containing 'AD' (area difference) and 
                                    'CD' (count difference) matrices.
        """
        I = len(self.images_id)
        K = len(self.categories)

        AD = np.zeros((K, I))
        CD = np.zeros((K, I))

        for c, cat in enumerate(self.categories):
            for i, image_id in enumerate(self.images_id):
                gt = self.gts.get((image_id, cat), np.empty((0, 4)))
                dt, score = dts.get((image_id, cat), (np.empty((0, 4)), np.empty(0)))

                AD[c, i] = area_relative_diff(gt, dt)
                CD[c, i] = count_relative_diff(gt, dt)

        return {"AD": AD, "CD": CD}

    def pr_eval(self, dts):
        """
        Evaluates precision-recall metrics.

        Args:
            dts (dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]): 
                A dictionary of detection results. Keys are (image_id, category_id) tuples, 
                values are (detected_boxes, scores) tuples.

        Returns:
            dict[str, np.ndarray]: A dictionary containing precision-recall curves ('pr_curve'), 
                                    precision ('precision'), recall ('recall'), and F1 scores ('F1').
        """
        K = len(self.categories)
        R = len(self.recall_thrs)

        pr_curve = np.zeros((K, R))
        precision = np.zeros((K,))
        recall = np.zeros((K,))
        F1 = np.zeros((K,))

        for c, cat in enumerate(self.categories):
            gt_matches = []
            dt_matches = []
            dt_scores = []

            for image_id in self.images_id:
                gt = self.gts.get((image_id, cat), np.empty((0, 4)))
                dt, score = dts.get(
                    (image_id, cat), (np.empty((0, 4)), np.empty(0))
                )

                if len(dt) == 0 and len(gt) == 0:
                    continue

                gt_match, dt_match = np.zeros(len(gt)), np.zeros(len(dt))
                if len(gt) != 0 and len(dt) != 0:
                    gt_match, dt_match = match_gts_dts(gt, dt, self.iou_thresh)

                gt_matches.extend(gt_match)
                dt_matches.extend(dt_match)
                dt_scores.extend(score)

            gt_matches = np.array(gt_matches)
            dt_matches = np.array(dt_matches)
            dt_scores = np.array(dt_scores)

            pr_res = pr_eval(gt_matches, dt_matches, dt_scores, self.recall_thrs)
            pr_curve[c] = pr_res["pr_curve"]
            precision[c] = pr_res["P"]
            recall[c] = pr_res["R"]
            F1[c] = pr_res["F1"]

        return {
            "pr_curve": pr_curve,
            "precision": precision,
            "recall": recall,
            "F1": F1,
        }

    def eval(self, dts):
        """
        Evaluates detection results and returns a dictionary of metrics.

        Args:
            dts (dict[int, dict[str, torch.Tensor]]): 
                A dictionary of detection results. Keys are image IDs, 
                values are dictionaries containing 'boxes', 'scores', and 'labels'.

        Returns:
            dict: A dictionary of evaluation metrics, including AP, Recall, Precision, F1, ACD, and AAD.
        """
        dts = prepare_dts(dts, self.use_categories)
        pr_res = self.pr_eval(dts)

        pr_curve = pr_res["pr_curve"]
        precision = pr_res["precision"]
        recall = pr_res["recall"]
        F1 = pr_res["F1"]

        q_results = self.quantitative_eval(dts)

        AD = q_results["AD"]
        CD = q_results["CD"]

        AP = np.mean(pr_curve)
        R = np.mean(recall)
        P = np.mean(precision)
        F1 = np.mean(F1)

        self.eval_res = {
            "pr_curve": pr_curve,
            "recall": recall,
            "precision": precision,
            "F1": F1,
            "quantitative_results": q_results,
        }

        metrics = [AP, R, P, F1, np.abs(AD).mean(), np.abs(CD).mean()]
        return dict(zip(VALIDATION_METRICS, metrics))