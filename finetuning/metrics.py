import numpy as np

from torch import Tensor

from utils.box_utils import match_gts_dts, box_area
from utils.consts import VALIDATION_METRICS


def prepare_gts(annots):
    """Prepares ground truth data for evaluation.

    Extracts ground truth bounding boxes and labels from the dataset and organizes them by image and category.

    Args:
        annots (list[dict]): A list of annotation dictionaries, each containing 'image_id', 'labels', and 'boxes'.

    Returns:
        tuple[dict, list, list]: A tuple containing:
            - A dictionary mapping (image_id, category_id) to a NumPy array of ground truth bounding boxes.
            - A sorted list of image IDs.
            - A sorted list of category IDs.
    """
    gts = {}
    categories = set()
    image_ids = set()
    for target in annots:
        img_id = target["image_id"]
        labels = (
            target["labels"].numpy(force=True)
            if isinstance(target["labels"], Tensor)
            else np.asarray(target["labels"])
        )
        bbox = (
            target["boxes"].numpy(force=True)
            if isinstance(target["boxes"], Tensor)
            else np.asarray(target["boxes"])
        )

        image_ids.add(img_id)
        categories.update(labels.tolist())

        for label in categories:
            ind = np.where(labels == label)
            if len(bbox[ind]) != 0:
                gts[img_id, label] = bbox[ind]

    return gts, sorted(image_ids), sorted(categories)


def prepare_dts(predictions):
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
        labels = (
            preds["labels"].numpy(force=True)
            if isinstance(preds["labels"], Tensor)
            else np.asarray(preds["labels"])
        )
        bbox = (
            preds["boxes"].numpy(force=True)
            if isinstance(preds["boxes"], Tensor)
            else np.asarray(preds["boxes"])
        )
        scores = (
            preds["scores"].numpy(force=True)
            if isinstance(preds["scores"], Tensor)
            else np.asarray(preds["scores"])
        )

        categories.update(labels.tolist())

        for cat in categories:

            ind = np.where(labels == cat)
            bbox_filtered = bbox[ind]
            scores_filtered = scores[ind]

            ind = np.argsort(
                -scores_filtered, kind="mergesort"
            )  # sort detections by score

            if len(bbox_filtered[ind]) != 0:
                dts[img_id, cat] = bbox_filtered[ind], scores_filtered[ind]

    return dts


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

    def __init__(self, annots):
        """
        Initializes the Evaluator class with ground truth data, image IDs, and categories.

        Parameters:
            annots (list[dict]): A list of annotation dictionaries, each containing 'image_id', 'labels', and 'boxes'.

        The function prepares ground truth data by calling the prepare_gts function. It initializes
        recall thresholds, IOU threshold, and an empty dictionary for storing evaluation results.
        """
        (self.gts, self.images_id, self.categories) = prepare_gts(annots)

        self.recall_thrs = np.linspace(0.0, 1.00, 101)
        self.iou_thresh = 0.5

        self.eval_res = {}

    def quantitative_eval(self, dts):
        """
        Performs quantitative evaluation of detection results.

        This function calculates the Area Difference (AD) and Count Difference (CD)
        between ground truth and detected bounding boxes for each category and image.

        Args:
            dts (dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]): A dictionary
                where keys are tuples of (image_id, category_id) and values are tuples
                of (detected_boxes, scores). detected_boxes is a numpy array of shape
                (n, 4) representing bounding boxes, and scores is a numpy array of
                shape (n,) representing confidence scores.

        Returns:
            dict[str, np.ndarray]: A dictionary containing:
                - "AD": numpy array of shape (K, I) representing Area Difference,
                        where K is the number of categories and I is the number of images.
                - "CD": numpy array of shape (K, I) representing Count Difference,
                        where K is the number of categories and I is the number of images.
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
        Evaluates precision-recall metrics for object detection results.

        This method calculates precision-recall curves, precision, recall, and F1 scores
        for each category in the dataset based on the provided detection results.

        Args:
            dts (dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]): A dictionary of detection results.
                Keys are tuples of (image_id, category_id), and values are tuples of
                (detected_boxes, confidence_scores).

        Returns:
            dict[str, np.ndarray]: A dictionary containing the following evaluation results:
                - 'pr_curve': Precision-recall curves for each category (shape: [K, R])
                - 'precision': Precision values for each category (shape: [K,])
                - 'recall': Recall values for each category (shape: [K,])
                - 'F1': F1 scores for each category (shape: [K,])
            where K is the number of categories and R is the number of recall thresholds.
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
                dt, score = dts.get((image_id, cat), (np.empty((0, 4)), np.empty(0)))

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
        Evaluates the detection results and calculates various metrics.

        This function processes the detection results, computes precision-recall metrics,
        and calculates quantitative metrics such as Average Precision (AP), Recall, Precision,
        F1 score, Average Count Difference (ACD), and Average Area Difference (AAD).

        Args:
            dts (dict[int, dict[str, torch.Tensor]]): A dictionary mapping image IDs to prediction
                dictionaries. Each prediction dictionary contains 'boxes', 'scores', and 'labels'
                as torch.Tensor objects.

        Returns:
            dict: A dictionary mapping metric names to their calculated values. The metrics include:
                - AP (Average Precision)
                - Recall
                - Precision
                - F1 score
                - ACD (Average Count Difference)
                - AAD (Average Area Difference)

        Note:
            This method also updates the `eval_res` attribute of the class with detailed evaluation results.
        """
        dts = prepare_dts(dts)
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
