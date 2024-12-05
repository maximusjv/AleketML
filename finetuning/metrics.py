
from torchvision import ops
import torch
from utils.consts import VALIDATION_METRICS

@torch.no_grad()
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
    iou_matrix = ops.box_iou(dts, gts)

    # Initialize matches
    dt_matches = torch.zeros(len(dts))
    gt_matches = torch.zeros(len(gts))

    for dind, _ in enumerate(dts):
        iou = iou_thresh
        match = -1
        for gind, _ in enumerate(gts):
            # If GT already matched
            if gt_matches[gind] != 0:
                continue
            # Continue to next GT unless better match made
            if iou_matrix[dind, gind] < iou:
                continue
            # If match successful and best so far, store appropriately
            iou = iou_matrix[dind, gind]
            match = gind

        if match != -1:
            dt_matches[dind] = 1
            gt_matches[match] = 1

    return gt_matches, dt_matches


@torch.no_grad()
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
        labels = torch.as_tensor(target["labels"]).cpu()
        bbox = torch.as_tensor(target["boxes"]).cpu()
   

        image_ids.add(img_id)
        categories.update(labels.tolist())

        for label in categories:
            ind = torch.where(labels == label)
            if len(bbox[ind]) != 0:
                gts[img_id, label] = bbox[ind]

    return gts, sorted(image_ids), sorted(categories)

@torch.no_grad()
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
        labels = torch.as_tensor(preds["labels"]).cpu()
        bbox = torch.as_tensor(preds["boxes"]).cpu()
        scores = torch.as_tensor(preds["scores"]).cpu()

        categories.update(labels.tolist())

        for cat in categories:

            ind = torch.where(labels == cat)
            bbox_filtered = bbox[ind]
            scores_filtered = scores[ind]

            ind = torch.argsort(
                scores_filtered, descending=True
            )  

            if len(bbox_filtered[ind]) != 0:
                dts[img_id, cat] = bbox_filtered[ind], scores_filtered[ind]

    return dts

@torch.no_grad()
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
    inds = torch.argsort(dt_scores,descending=True)
    dt_matches = dt_matches[inds]

    tps = torch.cumsum(dt_matches, axis=0, dtype=float)
    fps = torch.cumsum(torch.logical_not(dt_matches), axis=0, dtype=float)

    rc = tps / len(gt_matches)
    pr = tps / (fps + tps + 1e-7)

    pr_interpolated = pr.tolist()
    # Interpolate precision
    for i in range(len(pr_interpolated) - 1, 0, -1):
        if pr_interpolated[i] > pr_interpolated[i - 1]:
            pr_interpolated[i - 1] = pr_interpolated[i]
            
    inds = torch.searchsorted(torch.as_tensor(pr_interpolated), recall_thrs)
    pr_curve = torch.zeros(len(recall_thrs))

    for ri, pi in enumerate(inds):
        pr_curve[ri] = pr_interpolated[pi] if pi < len(pr_interpolated) else 0

    R = rc[-1] if len(rc) > 0 else 0
    P = pr[-1] if len(pr) > 0 else 0
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0
    return {"R": R, "P": P, "F1": F1, "pr_curve": pr_curve}

@torch.no_grad()
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
    gt_area = ops.box_area(gt).sum()
    dt_area = ops.box_area(dt).sum()
    mean = (gt_area + dt_area) / 2.0
    return (dt_area - gt_area) / mean if mean != 0 else 0

@torch.no_grad()
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

        self.recall_thrs = torch.linspace(0.0, 1.00, 101)
        self.iou_thresh = 0.5

        self.eval_res = {}
        
    @torch.no_grad()
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

        AD = torch.zeros((K, I))
        CD = torch.zeros((K, I))

        for c, cat in enumerate(self.categories):
            for i, image_id in enumerate(self.images_id):
                gt = self.gts.get((image_id, cat), torch.empty((0, 4)))
                dt, score = dts.get((image_id, cat), (torch.empty((0, 4)), torch.empty(0)))

                AD[c, i] = area_relative_diff(gt, dt)
                CD[c, i] = count_relative_diff(gt, dt)

        return {"AD": AD, "CD": CD}
    
    @torch.no_grad()
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

        pr_curve = torch.zeros((K, R))
        precision = torch.zeros((K,))
        recall = torch.zeros((K,))
        F1 = torch.zeros((K,))

        for c, cat in enumerate(self.categories):
            gt_matches = []
            dt_matches = []
            dt_scores = []

            for image_id in self.images_id:

                gt = self.gts.get((image_id, cat), torch.empty((0, 4)))
                dt, score = dts.get((image_id, cat), (torch.empty((0, 4)), torch.empty(0)))

                if len(dt) == 0 and len(gt) == 0:
                    continue

                gt_match, dt_match = torch.zeros(len(gt)), torch.zeros(len(dt))
                if len(gt) != 0 and len(dt) != 0:
                    gt_match, dt_match = match_gts_dts(gt, dt, self.iou_thresh)

                gt_matches.extend(gt_match)
                dt_matches.extend(dt_match)
                dt_scores.extend(score)

            gt_matches = torch.as_tensor(gt_matches)
            dt_matches = torch.as_tensor(dt_matches)
            dt_scores = torch.as_tensor(dt_scores)

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
    @torch.no_grad()
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

        AP = torch.mean(pr_curve).item()
        R = torch.mean(recall).item()
        P = torch.mean(precision).item()
        F1 = torch.mean(F1).item()

        self.eval_res = {
            "pr_curve": pr_curve,
            "recall": recall,
            "precision": precision,
            "F1": F1,
            "quantitative_results": q_results,
        }

        metrics = [AP, R, P, F1, torch.abs(AD).mean().item(), torch.abs(CD).mean().item()]
        return dict(zip(VALIDATION_METRICS, metrics))
