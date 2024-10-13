import time
# Third-party Libraries
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset

# Torchision
from torchvision import ops

# METRICS NAMES
VALIDATION_METRICS = ["AP@.50:.05:.95", "AP@0.5", "AP@0.75",
                      "Recall@.50:.05:.95", "Recall@0.5", "Recall@0.75",
                      "ACD", "AAD"]

LOSSES_NAMES = ["loss", "loss_classifier", "loss_box_reg", 'loss_objectness', 'loss_rpn_box_reg']


def prepare_gts(dataset: Dataset) -> tuple[dict[tuple[str, int], np.ndarray], list, list]:
    gts = {}
    categories = set()
    image_ids = set()

    for _, target in dataset:
        img_id = target["image_id"]
        bbox = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        image_ids.add(img_id)
        categories.update(labels.tolist())

        for label in categories:
            ind = np.where(labels == label)
            if len(bbox[ind]) != 0:
                gts[img_id, label] = bbox[ind]

    return gts, sorted(image_ids), sorted(categories)


def prepare_dts(predictions: dict[str, dict[str, torch.Tensor]]
                ) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
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

    gt_area = np.sum((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) if len(gt) != 0 else 0
    dt_area = np.sum((dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])) if len(dt) != 0 else 0

    mean = (dt_area + gt_area) / 2.0

    return abs(gt_area - dt_area) / mean if mean != 0 else 0


def count_relative_diff(gt: np.ndarray, dt: np.ndarray) -> float:
    gt_count = len(gt)
    dt_count = len(dt)

    mean = (gt_count + dt_count) / 2.0
    return abs(gt_count - dt_count) / mean if mean != 0 else 0


class Evaluator:
    max_dets = 100

    def __init__(self, ds: Dataset):
        (self.gts,
         self.images_id,
         self.categories) = prepare_gts(ds)

        self.recall_thrs = np.linspace(.0, 1.00, 101)
        self.iou_thrs = np.linspace(.50, 0.95, 10).tolist()

        self.eval_res = {}

    def _quantitative_eval(self, dts: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]):

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
            "AAD": AD.mean(),
            "ACD": CD.mean()
        }

    def _pr_eval_by_iou(self, dts: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], iou_thrs: list):

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

                    if len(dt) > Evaluator.max_dets:
                        dt = dt[:Evaluator.max_dets]
                        score = score[:Evaluator.max_dets]

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


    def eval(self, dts: dict[str, dict[str, torch.Tensor]]):
        dts = prepare_dts(dts)
        pr_res = self._pr_eval_by_iou(dts, iou_thrs=self.iou_thrs)

        pr_curve = pr_res["pr_curve"]
        precision = pr_res["precision"]
        recall = pr_res["recall"]

        # Compute AP

        pr = pr_curve[pr_curve > -1]
        if len(pr) > 0:
            mAP = np.mean(pr)
        else:
            mAP = -1

        pr = pr_curve[0, :, :]
        pr = pr[pr > -1]
        if len(pr) > 0:
            AP50 = np.mean(pr)
        else:
            AP50 = -1

        pr = pr_curve[5, :, :]
        pr = pr[pr > -1]
        if len(pr) > 0:
            AP75 = np.mean(pr)
        else:
            AP75 = -1

        # Compute AR

        r = recall[recall > -1]
        if len(pr) > 0:
            AR = np.mean(r)
        else:
            AR = -1

        r = recall[0, :]
        r = r[r > -1]
        if len(r) > 0:
            AR50 = np.mean(r)
        else:
            AR50 = -1

        r = recall[5, :]
        r = r[r > -1]
        if len(r) > 0:
            AR75 = np.mean(r)
        else:
            AR75 = -1


        q_results = self._quantitative_eval(dts)

        AAD = q_results["AAD"]
        ACD = q_results["ACD"]

        self.eval_res = {
            "pr_curve": pr_curve,
            "recall": recall,
            "precision": precision,

        }

        metrics = [mAP, AP50, AP75, AR, AR50, AR75, AAD, ACD]
        return dict(zip(VALIDATION_METRICS,metrics))
