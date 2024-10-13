# Standard Library
import io
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any

# Third-party Libraries
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# PyTorch
import torch
from sympy.stats.sampling.sample_numpy import numpy
from torch import Tensor
from torch.utils.data import Dataset

# Torchision
from torchvision import ops

from AleketDataset import AleketDataset

# METRICS NAMES
COCO_STATS_NAMES = ["AP@.50:.05:.95", "AP@0.5", "AP@0.75",
                    "AP small", "AP medium", "AP large", "AR max=1",
                    "AR max=10", "AR max=100", "AR small", "AR medium", "AR large"]
LOSSES_NAMES = ["loss", "loss_classifier", "loss_box_reg", 'loss_objectness', 'loss_rpn_box_reg']


# COCO METRICS UTILS
def convert_to_coco(dataset: Dataset):
    """Converts a custom dataset to COCO API format.
    Args:
        dataset: The custom dataset to convert.

    Returns:
        A COCO dataset object.
    """

    coco_api_dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ann_id = 1

    for idx in range(len(dataset)):
        img, targets = dataset[idx]
        img_id = targets["image_id"]

        img_entry = {"id": img_id, "height": img.shape[-2], "width": img.shape[-1]}
        coco_api_dataset["images"].append(img_entry)

        bboxes = targets["boxes"]

        areas = ((bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])).tolist()

        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh (coco format)
        bboxes = bboxes.tolist()

        labels = targets["labels"].tolist()
        iscrowd = [0] * len(labels)

        for i in range(len(labels)):
            ann = {
                "image_id": img_id,
                "bbox": bboxes[i],
                "category_id": labels[i],
                "area": areas[i],
                "iscrowd": iscrowd[i],
                "id": ann_id,
            }
            categories.add(labels[i])
            coco_api_dataset["annotations"].append(ann)
            ann_id += 1

    coco_api_dataset["categories"] = [
        {"id": i} for i in sorted(categories)
    ]  # TODO add names

    with redirect_stdout(io.StringIO()):  # Suppress COCO output during creation
        coco_ds = COCO()
        coco_ds.dataset = coco_api_dataset
        coco_ds.createIndex()
    return coco_ds


class CocoEvaluator:
    """Evaluates object detection predictions using COCO metrics."""

    def __init__(self, gt_dataset):
        """Initializes the CocoEvaluator.
        Args:
            gt_dataset: The ground truth dataset, either a Dataset or a COCO dataset object.
        """
        if isinstance(gt_dataset, Dataset):
            gt_dataset = convert_to_coco(gt_dataset)

        self.coco_gt = gt_dataset
        self.coco_dt = []
        self.img_ids = set()

    def clear_detections(self):
        """Clears the stored detection results."""

        self.coco_dt = []
        self.img_ids = set()

    def append(self, predictions: dict[int, dict]):
        """Appends predictions to the evaluator.
        Args:
            predictions: A dictionary mapping image IDs to prediction dictionaries.
        """
        for image_id, prediction in predictions.items():

            if image_id in self.img_ids:
                raise ValueError(f"Duplicate prediction for image ID: {image_id}")
            self.img_ids.add(image_id)

            boxes = prediction["boxes"].clone()
            boxes[:, 2:] -= boxes[:, :2]
            boxes = boxes.tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            self.coco_dt.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )

    def eval(self):
        """Evaluates the accumulated predictions.
        Returns:
            A dictionary of COCO evaluation statistics.
        """
        stats = np.zeros(12)
        if self.coco_dt:
            with redirect_stdout(io.StringIO()):  # Suppress COCO output during evaluation
                coco_dt = self.coco_gt.loadRes(self.coco_dt)
                coco = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
                coco.evaluate()
                coco.accumulate()
                coco.summarize()
                stats = coco.stats
            self.coco_ev = coco
        return {key: value for key, value in zip(COCO_STATS_NAMES, stats)}


def prepare_gts(dataset: Dataset) -> tuple[dict[tuple[str, int], Tensor], list, list]:
    gts = {}
    categories = set()
    image_ids = set()

    for _, target in dataset:
        img_id = target["image_id"]
        bbox = target["boxes"].cpu()
        labels = target["labels"].cpu()

        image_ids.add(img_id)
        categories.update(labels.tolist())

        for label in categories:
            ind = torch.where(labels == label)
            if len(bbox[ind]) != 0:
                gts[img_id, label] = bbox[ind]

    return gts, sorted(image_ids), sorted(categories)


def prepare_dts(predictions: dict[str, dict[str, Tensor]]) -> dict[tuple[str, int], tuple[Tensor, Tensor]]:
    dts = dict()
    categories = set()

    for img_id, preds in predictions.items():
        labels = preds["labels"].cpu().to(dtype=torch.float64)
        bbox = preds["boxes"].cpu().to(dtype=torch.float64)
        scores = preds["scores"].cpu().to(dtype=torch.float64)

        categories.update(labels.tolist())

        for cat in categories:

            ind = torch.where(labels == cat)
            bbox_filtered = bbox[ind]
            scores_filtered = scores[ind]

            ind = torch.argsort(scores_filtered, descending=True, stable=True)  # sort detections by score

            if len(bbox_filtered[ind]) != 0:
                dts[img_id, cat] = bbox_filtered[ind], scores_filtered[ind]

    return dts


def match_gts_dts(gts: np.ndarray,
                  dts:  np.ndarray,
                  iou_matrix: np.ndarray,
                  iou_thresh: float):

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

    inds = np.argsort(-dt_scores, kind='mergesort')
    dt_scores = dt_scores[inds]
    dt_matches = dt_matches[inds]

    tps = np.cumsum(dt_matches).astype(float)
    fps = np.cumsum(np.logical_not(dt_matches)).astype(float)

    rc = tps / len(gt_matches)
    pr = tps / (fps + tps + np.spacing(1))

    # Interpolate precision
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    inds = np.searchsorted(rc, recall_thrs, side='left')
    q = np.zeros((len(recall_thrs),))

    for ri, pi in enumerate(inds):
        q[ri] = pr[pi] if pi < len(pr) else 0

    return {
        "recall": rc[-1] if len(rc) > 0 else 0,
        "precision": q,
    }




class Evaluator:
    max_dets = 100
    COCO = None
    def __init__(self, ds: Dataset):
        self.gts, self.images_id, self.categories = prepare_gts(ds)
        self.recall_thrs = np.linspace(.0, 1.00, 101)
        self.iou_thrs = np.linspace(.50, 0.95, 10)

    def _pr_eval_by_iou(self, dts: dict[tuple[str, int], tuple[Tensor, Tensor]], iou_thrs: np.ndarray):

        T = len(iou_thrs)
        K = len(self.categories)
        R = len(self.recall_thrs)

        precision = np.zeros((T, R, K))
        recall = np.zeros((T, K, ))

        for t, iou_thresh in enumerate(iou_thrs.tolist()):
            for c, cat in enumerate(self.categories):
                gt_matches = []
                dt_matches = []
                dt_scores = []

                for image_id in self.images_id:

                    gt = self.gts.get((image_id, cat), torch.empty((0,4)))
                    dt, score = dts.get((image_id, cat), (torch.empty((0,4)), torch.empty(0)))


                    if len(dt) == 0 and len(gt) == 0:
                        continue

                    ind = np.argsort(-score, kind="mergesort")
                    dt = dt[ind]
                    score = score[ind]

                    if len(dt) > Evaluator.max_dets:
                        dt = dt[:Evaluator.max_dets]
                        score = score[:Evaluator.max_dets]

                    assert (len(gt) == len(Evaluator.COCO.coco_ev.by_imgs[image_id, 0, cat]["gtIds"]))
                    assert (len(dt) == len(Evaluator.COCO.coco_ev.by_imgs[image_id, 0, cat]["dtIds"]))

                    gt_match, dt_match = np.zeros(len(gt)), np.zeros(len(dt))
                    if len(gt) != 0 and len(dt) != 0:
                        # Compute IoU
                        iou_matrix = ops.box_iou(dt, gt).numpy()
                        assert(iou_matrix.shape == Evaluator.COCO.coco_ev.ious[(image_id, cat)].shape)
                        gt_match, dt_match = match_gts_dts(gt.numpy(), dt.numpy(), iou_matrix, iou_thresh)

                    gt_matches.extend(gt_match)
                    dt_matches.extend(dt_match)
                    dt_scores.extend(score)

                if not gt_matches:
                    continue

                gt_matches = np.array(gt_matches)
                dt_matches = np.array(dt_matches)
                dt_scores = np.array(dt_scores)

                assert (len(gt_matches) == Evaluator.COCO.coco_ev.NPIG[0, c, 2])
                assert (dt_matches.shape == Evaluator.COCO.coco_ev.DTMS[0, c, 2][0].shape)

                pr_res = pr_eval(gt_matches, dt_matches, dt_scores, self.recall_thrs)
                precision[t, :, c] = pr_res["precision"]
                recall[t, c] = pr_res["recall"]

        return {
            "precision": precision,
            "recall": recall,
        }



    def eval(self, dts: dict[str, dict[str, Tensor]]):

        dts = prepare_dts(dts)

        pr_res = self._pr_eval_by_iou(dts, iou_thrs=self.iou_thrs)


        precision = pr_res["precision"]
        recall = pr_res["recall"]

        self.precision = precision
        self.recall = recall

        # Compute AP

        p = precision[precision > -1]
        if len(p) > 0:
            mAP = np.mean(p)
        else:
            mAP = -1

        p = precision[0, :, :]
        p = p[p > -1]
        if len(p) > 0:
            AP50 = np.mean(p)
        else:
            AP50 = -1

        p = precision[5, :, :]
        p = p[p > -1]
        if len(p) > 0:
            AP75 = np.mean(p)
        else:
            AP75 = -1

        # Compute AR
        r = recall[recall > -1]
        if len(p) > 0:
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

        return {
                COCO_STATS_NAMES[0]: mAP,
                COCO_STATS_NAMES[1]: AP50,
                COCO_STATS_NAMES[2]: AP75,
                COCO_STATS_NAMES[8]: AR,
                "R@0.50": AR50,
                "R@0.75": AR75,

        }




