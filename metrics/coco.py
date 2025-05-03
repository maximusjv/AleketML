# COCO METRICS UTILS
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout
import copy
from ultralytics.engine.results import Results






def stats_dict(stats: np.ndarray):
    """Creates a dictionary of COCO evaluation statistics.
    Args:
        stats: A numpy array containing COCO statistics.
    Returns:
        A dictionary mapping statistic names to their values.
    """
    # According to https://cocodataset.org/#detection-eval
    return {
        "AP@.50:.05:.95": stats[0],
        "AP@0.5": stats[1],
        "AP@0.75": stats[2],
        "AP small": stats[3],
        "AP medium": stats[4],
        "AP large": stats[5],
        "AR max=1": stats[6],
        "AR max=10": stats[7],
        "AR max=100": stats[8],
        "AR small": stats[9],
        "AR medium": stats[10],
        "AR large": stats[11],
    }


class CocoEvaluator:
    """Evaluates object detection predictions using COCO metrics."""

    def __init__(self, gt_dataset):
        """Initializes the CocoEvaluator.
        Args:
            gt_dataset: The ground truth dataset, either a Dataset or a COCO dataset object.
        """
        with redirect_stdout(io.StringIO()):  # Suppress COCO output during creation
            self.coco_gt = COCO()
            self.coco_gt.dataset = gt_dataset
            self.coco_gt.createIndex()
       
        


    def eval(self, predictions: dict[str, Results], names_to_ids: dict[str, int], useCats=True):
        """Evaluates the accumulated predictions.
        Returns:
            A dictionary of COCO evaluation statistics.
        """
        
        coco_dt = []
        for image_name, results in predictions.items():
            if not results:  # Check if prediction is empty
                continue

            boxes = results.boxes.xyxy.copy()
            boxes[:, 2:] -= boxes[:, :2]
            boxes = boxes.tolist()
            scores = results.boxes.conf.tolist()
            labels = results.boxes.cls.tolist()

            coco_dt.extend(
                [
                    {
                        "image_id": names_to_ids[image_name],
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
            
        if not coco_dt:
            print("NO PREDICTIONS")
            return stats_dict(np.zeros(12))
        
        coco_dt = self.coco_gt.loadRes(coco_dt)
        coco = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco.params.useCats = useCats
        coco.params.catIds = [0, 1, 2] if not useCats else [1,2]
        coco.evaluate()
        coco.accumulate()
        coco.summarize()
        return stats_dict(coco.stats)