from PIL import Image
import numpy as np
import torch
import torchvision.ops.boxes as ops
from ultralytics.engine.results import Results, Boxes

from utils.metrics.utils import box_area

from .Detection import Detector
from .Classification import Classificator
from utils.data.patches import Patch, crop_patches


# Utility function to load an image
def load(image: str | Image.Image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image


def quantify(boxes, classes, class_num):

    areas = np.empty(class_num, dtype=np.float64)
    counts = np.empty(class_num, dtype=np.uint64)

    for cls in range(class_num):
        keep = classes == cls
        areas[cls] = np.sum(box_area(boxes) * keep, dtype=np.float64)
        counts[cls] = keep.sum()

    return {
        "areas": areas,
        "counts": counts,
    }


class Inference:
    def __init__(
        self, detector: Detector, classificator: Classificator, offset: float = 0.5
    ) -> None:

        self.detector = detector
        self.classificator = classificator
        self.offset = offset
        pass

    def detect(self, x: Image.Image) -> Results:
        det_results: Results = self.detector.forward(x)
        det_results.orig_img = None # remove image from results for memory efficiency
        return det_results

    def patch(self, image: Image.Image, boxes: Boxes):
        patches = [Patch(*(box)).expand(self.offset) for box in boxes.xyxy.tolist()]
        patched_images = crop_patches(image, patches)

        return patched_images, patches

    def classify(self, patched_images):
        cls_results = self.classificator.forward(patched_images)
        cls_results = [x.cpu() for x in cls_results]

        classes = np.asarray([cls_result.probs.top1 for cls_result in cls_results])
        confidences = np.asarray(
            [cls_result.probs.top1conf for cls_result in cls_results]
        )

        return classes, confidences

    def merge_detect_and_classification(self, image, boxes, classes, confidences):
        boxes[:, 5] = classes
        expanded_boxes = np.concatenate((boxes, confidences[:, np.newaxis]), axis=1)
        
        results = Results(
            np.asarray(image)[..., ::-1],
            "",
            self.classificator.model.names,
            boxes=boxes,
        )
        results.orig_img = None # remove image from results for memory efficiency

        return results

    def forward(self, x: str | Image.Image) -> dict:
        image = load(x)

        det_results = self.detect(image)
        detections = det_results.boxes
        
        detections_patched, _ = self.patch(image, detections)
        detections_classes, classifications_confidences = self.classify(
            detections_patched
        )

        results = self.merge_detect_and_classification(
            image, detections, detections_classes, classifications_confidences
        )
        return {
            "object_detection": results,
            "quantification": quantify(
                boxes=results.boxes.xyxy,
                classes=results.boxes.cls,
                class_num=len(results.names),
            ),
            "class_names": results.names,
        }
