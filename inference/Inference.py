from PIL import Image
import numpy as np
import torch
from ultralytics.engine.results import Results, Boxes

from metrics.utils import box_area

from .Detection import Detector
from .Classification import Classificator
from data.patches import Patch, crop_patches

from typing import List, Tuple, Dict, Union, Optional
# Utility function to load an image
def load_image(image: str | Image.Image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image


def quantify(boxes: torch.Tensor, classes: torch.Tensor, class_num: int) -> Dict:
    """Calculate areas and counts for each class."""
    areas = np.zeros(class_num, dtype=np.float64)
    counts = np.zeros(class_num, dtype=np.uint64)
    
    # Pre-calculate box areas for efficiency
    box_areas = box_area(boxes)
    
    for cls in range(class_num):
        mask = classes == cls
        areas[cls] = np.sum(box_areas[mask])
        counts[cls] = np.sum(mask)
    
    return {"areas": areas, "counts": counts}


class Inference:
    def __init__(
        self, 
        detector: Detector, 
        classificator: Classificator, 
        offset: float = 0.5
    ):
        self.detector = detector
        self.classificator = classificator
        self.offset = offset
        
    @property
    def names(self):
        return self.classificator.model.names
        
    def detect(self, image: Image.Image) -> Results:
        """Run object detection on an image."""
        det_results = self.detector.forward(image)
        return det_results
    
    def patch(self, image: Image.Image, boxes: Boxes) -> Tuple[List[Image.Image], List[Patch]]:
        """Create and crop patches from detected boxes."""
        patches = [Patch(*box).expand(self.offset) for box in boxes.xyxy.tolist()]
        return crop_patches(image, patches), patches
    
    def classify(self, patched_images: List[Image.Image], to_classes, from_classes) -> Tuple[np.ndarray, np.ndarray]:
        """Classify the patched images."""
        if not patched_images:
            return np.array([]), np.array([])
            
        cls_results = self.classificator.forward(patched_images)
        
        # Batch processing of results
        classes = np.array([from_classes[to_classes[result.probs.top1]] for result in cls_results])
        confidences = np.array([result.probs.top1conf for result in cls_results])
        
        return classes, confidences
    
    def merge_detect_and_classification(
        self, 
        image: Image.Image, 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        confidences: np.ndarray
    ) -> Results:
        """Merge detection and classification results."""
        if len(classes) == 0:
            return Results(
                np.asarray(image)[..., ::-1],
                "",
                self.classificator.model.names,
                boxes=boxes,
            )
            
        # Update class information
        keep = classes != 0
        boxes = boxes[keep]
        classes = classes[keep]
        confidences = confidences[keep]
        
        boxes[:, 5] = classes
        boxes[:, 4] = confidences
        
        # Create results object
        results = Results(
            np.asarray(image)[..., ::-1],
            "",
            self.classificator.model.names,
            boxes=boxes
        )
        results.orig_img = None  # Free memory
        
        return results
    
    def forward(self, x: Union[str, Image.Image]) -> Dict:
        """Process an image through detection and classification pipeline."""
        image = load_image(x)
        
        # Detection step
        det_results = self.detect(image)
        detections = det_results.boxes.data.clone()
        
        # Skip classification if no detections
        if len(detections) == 0:
            return {
                "object_detection": det_results,
                "quantification": {
                    "areas": np.zeros(len(det_results.names), dtype=np.float64),
                    "counts": np.zeros(len(det_results.names), dtype=np.uint64)
                },
                "class_names": det_results.names,
            }
        
        # Classification step
        patched_images, _ = self.patch(image, detections)
        classes, confidences = self.classify(patched_images)
        
        # Merge results
        results = self.merge_detect_and_classification(
            image, detections, classes, confidences
        )
        
        # Return comprehensive results
        return {
            "object_detection": results,
            "quantification": quantify(
                boxes=results.boxes.xyxy,
                classes=results.boxes.cls,
                class_num=len(results.names),
            ),
            "class_names": results.names,
        }
