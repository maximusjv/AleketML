from PIL import Image
import numpy as np
import torch
import torchvision.ops.boxes as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .Detection import Detector
from .Classification import Classificator
from utils.data.patches import Patch, crop_patches

# Utility function to load an image
def load(image: str | Image.Image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image

def quantify(boxes, classes, class_num):
    
    areas = torch.empty(class_num, dtype=torch.float64)
    counts = torch.empty(class_num, dtype=torch.uint64)
    
    for cls in range(class_num):
        keep = classes == cls
        areas[cls] = torch.sum(ops.box_area(boxes) * keep, dtype=torch.float64)
        counts[cls] = keep.sum()
        
        
    return {
        "areas": areas,
        "counts": counts,
    }

class Inference:
    def __init__(self, 
                 detector: Detector,
                 classificator: Classificator,
                 offset: float = 0.5
                 ) -> None:
        
        self.detector = detector
        self.classificator = classificator
        self.offset = offset
        pass
    
    def forward(self, x: str | Image.Image) -> dict:
        image = load(x)
        det_results: Results = self.detector.forward(image).cpu().numpy()
        boxes = det_results.boxes.data
        
        patches = [Patch(*(box)).expand(self.offset) for box in boxes[:, :4].tolist()]
        patched_images = crop_patches(image, patches) 
        
        cls_results = self.classificator.forward(patched_images)
        cls_results = [x.cpu() for x in cls_results]
        
        boxes[:,5] = np.asarray([cls_result.probs.top1 for cls_result in cls_results])
        
        confidences = np.asarray([cls_result.probs.top1conf for cls_result in cls_results])
       # expanded_boxes = np.concatenate((boxes, confidences[:, np.newaxis]), axis=1)
        
        results = Results(
            np.asarray(image)[..., ::-1],
            "",
            self.classificator.model.names,
            boxes=boxes,
        )
        
        results = results.cpu().numpy()
        return {
            "object_detection": results.boxes,
            "quantification": quantify(
                boxes=results.boxes.xyxy,
                classes=results.boxes.cls,
                class_num = len(results.names)
                ),
            "class_names": results.names,
        }
        
        
        