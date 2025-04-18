from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

from inference.Detection import Detector
from inference.Classification import Classificator
from utils.data.patches import Patch, crop_patches

# Utility function to load an image
def load(image: str | Image.Image) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image

def build_inference_module(
    device: str | int | list,
    detection_model_path: str, 
    classification_model_path: str
    ):
    det_model = Detector(device,
                          detection_model_path,
                          single_cls=True)
    cls_model = Classificator(device, 
                               classification_model_path)
    return Inference(det_model,
                     cls_model,
                        )


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
    
    def forward(self, x: str | Image.Image) -> Results:
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
        
        return Results(
            np.asarray(image)[..., ::-1],
            "",
            self.classificator.model.names,
            boxes=boxes,
        )
        
        
        