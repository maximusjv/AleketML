import torch
from ultralytics import YOLO as mYOLO
import torch
from torchvision.transforms import v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.checkpoints import get_default_model


class Model:
    def predict(self, 
                x: torch.Tensor, 
                score_thresh: float,
                nms_iou_thresh: float,
                max_det: int,):
        raise NotImplementedError()

    
        
class YOLO(Model):
    def __init__(self, 
                 source):
        super().__init__()
        self.model = mYOLO(
            source,
            task="detect",
            verbose=False,
        )
        
    @torch.no_grad()    
    def predict(self, 
                x: torch.Tensor, 
                score_thresh: float,
                nms_iou_thresh: float,
                max_det: int,):
        batch = x.shape[0]
        imgsz = x.shape[-1]
        self.model.to(x.device)
        res = self.model.predict(
            x,
            conf=score_thresh,
            iou=nms_iou_thresh,
            imgsz=imgsz,
            max_det=max_det,
            batch=batch,
            verbose=False,
        )
        return [{
            "boxes": r.boxes.xyxy.clone(),
            "scores": r.boxes.conf.clone(),
            "labels": r.boxes.cls.clone(),
        } for r in res]
        
        
class FasterRCNN_ResNet50_FPN_v2(Model):
    def __init__(self,
                 source):
        self.model = get_default_model("cpu", 0)
        self.model.load_state_dict(torch.load(source))
        
    @torch.no_grad()
    def predict(self,
                x: torch.Tensor,
                score_thresh: float,
                nms_iou_thresh: float,
                max_det: int,):
        
        
        self.model.roi_heads.score_thresh = score_thresh
        self.model.roi_heads.nms_thresh = nms_iou_thresh 
        self.model.roi_heads.detections_per_img = max_det
        self.model.eval()
        self.model.to(x.device)
        

        preds = self.model(x)
        return preds