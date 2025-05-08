import torch
from ultralytics import YOLO as mYOLO
import torch
from torchvision.transforms import v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_default_model(device, trainable_backbone_layers=3):
    """
    Loads a pretrained Faster R-CNN ResNet-50 FPN model and modifies the classification head
    to accommodate the specified number of classes in dataset (3 - including background).

    Args:
        device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').
        trainable_backbone_layers (int, optional): Number of trainable backbone layers. Defaults to 3.

    Returns:
        FasterRCNN: The Faster R-CNN model with the modified classification head.
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT", trainable_backbone_layers=trainable_backbone_layers
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    return model.to(device)

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
        self.yolo = mYOLO(
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
        self.yolo.to(x.device)
        res = self.yolo.predict(
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
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT"
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
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