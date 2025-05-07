
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.boxes import box_iou

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

def load_model(model_path, device):
    """
    Loads a FasterRCNN_ResNet50_FPN_V2 model with the specified number of classes.

    Args:
        model_path (str): Path to the .pth model file.

    Returns:
        FasterRCNN: A loaded FasterRCNN model.
    """
    model = get_default_model(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model 

def predict_torch(patched_images, batch, device, model):
    dts = []
    for b_imgs in torch.split(patched_images, batch):
        b_imgs = b_imgs.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(b_imgs)
            for pred in preds:
                if len(pred["boxes"]) != 0:
                    boxes = pred["boxes"]
                    labels = pred["labels"].unsqueeze(dim=1)
                    scores = pred["scores"].unsqueeze(dim=1)
                else:
                    boxes = torch.empty((0, 4))
                    labels = torch.empty((0, 1))
                    scores = torch.empty((0, 1))
                    
                dts.append(torch.cat((boxes, labels, scores), dim=1))
                
    return dts

class ModelInterface:
    def __init__(self, model_source: str,
                yolo: bool = True):
        if yolo:
            self.model = YOLO(model_source, task="detect")
        else:
            self.model = torch.load(model_source, weights_only=False)
            self.model
        self.yolo = yolo
        
    def predict(
                self,
                source,
                conf,
                iou,
                imgsz,
                max_det,
                device,
                batch,
            ):
        if self.yolo:
            return self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                batch=batch, 
            )
        else: 
            self.model.roi_heads.score_thresh = conf
            self.model.roi_heads.nms_thresh = iou
            self.model.roi_heads.detections_per_img = max_det
            self.model.eval()
            self.model.to(device)
            
            with torch.no_grad():
                preds = predict_torch(source, batch, device, self.model)
                preds = preds
                return preds
            
    
    


def wbf(boxes, scores, iou_threshold):
    """
    Merges object detections based on Weighted Boxes Fusion (WBF).

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 4) representing bounding boxes.
        scores (np.ndarray): A numpy array of shape (N,) representing confidence scores.
        iou_threshold (float): The IoU threshold for merging boxes.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
            - merged_scores: A numpy array of shape (M,) representing merged scores.
    """ 
    # 1. Sort Detections
    indices = np.argsort(-scores)  # Sort by decreasing confidence scores
    boxes = boxes[indices]
    scores = scores[indices]
    
    if iou_threshold >= 1: # no neeed in wbf
        return boxes, scores
    
    merged_boxes, merged_scores, cluster_boxes, cluster_scores = [], [], [], []

    # 2. Iterate through predictions
    for current_box, current_score in zip(boxes, scores):
        found_cluster = False
        # 3. Find cluster
        for i, merged_box in enumerate(merged_boxes):
            # Calculate IoU between current box and merged box
            iou = box_iou(current_box[None, ...], merged_box[None, ...])[0, 0]  
            if iou > iou_threshold: # 4. Cluster Found
                found_cluster = True
                
                # Add current box to the cluster
                cluster_boxes[i].append(current_box)
                cluster_scores[i].append(current_score)

                # Get all boxes and scores in the cluster
                matched_boxes = np.stack(cluster_boxes[i])
                matched_scores = np.stack(cluster_scores[i])

                # Merge boxes using weighted average based on scores
                merged_boxes[i] = (matched_boxes * matched_scores[:, np.newaxis]).sum(axis=0) / matched_scores.sum()  
                merged_scores[i] = matched_scores.mean()  # Average the scores
                break  # Move to the next box
                
        # 5. Cluster not found
        if not found_cluster:
            # If no overlap, add the current box as a new merged box
            merged_boxes.append(current_box)
            merged_scores.append(current_score)

            # Create a new cluster for this box
            cluster_boxes.append([current_box])
            cluster_scores.append([current_score])

    # 6. Return merged boxes, scores, and labels
    return np.stack(merged_boxes), np.stack(merged_scores)

def batched_wbf(boxes, scores, labels, iou_threshold):
    """
    Merges object detections by classes based on Weighted Boxes Fusion (WBF).

    Args:
        boxes (np.ndarray): A numpy array of shape (N, 4) representing bounding boxes.
        scores (np.ndarray): A numpy array of shape (N,) representing confidence scores.
        classes (np.ndarray): A numpy array of shape (N,) representing class IDs.
        iou_threshold (float): The IoU threshold for merging boxes.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - merged_boxes: A numpy array of shape (M, 4) representing merged boxes.
            - merged_scores: A numpy array of shape (M,) representing merged scores.
            - merged_classes: A numpy array of shape (M,) representing merged classes.
    """
    classes = np.unique(labels)

    merged_boxes = []
    merged_scores = []
    merged_labels = []
    
    for class_id in classes:
        keep = np.where(labels == class_id)
        wbf_boxes, wbf_scores = wbf(boxes[keep], scores[keep], iou_threshold)
        merged_boxes.append(wbf_boxes)
        merged_scores.append(wbf_scores)
        merged_labels.append(np.full_like(wbf_scores, class_id))

    merged_boxes = np.concatenate(merged_boxes)
    merged_scores = np.concatenate(merged_scores)
    merged_labels = np.concatenate(merged_labels)

    indices = np.argsort(-merged_scores)
    merged_boxes = merged_boxes[indices]
    merged_scores = merged_scores[indices]
    merged_labels = merged_labels[indices]
    
    return merged_boxes, merged_scores, merged_labels



