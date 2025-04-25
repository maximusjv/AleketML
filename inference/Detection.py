from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils.data import Patch, make_patches, crop_patches

def resize_image(image: Image.Image, size_factor: float) -> Image.Image:
    """Resize image using PIL with proportional scaling."""
    wd, ht = image.size
    return image.resize((int(wd * size_factor), int(ht * size_factor)))

def create_patches(image: Image.Image, patch_size: int, overlap: float) -> tuple[list[Patch], torch.Tensor]:
    """Create image patches and return as batched tensor."""
    wd, ht = image.size
    padded_wd, padded_ht, patches = make_patches(wd, ht, patch_size, overlap)
    padded_img = Image.new("RGB", (padded_wd, padded_ht))
    padded_img.paste(image)
    
    # Vectorized patch extraction
    np_img = np.array(padded_img)
    patch_arrays = [np_img[p.ymin:p.ymax, p.xmin:p.xmax] for p in patches]
    batch = torch.stack([torch.from_numpy(p).permute(2,0,1).float()/255 for p in patch_arrays])
    
    return patches, batch

def merge_predictions(patches: list[Patch], results: list[torch.Tensor], size_factor: float) -> torch.Tensor:
    """Merge patch-based predictions to original image coordinates."""
    merged = []
    for patch, preds in zip(patches, results):
        if preds.shape[0] > 0:
            adjusted = preds.clone()
            adjusted[:, [0,2]] += patch.xmin
            adjusted[:, [1,3]] += patch.ymin
            merged.append(adjusted)
    
    if not merged:
        return torch.empty((0, 6))
    
    merged = torch.cat(merged)
    merged[:, :4] /= size_factor
    return merged

def weighted_box_fusion(
    boxes: torch.Tensor,
    ios_thresh: float,
    pre_limit: int,
    post_limit: int,
    single_cls: bool
) -> torch.Tensor:
    """Optimized weighted box fusion implementation."""
    if boxes.shape[0] == 0 or ios_thresh >= 1:
        return boxes[:post_limit]
    
    boxes = boxes[:pre_limit]
    device, dtype = boxes.device, boxes.dtype
    clusters = []
    
    # Vectorized IO calculation
    for cls_id in [0] if single_cls else boxes[:,5].unique():
        cls_boxes = boxes[boxes[:,5] == cls_id] if not single_cls else boxes
        cls_boxes = cls_boxes[cls_boxes[:,4].argsort(descending=True)]
        
        for box in cls_boxes:
            box_coords = box[:4] * box[4]
            best_iou, best_idx = -1, -1
            
            for i, (total, score, count) in enumerate(clusters):
                merged_coords = total / score
                iou = _box_ios(merged_coords, box[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_iou > ios_thresh:
                clusters[best_idx] = (
                    clusters[best_idx][0] + box_coords,
                    clusters[best_idx][1] + box[4],
                    clusters[best_idx][2] + 1
                )
            else:
                clusters.append((box_coords, box[4], 1))
    
    # Convert clusters to final boxes
    if not clusters:
        return torch.empty((0, 6), device=device)
    
    fused = []
    for total, score, count in clusters:
        box = torch.empty(6, device=device, dtype=dtype)
        box[:4] = total / score
        box[4] = score / count
        box[5] = 0 if single_cls else boxes[0,5]
        fused.append(box)
    
    fused = torch.stack(fused)
    return fused[fused[:,4].argsort(descending=True)][:post_limit]

def _box_ios(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate intersection over smallest area between two boxes."""
    lt = torch.max(box1[:2], box2[:2])
    rb = torch.min(box1[2:], box2[2:])
    inter = (rb - lt).clamp(min=0).prod()
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / min(area1, area2)

class Detector:
    def __init__(
        self,
        model_path: str,
        device: int | str | list = None,
        overlap: float = 0.2,
        patch_size: int = 1024,
        size_factor: float = 1.0,
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.7,
        max_patch_detections: int = 300,
        patch_per_batch: int = 4,
        pre_wbf_detections: int = 3000,
        wbf_ios_thresh: float = 0.5,
        max_detections: int = 1000,
        single_cls: bool = True,
    ):
        self.single_cls = single_cls
        self.device = device
        self.conf = conf_thresh
        self.iou = nms_iou_thresh
        self.max_det = max_patch_detections
        self.batch_size = patch_per_batch
        self.patch_size = patch_size
        self.size_factor = size_factor
        self.wbf_ios_thresh = wbf_ios_thresh
        self.pre_wbf_detections = pre_wbf_detections
        self.max_detections = max_detections
        self.overlap = overlap
        self.model = YOLO(model_path, task="detect")

    @torch.no_grad()
    def forward(self, image: Image.Image):
        # Preprocessing
        resized = resize_image(image, self.size_factor)
        patches, batch = create_patches(resized, self.patch_size, self.overlap)
        
        # Batch prediction
        results = self.model.predict(
            source=batch.to(self.device),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.patch_size,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            batch=self.batch_size
        )
        
        # Postprocessing
        merged = merge_predictions(patches, [r.boxes.data for r in results], self.size_factor)
        final = weighted_box_fusion(
            merged, 
            self.wbf_ios_thresh,
            self.pre_wbf_detections,
            self.max_detections,
            self.single_cls
        )
        
        # Format results
        names = {0: "object"} if self.single_cls else \
               self.model.model.names if hasattr(self.model.model, "names") else \
               self.model.model.module.names
        
        return Results(
            orig_img=np.array(image)[..., ::-1],
            path="",
            names=names,
            boxes=final.cpu()
        ).numpy()