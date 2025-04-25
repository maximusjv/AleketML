from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

class Classificator:
    def __init__(self, device: str | int | list, model_path: str, img_size: int) -> None:
        self.device = device
        self.model = YOLO(model_path, task="classify")
        self.img_size = img_size
    
    def forward(self, source: list[Image.Image]) -> Results:
        source = [img.resize((self.img_size, self.img_size)) for img in source]
        batch_np = np.stack([np.array(img) for img in source])
        batch_tensor = (
            torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
        )
        return self.model.predict(batch_tensor,
                                  device=self.device, 
                                  imgsz=self.img_size,
                                  verbose=False,
                                    )
    