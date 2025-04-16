from PIL import Image
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

class Classificator:
    def __init__(self, device: str | int | list, model_path: str) -> None:
        self.device = device
        self.model = YOLO(model_path, task="classify")
    
    def forward(self, source):
        return self.model.predict(source,
                                  device=self.device, 
                                    )
    