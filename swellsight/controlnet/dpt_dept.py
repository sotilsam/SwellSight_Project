from typing import Tuple
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation


class DPTDepthExtractor:
    def __init__(self, model_name: str = "Intel/dpt-large", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def predict_depth(self, image: Image.Image) -> np.ndarray:
        inp = self.processor(images=image, return_tensors="pt").to(self.device)
        pred = self.model(**inp).predicted_depth

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze(0).squeeze(0)

        return pred.detach().cpu().numpy().astype(np.float32)
