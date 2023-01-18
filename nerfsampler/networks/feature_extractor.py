import torch
nn=torch.nn
F=nn.functional

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        
    def forward(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return inputs
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)