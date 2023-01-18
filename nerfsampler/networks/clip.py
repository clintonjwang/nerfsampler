"""Data augmentation"""
import torch
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

nn=torch.nn
F=nn.functional

class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = 'cuda'

    def forward(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings