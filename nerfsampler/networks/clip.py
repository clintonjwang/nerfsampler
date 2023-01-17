"""Data augmentation"""
import torch
import numpy as np

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

nn=torch.nn
F=nn.functional

class CLIP(nn.Module):
    def __init__(
        self,
        text_encoder: CLIPTextModel,
    ):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, prompts, device):
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
