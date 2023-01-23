"""Data augmentation"""
import torch
import clip

nn=torch.nn
F=nn.functional

class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.model, preprocess = clip.load("ViT-L/14@336px")

    def forward(self, categories):
        with torch.no_grad():
            all_text_embeddings = []
            for category in categories:
                texts = clip.tokenize(category).cuda()  #tokenize
                text_embeddings = self.model.encode_text(texts)  #embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            
        return all_text_embeddings
        