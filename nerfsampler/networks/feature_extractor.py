import torch, pdb
nn=torch.nn
F=nn.functional

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

import tensorflow.compat.v1 as tf
import tensorflow as tf2


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = tf2.saved_model.load('./models', tags=[tf.saved_model.tag_constants.SERVING],)
        
    def forward(self, image, text_emb):
        img = tf.convert_to_tensor(image.cpu())
        pdb.set_trace()
        text_emb = tf.convert_to_tensor(text_emb.cpu())
        features = self.model.signatures['serving_default'](inp_image_bytes=img, inp_text_emb=text_emb)

        return features
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
