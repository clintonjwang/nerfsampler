import torch, pdb
nn=torch.nn
F=nn.functional

import tensorflow.compat.v1 as tf
import tensorflow as tf2

from PIL import Image
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with tf.device('/cpu:0'):
            self.model = tf2.saved_model.load('./exported_model', tags=[tf.saved_model.tag_constants.SERVING],)
        
    def forward(self, imgs, text_embedding):
        text_embedding = text_embedding.numpy()
        text_embedding = tf.reshape(
            text_embedding, [-1, 1, text_embedding.shape[-1]])
        text_embedding = tf.cast(text_embedding, tf.float32)

        outputs = []
        for img in imgs:
            tmp_path = 'tmp.png'
            img -= img.min()
            img /= img.max()/255.0
            Image.fromarray(img.cpu().numpy().astype('uint8')).save(tmp_path)
            np_image_string = np.array([tf.gfile.GFile(tmp_path, 'rb').read()])
            inp_img = tf.convert_to_tensor(np_image_string[0])
            output = self.model.signatures['serving_default'](
                inp_image_bytes=inp_img,
                inp_text_emb=text_embedding)
            outputs.append(output['image_embedding_feat'])

        return torch.tensor(tf.concat(axis=0, values=outputs).numpy())
