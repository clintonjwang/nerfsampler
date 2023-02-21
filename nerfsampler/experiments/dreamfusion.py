import torch
import os

from PIL import Image
from nerfsampler.diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline

class Dreamfusion:
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to('cuda')
        self.pipe.enable_xformers_memory_efficient_attention()

    def forward(self, img: Image, mask: Image, prompt: str):
        init_img = img.convert('RGB').resize((768,768), Image.Resampling.BICUBIC)
        mask_img = Image.open('mask2.png').resize((768,768), Image.Resampling.BICUBIC)
        # = "a highly detailed snake in front of a yellow bowl on a playground"
        os.makedirs('out0', exist_ok=True)
        image = self.pipe(prompt=prompt, image=init_img, mask_image=mask_img,
            height=768, width=768, guidance_scale=6,
            num_inference_steps=80,
        )['images'][0]
        return image