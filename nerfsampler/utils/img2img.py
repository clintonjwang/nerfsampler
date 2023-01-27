import torch, os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

def img2img(image, model_type):
    if model_type == 'protogen':
        pipe = StableDiffusionPipeline.from_pretrained(
            'darkstorm2150/Protogen_x5.3_Official_Release', #Photorealism
            torch_dtype=torch.float16, safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = StableDiffusionImg2ImgPipeline(**pipe.components).to("cuda")
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
            safety_checker=None,
            torch_dtype=torch.float16).to(
            'cuda'
        )
    pipe.enable_xformers_memory_efficient_attention()

    prompt = "cartoon wallet, creative masterpiece, hand-drawn, Disney, large eyes, hyper detailed"
    folder = f'out'
    os.makedirs(folder, exist_ok=True)
    init_image = Image.fromarray(image).convert('RGB').resize((768, 768), Image.Resampling.BICUBIC)
    for ix in range(20):
        images = pipe(prompt=prompt, image=init_image,
            strength=0.7, guidance_scale=8, num_inference_steps=70).images
        images[0].save(f"{folder}/{ix:02d}.png")
