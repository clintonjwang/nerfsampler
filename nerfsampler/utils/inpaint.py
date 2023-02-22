import shutil
import torch, sys, os, argparse
from nerfsampler.diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
torch.backends.cudnn.benchmark = True

from . import cnet
from nerfsampler.diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline

class DreamfusionTrainer(nn.Module):
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

    def train_step(self, input_imgs, text_embeddings, pred_rgb, pred_depth, pred_normals, guidance_scale=100):
        # interp to 512x512 to be fed into vae.
        cnet.controlnet(self, 'a hamburger', mode='depth', target_map=pred_depth, input_image=pred_rgb)
        # cnet.controlnet(pred_rgb_512, mode='normal', target_map=pred_normals)
        pdb.set_trace()

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(input_imgs)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad) 

        return loss 
