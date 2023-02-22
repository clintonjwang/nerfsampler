from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

import pdb
from controlnet import config
import os.path as osp

import torch
import random

from controlnet.cldm.model import create_model, load_state_dict

class DreamFusion:
    def __init__(self):
        self.depth_model = create_model(osp.expandvars('$NFS/code/controlnet/controlnet/models/cldm_v15.yaml')).cpu()
        self.depth_model.load_state_dict(load_state_dict(osp.expandvars('$NFS/code/controlnet/controlnet/models/control_sd15_depth.pth'), location='cuda'))
        # self.normal_model = create_model(osp.expandvars('$NFS/code/controlnet/controlnet/models/cldm_v15.yaml')).cpu()
        # self.normal_model.load_state_dict(load_state_dict(osp.expandvars('$NFS/code/controlnet/controlnet/models/control_sd15_normal.pth'), location='cuda'))
        N = torch.numel(self.depth_model.alphas_cumprod)
        self.min_step = int(N * 0.02)
        self.max_step = int(N * 0.98)
        self.device = torch.device('cuda')

    def embed_prompts(self, prompt, n_prompt, mode='depth'):
        if mode == 'depth':
            model = self.depth_model.cuda()
        elif mode == 'normal':
            model = self.normal_model.cuda()
        return (model.get_learned_conditioning([prompt]), model.get_learned_conditioning([n_prompt]))

    def step(self, prompt_embeds, control_img, init_image, mode='depth',
        guidance_scale=7.5, seed=-1): #detect_resolution=384, bg_threshold=.4
        H = W = 512
        # if target_map is not None:
        #     target_map = (target_map.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
        if mode == 'depth':
            model = self.depth_model.cuda()
            # if target_map is None:
            #     input_image = HWC3(input_image)
            #     target_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        elif mode == 'normal':
            model = self.normal_model.cuda()
            # if target_map is None:
            #     input_image = HWC3(input_image)
            #     _, target_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
        # control = HWC3(target_map)
        # control = cv2.resize(control, (W, H), interpolation=cv2.INTER_LINEAR)
        # control = torch.from_numpy(control).float().cuda() / 255.0
        init_image = init_image.permute(2,0,1).unsqueeze(0)
        control = control_img.permute(2,0,1).unsqueeze(0) # b,3,h,w
        if control.size(1) == 1:
            control = control.tile(1,3,1,1)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [prompt_embeds[0]]}
        un_cond = {"c_concat": [control], "c_crossattn": [prompt_embeds[1]]}
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        latents = model.get_first_stage_encoding(model.first_stage_model.encode(init_image))
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = (extract_into_tensor(model.sqrt_alphas_cumprod, t, latents.shape) * latents +
                extract_into_tensor(model.sqrt_one_minus_alphas_cumprod, t, latents.shape) * noise)
            noise_pred_text = model.apply_model(latents_noisy, t, cond)
            noise_pred_uncond = model.apply_model(latents_noisy, t, un_cond)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        w = (1 - model.alphas_cumprod[t]/model.alphas_cumprod[t-1])
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        loss = SpecifyGradient.apply(latents, grad)
        return loss

from torch.cuda.amp import custom_bwd, custom_fwd 
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

# class DreamFusionDiffusers(StableDiffusionPipeline):
#     def sds(
#         self,
#         prompt: Union[str, List[str]] = None,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         height: Optional[int] = None,
#         width: Optional[int] = None,
#         num_inference_steps: int = 50,
#         guidance_scale: float = 7.5,
#         num_images_per_prompt: Optional[int] = 1,
#         eta: float = 0.0,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: Optional[str] = "pil",
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         # 0. Default height and width to unet
#         height = height or self.unet.config.sample_size * self.vae_scale_factor
#         width = width or self.unet.config.sample_size * self.vae_scale_factor

#         # 2. Define call parameters
#         if prompt is not None and isinstance(prompt, str):
#             batch_size = 1
#         elif prompt is not None and isinstance(prompt, list):
#             batch_size = len(prompt)
#         else:
#             batch_size = prompt_embeds.shape[0]

#         device = self._execution_device
#         # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
#         # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
#         # corresponds to doing no classifier free guidance.
#         do_classifier_free_guidance = guidance_scale > 1.0

#         # 3. Encode input prompt
#         prompt_embeds = self._encode_prompt(
#             prompt,
#             device,
#             num_images_per_prompt,
#             do_classifier_free_guidance,
#             negative_prompt,
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#         )

#         # 4. Prepare timesteps
#         self.scheduler.set_timesteps(num_inference_steps, device=device)
#         timesteps = self.scheduler.timesteps

#         # 5. Prepare latent variables
#         num_channels_latents = self.unet.in_channels
#         latents = self.prepare_latents(
#             batch_size * num_images_per_prompt,
#             num_channels_latents,
#             height,
#             width,
#             prompt_embeds.dtype,
#             device,
#             generator,
#             latents,
#         )

#         # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#         extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#         # 7. Denoising loop
#         num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
#         with self.progress_bar(total=num_inference_steps) as progress_bar:
#             for i, t in enumerate(timesteps):
#                 # expand the latents if we are doing classifier free guidance
#                 latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
#                 latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                 # predict the noise residual
#                 noise_pred = self.unet(
#                     latent_model_input,
#                     t,
#                     encoder_hidden_states=prompt_embeds,
#                     cross_attention_kwargs=cross_attention_kwargs,
#                 ).sample

#                 # perform guidance
#                 if do_classifier_free_guidance:
#                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#                 # compute the previous noisy sample x_t -> x_t-1
#                 latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

#         return score