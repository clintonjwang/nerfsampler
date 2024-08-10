from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

import pdb
from controlnet import config
import os.path as osp

import torch
from PIL import Image
from controlnet.cldm.model import create_model, load_state_dict

def to_pil(grad):
    grad -= grad.min()
    grad = (grad * 20).clamp_max(1)
    return Image.fromarray((grad[0,:3].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8'))

class DreamFusion:
    def __init__(self, init_min=0.2, init_max=0.4):
        self.depth_model = create_model(osp.expandvars('$NFS/code/controlnet/controlnet/models/cldm_v15.yaml')).cuda()
        self.depth_model.load_state_dict(load_state_dict(osp.expandvars('$NFS/code/controlnet/controlnet/models/control_sd15_depth.pth'), location='cuda'))
        # self.normal_model = create_model(osp.expandvars('$NFS/code/controlnet/controlnet/models/cldm_v15.yaml')).cuda()
        # self.normal_model.load_state_dict(load_state_dict(osp.expandvars('$NFS/code/controlnet/controlnet/models/control_sd15_normal.pth'), location='cuda'))
        self.T = N = torch.numel(self.depth_model.alphas_cumprod)
        self.min_step = int(N * init_min) #.02
        self.max_step = int(N * init_max) #0.98
        self.device = torch.device('cuda')

    def embed_prompts(self, prompt, n_prompt, mode='depth'):
        if mode.startswith('normal'):
            model = self.normal_model
        else:
            model = self.depth_model
        return (model.get_learned_conditioning([prompt]), model.get_learned_conditioning([n_prompt]))
     
    def add_noise(self, latents, t2, t1=None, noise=None):
        ldm = self.depth_model
        if noise is None:
            noise = torch.randn_like(latents)
        if isinstance(t2, torch.Tensor):
            t2 = t2.item()
        if t1 is None:
            return ldm.sqrt_alphas_cumprod[t2] * latents + \
                ldm.sqrt_one_minus_alphas_cumprod[t2] * noise

        if isinstance(t1, torch.Tensor):
            t1 = t1.item()
        a1 = ldm.sqrt_alphas_cumprod[t1]
        sig1 = ldm.sqrt_one_minus_alphas_cumprod[t1]
        a2 = ldm.sqrt_alphas_cumprod[t2]
        sig2 = ldm.sqrt_one_minus_alphas_cumprod[t2]

        scale = a2/a1
        sigma = (sig2**2 - (scale * sig1)**2).sqrt()
        return scale * latents + sigma * noise
    
    def sds(self, prompt_embeds, init_image=None, latents=None, control_img=None, mode='depth',
        control_strength=1., guidance_scale=7.5, latents_grad=False, normalize=False, estimate_noise=True, verbose=False, t1=None):
        if mode.startswith('normal'):
            model = self.normal_model
        else:
            model = self.depth_model

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[0]]}
        un_cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[1]]}
        model.control_scales = ([control_strength] * 13) if control_img is not None else ([0] * 13)

        if estimate_noise:
            t0 = self.max_step
            with torch.no_grad():
                latent_ft = torch.fft.fft2(latents)
                high_f = latent_ft[...,28:36,28:36].abs().mean()
                low_f = latent_ft[...,:4,:4].abs().mean()
                snr = max(low_f / high_f - 1, 0) #lower bound
            for t in range(self.max_step):
                if model.sqrt_alphas_cumprod[t] / model.sqrt_one_minus_alphas_cumprod[t] < snr:
                    t0 = t
                    break
            if verbose:
                print(snr.item(), t0)
        else:
            t0 = self.min_step
        
        if t1 is not None:
            t = torch.tensor([max(t0, t1)], dtype=torch.long, device=self.device)
        else:
            t = torch.randint(t0, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        if latents is None:
            assert init_image.size(1) == 3 # b,3,h,w
            latents = model.get_first_stage_encoding(model.first_stage_model.encode(init_image))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        noise = torch.randn_like(latents)
        if latents_grad:
            latents_noisy = self.add_noise(latents, t, t0, noise=noise)
            noise_pred_text = model.apply_model(latents_noisy, t, cond, with_grad=True)
            noise_pred_uncond = model.apply_model(latents_noisy, t, un_cond, with_grad=True)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            grad = noise_pred - noise
            if normalize:
                grad = grad / (torch.norm(grad, dim=1, keepdim=True) + 0.1)
            loss = grad.pow(2).sum()
            return loss
        else:
            with torch.no_grad():
                latents_noisy = self.add_noise(latents, t, t0, noise=noise)
                noise_pred_text = model.apply_model(latents_noisy, t, cond)
                noise_pred_uncond = model.apply_model(latents_noisy, t, un_cond)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # w = (1 - model.alphas_cumprod[t]/model.alphas_cumprod[t-1])
            grad = noise_pred - noise
            if normalize:
                grad = grad / (torch.norm(grad, dim=1, keepdim=True) + 0.1)
            return grad

    @torch.no_grad()
    def visualize_scores(self, prompt_embeds, init_image=None, latents=None, control_img=None, mode='depth',
        control_strength=1., guidance_scale=7.5, folder=None):
        model = self.depth_model
        cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[0]]}
        un_cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[1]]}
        model.control_scales = ([control_strength] * 13) if control_img is not None else ([0] * 13)

        if latents is None:
            assert init_image.size(1) == 3 # b,3,h,w
            latents = model.get_first_stage_encoding(model.first_stage_model.encode(init_image))
        
        noise = torch.randn_like(latents)
        for t in torch.arange(50, 950 + 1, 50, dtype=torch.long, device=self.device):
            t.unsqueeze_(0)
            latents_noisy = (model.sqrt_alphas_cumprod[t.item()] * latents +
                model.sqrt_one_minus_alphas_cumprod[t.item()] * noise)
            noise_pred_text = model.apply_model(latents_noisy, t, cond)
            noise_pred_uncond = model.apply_model(latents_noisy, t, un_cond)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            w = (1 - model.alphas_cumprod[t]/model.alphas_cumprod[t-1])
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            to_pil(grad).save(f'{folder}/{t.item():04d}_0.png')

        noise = torch.randn_like(latents)
        for t in torch.arange(50, 950 + 1, 50, dtype=torch.long, device=self.device):
            t.unsqueeze_(0)
            latents_noisy = (model.sqrt_alphas_cumprod[t.item()] * latents +
                model.sqrt_one_minus_alphas_cumprod[t.item()] * noise)
            noise_pred_text = model.apply_model(latents_noisy, t, cond)
            noise_pred_uncond = model.apply_model(latents_noisy, t, un_cond)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            w = (1 - model.alphas_cumprod[t]/model.alphas_cumprod[t-1])
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            to_pil(grad).save(f'{folder}/{t.item():04d}_1.png')

    def direct_score_ascent(self, prompt_embeds, init_image=None, latents=None, control_img=None, mode='depth',
        control_strength=1., guidance_scale=7.5):
        if mode.startswith('normal'):
            model = self.normal_model
        else:
            model = self.depth_model

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[0]]}
        un_cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[1]]}
        model.control_scales = ([control_strength] * 13) if control_img is not None else ([0] * 13)

        t = torch.randint(0, self.max_step+1, [1], dtype=torch.long, device=self.device)
        if latents is None:
            assert init_image.size(1) == 3 # b,3,h,w
            latents = model.get_first_stage_encoding(model.first_stage_model.encode(init_image))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        noise_pred_text = model.apply_model(latents, t, cond, with_grad=True)
        noise_pred_uncond = model.apply_model(latents, t, un_cond, with_grad=True)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        w = (1 - model.alphas_cumprod[t]/model.alphas_cumprod[t-1])
        return w * noise_pred
    
    def get_timesteps(self, num_inference_steps, strength):
        # when strength is 1, same as unconditional
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def direct_reversal(self, prompt_embeds, init_image=None, latents=None,
        control_img=None, mode='depth',
        control_strength=1., guidance_scale=7.5, strength=50):
        from controlnet.cldm.ddim_hacked import DDIMSampler
        if mode.startswith('normal'):
            model = self.normal_model
        else:
            model = self.depth_model

        timesteps, num_inference_steps = self.get_timesteps(self.T, strength)
        latent_timestep = timesteps[:1]
        
        cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[0]]}
        un_cond = {"c_concat": control_img, "c_crossattn": [prompt_embeds[1]]}
        model.control_scales = ([control_strength] * 13) if control_img is not None else ([0] * 13)

        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        if latents is None:
            assert init_image.size(1) == 3 # b,3,h,w
            latents = model.get_first_stage_encoding(model.first_stage_model.encode(init_image))
        H, W, C = init_image.shape
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        ddim_sampler = DDIMSampler(model)
        samples, _ = ddim_sampler.sample(num_inference_steps, 1,
                                shape, cond, verbose=False, eta=0,
                                unconditional_guidance_scale=guidance_scale,
                                unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        out_img = (model.decode_first_stage(samples) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype('uint8')
        return out_img


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
