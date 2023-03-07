import gc
import os, torch
import shutil

import wandb
osp = os.path
nn = torch.nn
F = nn.functional

from tqdm import trange
from torch_ema import ExponentialMovingAverage

from nerfsampler import CODE_DIR
from nerfsampler.networks.siren import Siren
from nerfsampler.utils import util
from nerfsampler.experiments.render import *

import numpy as np
import os
from PIL import Image
from nerfsampler.networks import dreamfusion
torch.backends.cudnn.benchmark = True

def toy_dreamfusion(args={}):
    # wandb.init(project="dreamfusion", name=args["job_id"],
    #     config=wandb.helper.parse_config(args, exclude=['job_id']))
    os.chdir(f'{CODE_DIR}/nerfsampler')
    use_feats = args['use features']
    folder = 'results/optimized'
    while osp.exists(folder):
        folder += '_'
    os.makedirs(folder, exist_ok=True)
    shutil.copy(args['config path'], osp.join(folder, 'config.yaml'))
    sd_model = dreamfusion.DreamFusion()
    ldm = sd_model.depth_model
    # for p in ldm.parameters():
    #     p.requires_grad_(False)
    init_img = torch.tensor(np.array(Image.open('00.png')), device='cuda', dtype=torch.float32)/255
    if use_feats:
        dim = 64
        C = 4
        init_feats = ldm.get_first_stage_encoding(ldm.encode_first_stage(init_img.permute(2,0,1).unsqueeze(0))).detach()
    else:
        dim = 512
        C = 3

    if args['image parameterization'] == 'pixels':
        if use_feats:
            feats = sd_model.add_noise(init_feats, args['init noise step'])
            feats.requires_grad_(True)
            optimizer = util.get_optimizer(feats, args)
        else:
            rgb = init_img.permute(2,0,1).unsqueeze(0)
            rgb.requires_grad_(True)
            optimizer = util.get_optimizer(rgb, args)
            
    elif args['image parameterization'] == 'siren':
        if use_feats:
            siren = Siren(out_channels=4).cuda()
            rgb = ldm.decode_first_stage(init_feats)
            to_pil(rgb).save(f'{folder}/gt.png')
        else:
            siren = Siren(out_channels=3).cuda()

        tensors = [torch.linspace(-1, 1, steps=dim, device='cuda'), torch.linspace(-1, 1, steps=dim, device='cuda')]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        coords = mgrid.reshape(-1, 2)
        optimizer = torch.optim.Adam(siren.parameters(), lr=1e-3)
        
        for _ in range(1000):
            if use_feats:
                feats = siren(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
                loss = (feats - init_feats).norm()
            else:
                rgb = siren(coords).reshape(dim, dim, C)
                loss = (rgb - init_img).norm()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        to_pil(rgb).save(f'{folder}/siren.png')
        optimizer = torch.optim.Adam(siren.parameters(), lr=float(args['optimizer']['learning rate']))


    print('Starting training')
    prompt_embeds = sd_model.embed_prompts(prompt=args['prompt'], n_prompt=args['neg_prompt'])#, mode='normal')
    gs = args.get('guidance_scale', 10)
    for cur_iter in trange(args['n_iterations']):
        if args['step reduce frequency']:
            if cur_iter % args['step reduce frequency'] == 0 and sd_model.min_step > 5:
                sd_model.min_step -= 1
            if cur_iter % args['step reduce frequency'] == 0 and sd_model.max_step > 50:
                sd_model.max_step -= 1
        gs += args.get('guidance_change', 0)
        if use_feats:
            if args['image parameterization'] == 'siren':
                feats = siren(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
            # elif cur_iter == 30:
            #     feats = sd_model.add_noise(feats.detach())
            #     feats.requires_grad_(True)
            #     optimizer = util.get_optimizer(feats, args)
        else:
            if args['image parameterization'] == 'siren':
                rgb = siren(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
            feats = ldm.get_first_stage_encoding(ldm.first_stage_model.encode(rgb))
            
        # if sd_model.max_step < args['direct score ascent threshold']:
        #     grad = sd_model.direct_score_ascent(prompt_embeds, latents=feats,
        #                 guidance_scale=gs)
        #     feats.backward(grad)

        if sd_model.max_step < args['direct reversal threshold']:
            grad = sd_model.direct_reversal(prompt_embeds, latents=feats,
                        guidance_scale=gs)
            feats.backward(grad)

        elif args['backprop unet']:
            loss = sd_model.sds(prompt_embeds, latents=feats,
                                guidance_scale=gs, latents_grad=True)
            loss.backward()
        else:
            grad = sd_model.sds(prompt_embeds, latents=feats,
                                guidance_scale=gs, latents_grad=False)
            feats.backward(grad)
        # feats.grad.clip_(-1, 1)

        if cur_iter % 29 == 0:
            if use_feats:
                rgb = ldm.decode_first_stage(feats)
                if feats.grad is not None:
                    print(feats.grad.norm().item(), feats.grad.abs().max().item())
            
            to_pil(rgb).save(f'{folder}/{cur_iter//29:02d}.png')
        # wandb.log({'loss': loss.norm().item()}, step=cur_iter)
        # print(loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    rgb = ldm.decode_first_stage(feats)
    to_pil(rgb).save(f'{folder}/final.png')

def to_pil(rgb):
    if len(rgb.shape) == 3:
        return Image.fromarray((rgb*255).detach().cpu().numpy().astype('uint8'))
    return Image.fromarray((rgb[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8'))
