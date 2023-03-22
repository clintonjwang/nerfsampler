import gc
import os, torch
import shutil

import wandb
osp = os.path
nn = torch.nn
F = nn.functional

from tqdm import trange
from torch_ema import ExponentialMovingAverage
from torchvision import transforms

from nerfsampler import CODE_DIR
from nerfsampler.networks.siren import Siren
from nerfsampler.networks import tcnn
from nerfsampler.utils import util
from nerfsampler.experiments.render import *

import numpy as np
import os
from PIL import Image
from nerfsampler.networks import dreamfusion
import clip


def tv_norm(img):
    return torch.sum(torch.abs(img[..., :-1] - img[..., 1:])) + \
           torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))

def toy_dreamfusion(args={}):
    # wandb.init(project="dreamfusion", name=args["job_id"],
    #     config=wandb.helper.parse_config(args, exclude=['job_id']))
    os.chdir(f'{CODE_DIR}/nerfsampler')
    use_feats = args['use features']
    folder = 'results/'+args['config_name']
    while osp.exists(folder):
        folder += '_'
    os.makedirs(folder, exist_ok=True)
    shutil.copy(args['config path'], osp.join(folder, 'config.yaml'))
    sd_model = dreamfusion.DreamFusion(init_min=args['init min frac'], init_max=args['init max frac'])
    ldm = sd_model.depth_model
    # for p in ldm.parameters():
    #     p.requires_grad_(False)
    # init_img = torch.tensor(np.array(Image.open('00.png').resize((512,512))), device='cuda', dtype=torch.float32)/255
    init_img = torch.tensor(np.array(Image.open('cat.png').resize((512,512))), device='cuda', dtype=torch.float32)/255
    if use_feats:
        dim = 64
        C = 4
        init_feats = ldm.get_first_stage_encoding(ldm.encode_first_stage(init_img.permute(2,0,1).unsqueeze(0))).detach()
    else:
        dim = 512
        C = 3

    net_type = args['image parameterization']
    accumulation = args['accumulation']
    acc_step = 1
    if net_type == 'pixels':
        if use_feats:
            if args['init noise step']:
                feats = sd_model.add_noise(init_feats, args['init noise step'])
            else:
                feats = init_feats
            feats.requires_grad_(True)
            optimizer = util.get_optimizer(feats, args)
            params = feats
        else:
            rgb = init_img.permute(2,0,1).unsqueeze(0)
            rgb.requires_grad_(True)
            optimizer = util.get_optimizer(rgb, args)
            params = rgb
            
    elif net_type in ('siren', 'instant-ngp', 'mlp'):
        if use_feats:
            if net_type == 'siren':
                inr = Siren(out_channels=4, layers=4).cuda()
            else:
                inr = tcnn.get_tcnn_inr(args).cuda()
            rgb = ldm.decode_first_stage(init_feats)
            to_pil(rgb).save(f'{folder}/gt.png')
        else:
            if net_type == 'siren':
                inr = Siren(out_channels=3, layers=4).cuda()
            else:
                inr = tcnn.get_tcnn_inr(args).cuda()

        tensors = [torch.linspace(-1, 1, steps=dim, device='cuda'), torch.linspace(-1, 1, steps=dim, device='cuda')]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        coords = mgrid.reshape(-1, 2)
        params = inr.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-4)
        
        for iteration in range(args.get('fitting_iterations', 2500)):
            if use_feats:
                feats = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
                loss = (feats - init_feats).norm()
            else:
                rgb = inr(coords).reshape(dim, dim, C)
                loss = (rgb - init_img).norm()
            if iteration % 200 == 0:
                print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if use_feats:
            with torch.autocast('cuda'):
                rgb = ldm.decode_first_stage(feats)
        to_pil(rgb).save(f'{folder}/inr.png')
        optimizer = torch.optim.Adam(inr.parameters(), lr=float(args['optimizer']['learning rate']))

    augment = args['augment']
    if augment:
        augments = transforms.Compose([
            transforms.RandomPerspective(p=.2),
            transforms.RandomResizedCrop(size=(dim,dim), scale=(0.8,1.)),
        ])


    print('Starting training')
    prompt_embeds = sd_model.embed_prompts(prompt=args['prompt'], n_prompt=args['neg_prompt'])#, mode='normal')
    gs = args.get('guidance_scale', 10)
    if args['clip guidance']:
        clip_model, preprocess = clip.load('ViT-L/14', device='cuda')
        with torch.no_grad():
            pos_embedding = clip_model.encode_text(clip.tokenize([args['prompt']]).to('cuda'))
            neg_embedding = clip_model.encode_text(clip.tokenize([args['neg_prompt']]).to('cuda'))

    if False:
        if use_feats:
            if net_type in ('siren', 'instant-ngp', 'mlp'):
                feats = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
        else:
            if net_type in ('siren', 'instant-ngp', 'mlp'):
                rgb = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
            with torch.autocast('cuda'):
                feats = ldm.get_first_stage_encoding(ldm.first_stage_model.encode(rgb))

        sd_model.visualize_scores(prompt_embeds, latents=feats, guidance_scale=gs, folder=folder)

        exit()

    for cur_iter in trange(args['n_iterations']):
        if args['step reduce frequency']:
            if cur_iter % args['step reduce frequency'] == 0 and sd_model.min_step > 5:
                sd_model.min_step -= 1
            if cur_iter % args['step reduce frequency'] == 0 and sd_model.max_step > 50:
                sd_model.max_step -= 1
        gs += args.get('guidance_change', 0)
        if use_feats:
            if net_type in ('siren', 'instant-ngp', 'mlp'):
                feats = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
            if augment:
                rgb = ldm.decode_first_stage_grad(feats)
                aug_rgb = augments(rgb)
                aug_feats = ldm.get_first_stage_encoding(ldm.first_stage_model.encode(aug_rgb))
                # aug_feats = augments(feats)
            else:
                aug_feats = feats
            # elif cur_iter == 30:
            #     feats = sd_model.add_noise(feats.detach())
            #     feats.requires_grad_(True)
            #     optimizer = util.get_optimizer(feats, args)
        else:
            if net_type in ('siren', 'instant-ngp', 'mlp'):
                rgb = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
            with torch.autocast('cuda'):
                if augment:
                    aug_rgb = augments(rgb)
                    aug_feats = ldm.get_first_stage_encoding(ldm.first_stage_model.encode(aug_rgb))
                else:
                    aug_feats = ldm.get_first_stage_encoding(ldm.first_stage_model.encode(rgb))
        # if sd_model.max_step < args['direct score ascent threshold']:
        #     grad = sd_model.direct_score_ascent(prompt_embeds, latents=feats,
        #                 guidance_scale=gs)
        #     feats.backward(grad)
        tv_w = 0 #1e-4

        # if sd_model.max_step < args['direct reversal threshold']:
        #     grad = sd_model.direct_reversal(prompt_embeds, latents=aug_feats,
        #                 guidance_scale=gs)
        #     aug_feats.backward(grad)

        if args['backprop unet']:
            loss = sd_model.sds(prompt_embeds, latents=aug_feats,
                                guidance_scale=gs, latents_grad=True)
            if tv_w:
                if use_feats:
                    loss -= tv_norm(feats) * tv_w
                else:
                    loss -= tv_norm(rgb) * tv_w
            loss.backward()

        else:
            grad = sd_model.sds(prompt_embeds, latents=aug_feats,
                                guidance_scale=gs, latents_grad=False)
            if tv_w:
                if use_feats:
                    loss = -tv_norm(feats) * tv_w
                else:
                    loss = -tv_norm(rgb) * tv_w
                loss.backward(inputs=aug_feats)
                aug_feats.backward(grad + aug_feats.grad)
            else:
                aug_feats.backward(grad, inputs=params)

        if args['clip guidance']:
            if use_feats:
                rgb = ldm.decode_first_stage_grad(feats)
            image = preprocess.transforms[0](rgb)
            if rgb.shape[-1] != rgb.shape[-2]:
                image = preprocess.transforms[1](image)
            image_features = clip_model.encode_image(image)
            clip_loss = args['clip guidance'] * (-F.cosine_similarity(image_features, pos_embedding) + F.cosine_similarity(image_features, neg_embedding))
            clip_loss.backward(inputs=params)

        if cur_iter % 29 == 0:
            with torch.autocast('cuda'):
                if use_feats:
                    rgb = ldm.decode_first_stage(feats)
                    if feats.grad is not None:
                        print(feats.grad.norm().item(), feats.grad.abs().max().item())
                
                to_pil(rgb).save(f'{folder}/{cur_iter//29:02d}.png')
        # wandb.log({'grad_norm': grad.norm().item()}, step=cur_iter)
        # print(loss.item())
        if acc_step == accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            acc_step = 1
        else:
            acc_step += 1
    
    if use_feats:
        if net_type in ('siren', 'instant-ngp', 'mlp'):
            feats = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
        with torch.autocast('cuda'):
            rgb = ldm.decode_first_stage(feats)
    else:
        if net_type in ('siren', 'instant-ngp', 'mlp'):
            rgb = inr(coords).reshape(dim, dim, C).permute(2,0,1).unsqueeze(0)
    to_pil(rgb).save(f'{folder}/final.png')

def to_pil(rgb):
    if len(rgb.shape) == 3:
        return Image.fromarray((rgb*255).detach().cpu().numpy().astype('uint8'))
    return Image.fromarray((rgb[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8'))
