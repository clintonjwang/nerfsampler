import gc
import os, pdb, torch
import shutil

import wandb
osp = os.path
nn = torch.nn
F = nn.functional

from tqdm import trange
from torchvision.transforms.functional import pil_to_tensor, gaussian_blur
from nerfstudio.cameras.cameras import Cameras
# from torch_ema import ExponentialMovingAverage

from nerfsampler import CODE_DIR
from nerfsampler.utils.camera import animate_camera
from nerfsampler.data.nerfacto import load_nerf_pipeline_for_scene
from nerfsampler.utils import util
from nerfsampler.experiments.render import *

import numpy as np
import os
from PIL import Image
from nerfsampler.networks import dreamfusion
torch.backends.cudnn.benchmark = True

def setup(args):
    wandb.init(project="dreamfusion", name=args["job_id"],
        config=wandb.helper.parse_config(args, exclude=['job_id']))
    dl_args = args["data"]
    scene_id = dl_args['scene_id']
    n_frames = args['n_frames']
    resolution = 512
    base_frame = 0
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    pipeline.datamanager.setup_train()
    cams = pipeline.datamanager.train_dataset.cameras
    os.chdir(f'{CODE_DIR}/nerfsampler')

    native_res = (cams[0].width.item(), cams[0].height.item())
    if native_res[0] != native_res[1]:
        cams = Cameras(camera_to_worlds=cams.camera_to_worlds,
            fx=cams.fx, fy=cams.fx,
            cx=cams.cx, cy=cams.cx,
            width=cams.width, height=cams.width,
            distortion_params=cams.distortion_params,
            camera_type=cams.camera_type,
            times=cams.times)
    native_res = native_res[0]
    cams.rescale_output_resolution(resolution/native_res)

    folder = 'results/original'
    if not osp.exists(folder):
        os.makedirs(folder)
        camera_ray_bundle = cams.generate_rays(base_frame).to(pipeline.model.device)
        orig = torch.clone(camera_ray_bundle.origins)
        dirs = torch.clone(camera_ray_bundle.directions)

        for frame in range(n_frames):
            animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
            generate_view(pipeline.model, camera_ray_bundle, f'{folder}/{frame:02d}.png')
        return

    return pipeline.model, cams


def run_dreamfusion(args={}):
    ret = setup(args)
    if ret is None:
        return
    folder = 'results/optimized'
    while osp.exists(folder):
        folder += '_'
        # shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    shutil.copy(args['config path'], osp.join(folder, 'config.yaml'))
    base_frame = 0

    nerfacto, cams = ret
    base_ray_bundle = cams.generate_rays(base_frame).to(nerfacto.device)
    n_frames = args['n_frames']
    sd_model = dreamfusion.DreamFusion()
    counter = 0
    def cams2outs():
        camera_ray_bundle = cams.generate_rays(np.random.randint(len(cams))).to(nerfacto.device)
        animate_camera(camera_ray_bundle, frac=np.random.random())
        outputs = nerfacto.get_outputs_for_camera_ray_bundle_grad(camera_ray_bundle)
        if 'normals' in outputs:
            normals = outputs['normals'].permute(2,0,1).unsqueeze(0)
        else:
            normals = None
        depths = outputs['depth']
        depths.clamp_(*args['depth_cutoff'])
        depths = depths.min()/depths.permute(2,0,1).unsqueeze(0).tile(1,3,1,1)
        # depths = gaussian_blur(depths, 5)
        rgb = outputs['rgb'].permute(2,0,1).unsqueeze(0)
        # if counter < 6:
        #     Image.fromarray((rgb[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8')).save(f'{folder}/{counter}_rgb.png')
        #     Image.fromarray((depths[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8')).save(f'{folder}/{counter}_depth.png')
        #     counter += 1
        return rgb, depths, normals

    print('Starting training')
    with torch.autocast('cuda'):
        # ema = ExponentialMovingAverage(nerfacto.parameters(), decay=args.get('ema_decay', 0.9))
        if 'stage 1' in args:
            s1args = args['stage 1']
            optimizer = util.get_optimizer(nerfacto, s1args)
            prompt_embeds = sd_model.embed_prompts(prompt=s1args['prompt'], n_prompt=s1args['neg_prompt'])
            for cur_iter in trange(s1args['n_iterations']):
                outputs = nerfacto.get_outputs_for_camera_ray_bundle_grad(base_ray_bundle)
                rgb = outputs['rgb'].permute(2,0,1).unsqueeze(0)
                sd_model.max_step -= 1
                if cur_iter % 29 == 0:
                    generate_view(nerfacto, base_ray_bundle,
                        f'{folder}/stage1_{cur_iter//29:02d}.png', include_depth=True)
                optimizer.zero_grad()
                grad = sd_model.step(prompt_embeds, init_image=rgb,
                    guidance_scale=s1args.get('guidance_scale', 10))
                optimizer.step()
                # ema.update()
                del outputs, rgb, grad
                gc.collect()
                torch.cuda.empty_cache()

        optimizer = util.get_optimizer(nerfacto, args)
        prompt_embeds = sd_model.embed_prompts(prompt=args['prompt'], n_prompt=args['neg_prompt'], mode='normal')
        gs = args.get('guidance_scale', 10)
        if args.get('control_strength', None) is None:
            CS = [1] * args['n_iterations']
        else:
            CS = np.linspace(*args['control_strength'], args['n_iterations'])
        for cur_iter in trange(args['n_iterations']):
            if sd_model.max_step > 200:
                sd_model.max_step -= 1
            gs *= args.get('guidance_decay', 1)
            if cur_iter % 29 == 0:
                generate_view(nerfacto, base_ray_bundle,
                    f'{folder}/stage2_{cur_iter//29:02d}.png', include_depth=True)
            rgb, depths, normals = cams2outs()
            optimizer.zero_grad()
            # grad = sd_model.step(prompt_embeds, init_image=rgb, control_img=[normals], mode='normal',
            #     guidance_scale=gs, control_strength=CS[cur_iter])
            grad = sd_model.step(prompt_embeds, init_image=rgb, control_img=[depths],
                guidance_scale=gs, control_strength=CS[cur_iter])
            wandb.log({'grad': grad.norm().item()}, step=cur_iter)
            optimizer.step()
            # ema.update()
            del depths, normals, rgb, grad
            gc.collect()
            torch.cuda.empty_cache()
                
    orig = torch.clone(base_ray_bundle.origins)
    dirs = torch.clone(base_ray_bundle.directions)
    # with ema.average_parameters():
    for frame in range(n_frames):
        animate_camera(base_ray_bundle, (orig, dirs), frac=frame/n_frames)
        generate_view(nerfacto, base_ray_bundle, f'{folder}/{frame:02d}.png')
