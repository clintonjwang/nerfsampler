import gc
import os, pdb, torch
import shutil
osp = os.path
nn = torch.nn
F = nn.functional

from tqdm import trange
from torchvision.transforms.functional import pil_to_tensor, gaussian_blur
from nerfstudio.cameras.cameras import Cameras

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

    nerfacto, cams = ret
    n_frames = args['n_frames']
    sd_model = dreamfusion.DreamFusion()
    base_frame = 0

    print('Starting training')
    counter = 0
    with torch.autocast('cuda'):
        optimizer = util.get_optimizer(nerfacto, args)
        prompt_embeds = sd_model.embed_prompts(prompt=args['prompt'], n_prompt=args['neg_prompt'])
        gs = args['guidance_scale']
        for cur_iter in trange(args['n_iterations']):
            gs *= args['guidance_decay']
            camera_ray_bundle = cams.generate_rays(np.random.randint(len(cams))).to(nerfacto.device)
            animate_camera(camera_ray_bundle, frac=np.random.random())
            outputs = nerfacto.get_outputs_for_camera_ray_bundle_grad(camera_ray_bundle)
            depths = outputs['depth']
            depths.clamp_(*args['depth_cutoff'])
            depths = depths.min()/depths
            # depths -= depths.min()
            # depths /= depths.max()
            depths = gaussian_blur(depths.permute(2,0,1).unsqueeze(0), 5)
            depths = depths.tile(1,3,1,1)
            rgb = outputs['rgb'].permute(2,0,1).unsqueeze(0)
            # if counter < 6:
            #     Image.fromarray((rgb[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8')).save(f'{folder}/{counter}_rgb.png')
            #     Image.fromarray((depths[0].permute(1,2,0)*255).detach().cpu().numpy().astype('uint8')).save(f'{folder}/{counter}_depth.png')
            #     counter += 1
            loss = sd_model.step(prompt_embeds, init_image=rgb, control_img=depths, guidance_scale=gs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del outputs, depths, rgb, loss
            gc.collect()
            torch.cuda.empty_cache()

    camera_ray_bundle = cams.generate_rays(base_frame).to(nerfacto.device)
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')



# def render_with_seg_fit_inpaint(nerfacto, camera_ray_bundle, inpainting, seg, n_frames=64):
#     for ix in range(len(nerfacto.density_fns)):
#         nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], seg)
#     nerfacto.field.get_density = attenuate_field_density(nerfacto.field, seg)
#     outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
#     inpaint_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth'])
#     in_range = inpaint_coords.norm(dim=-1) < 1
#     rgb = outputs['rgb'][in_range]
#     inpaint_coords = inpaint_coords[in_range]
#     inpaint_scale = inpainting[in_range] / (rgb * 255)
#     nerfacto.field.forward = forward_inpaint(nerfacto.field,
#         inpaint_scale, inpaint_coords)
    
#     folder = 'results/fit_inpaint'
#     os.makedirs(folder, exist_ok=True)
    
#     orig = torch.clone(camera_ray_bundle.origins)
#     dirs = torch.clone(camera_ray_bundle.directions)
#     for frame in range(n_frames):
#         animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
#         nerfacto.field.t = frame / n_frames * 2 * pi
#         generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')
