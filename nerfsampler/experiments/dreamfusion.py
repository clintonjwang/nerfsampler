import gc
import os, pdb, torch
import shutil
osp = os.path
nn = torch.nn
F = nn.functional

from torchvision.transforms.functional import pil_to_tensor
from nerfstudio.cameras.cameras import Cameras

from nerfsampler import CODE_DIR
from nerfsampler.utils.camera import animate_camera
from nerfsampler.data.nerfacto import load_nerf_pipeline_for_scene
from nerfsampler.utils import util, point_cloud, mesh
from nerfsampler.experiments.render import *

import numpy as np
import os
from PIL import Image
torch.backends.cudnn.benchmark = True

from PIL import Image
from nerfsampler.networks import dreamfusion

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

    if False: #args['task'] == 'render RGB':
        folder = 'results/original'
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

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
    nerfacto, cams = ret
    n_frames = args['n_frames']
    sd_model = dreamfusion.DreamFusion()
    base_frame = 0

    print('Starting training')

    with torch.autocast('cuda'):
        optimizer = util.get_optimizer(nerfacto, args)
        prompt_embeds = sd_model.embed_prompts(prompt=args['prompt'], n_prompt=args['neg_prompt'])
        for _ in range(args['n_iterations']):
            camera_ray_bundle = cams.generate_rays(np.random.randint(len(cams))).to(nerfacto.device)
            animate_camera(camera_ray_bundle, frac=np.random.random())
            outputs = nerfacto.get_outputs_for_camera_ray_bundle_grad(camera_ray_bundle)
            depths = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth'])
            loss = sd_model.step(prompt_embeds, init_image=outputs['rgb'], control_img=depths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del outputs, depths, loss
            gc.collect()
            torch.cuda.empty_cache()

    folder = 'results/optimized'
    while osp.exists(folder):
        folder += '_'
        # shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    open(osp.join(folder, 'prompt.txt'), 'w').write(args['prompt'])
    shutil.copy(args['config path'])

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
