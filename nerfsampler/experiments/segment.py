import os, pdb, torch
import wandb
import yaml

from nerfsampler.utils import tf_util
from nerfsampler.utils import jobs as job_mgmt
osp = os.path
nn = torch.nn
F = nn.functional

from nerfsampler.networks.clip import TextEmbedder
from nerfsampler.networks.feature_extractor import FeatureExtractor
from nerfsampler.data.nerfacto import load_nerf_pipeline_for_scene
from nerfsampler.utils import point_cloud
from nerfsampler.utils.edits import *
from nerfsampler.utils.textures import *
from math import sin, cos, pi

import torch
import os
from PIL import Image
import yaml

def run_segmenter(args={}):
    # paths = args["paths"]
    # dl_args = args["data loading"]
    cutoff = 0.75
    scene_id = 2
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    cams = pipeline.datamanager.eval_dataset.cameras
    num_images = len(cams)#pipeline.datamanager.fixed_indices_eval_dataloader)

    class_labels = ['a golden cube', 'a yellow hoop', 'a green cube', 'a brown cube', 'a purple cube', 'a purple cylinder', 'a gray ball', 'a gray floor']
    text_embeddings = TextEmbedder().cuda()(class_labels).T.cpu()
    feature_extractor = FeatureExtractor()
    
    all_coords = []
    all_segs = []
    cams.rescale_output_resolution(640/256)
    for camera_index in [1,2,3,0]:#range(4):#range(num_images):
        camera_ray_bundle = cams.generate_rays(camera_index).to(pipeline.model.device)
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb = outputs['rgb']

        with torch.no_grad():
            pix_embeddings = feature_extractor([rgb], text_embeddings)

        world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']).unsqueeze(0)
        in_range = world_coords.norm(dim=-1) < cutoff
        world_coords = world_coords[in_range]
        all_coords.append(world_coords)
        pix_embeddings = pix_embeddings[in_range]
        sims = F.cosine_similarity(pix_embeddings.unsqueeze(0),
            text_embeddings.unsqueeze(1), dim=-1)
        seg = torch.max(sims, dim=0).indices
        all_segs.append(seg)
    
    save_segs = False
    if save_segs:
        rgb = outputs['rgb']
        Image.fromarray(rgb.cpu().numpy().astype('uint8')).save('results/rgb.png')

        os.makedirs('results/sing_view_segs', exist_ok=True)
        seg = torch.zeros(1,640,640, dtype=torch.long) - 1
        seg[in_range] = torch.max(sims, dim=0).indices
        seg.squeeze_(0)
        for i in range(len(class_labels)):
            Image.fromarray(((seg == i)*255).cpu().numpy().astype('uint8')).save(f'results/sing_view_segs/{class_labels[i]}.png')

    all_coords = torch.cat(all_coords)
    all_segs = torch.cat(all_segs)
    seg_coords = all_coords[all_segs == 1]
    seg_coords = point_cloud.select_largest_subset(seg_coords)
    del all_coords, all_segs, seg, rgb, pix_embeddings, text_embeddings, sims, outputs
    
    # render_with_seg_removed(pipeline.model, camera_ray_bundle, seg_coords)
    # pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    # render_with_seg_recolored(pipeline.model, camera_ray_bundle, seg_coords)
    # pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    # render_with_seg_duplicated(pipeline.model, camera_ray_bundle, seg_coords)
    # render_with_affine_tx(pipeline.model, camera_ray_bundle, seg_coords)

    render_with_procedural_texture(pipeline.model, camera_ray_bundle, seg_coords)
    pdb.set_trace()

    texture_map = Image.open('texture_map.png')
    render_with_texture_map(pipeline.model, camera_ray_bundle, seg_coords, texture_map)
    
    return seg

def render_with_procedural_texture(nerfacto, camera_ray_bundle, coords):
    nerfacto.field.forward = forward_proc_texture(nerfacto.field, coords=coords)
    folder = 'results/proc_tex'
    os.makedirs(folder, exist_ok=True)
    n_frames = 32 #len(ray_bundles)
    for ix in range(n_frames):
        nerfacto.field.frame_frac = ix / n_frames
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb'] #ray_bundles[ix])['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_with_texture_map(nerfacto, camera_ray_bundle, object_coords):
    nerfacto.field.forward = forward_texture_map(nerfacto.field, recolored_coords=recolored_coords)
    os.makedirs('results/recolor', exist_ok=True)
    n_frames = len(ray_bundles)
    for ix in range(n_frames):
        nerfacto.field.frame_frac = ix / n_frames
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(ray_bundles[ix])['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'results/recolor/{ix:02d}.png')


def render_with_seg_recolored(nerfacto, ray_bundles, recolored_coords):
    nerfacto.field.forward = forward_recolor(nerfacto.field, recolored_coords=recolored_coords)
    os.makedirs('results/recolor', exist_ok=True)
    n_frames = len(ray_bundles)
    for ix in range(n_frames):
        nerfacto.field.frame_frac = ix / n_frames
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(ray_bundles[ix])['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'results/recolor/{ix:02d}.png')

def render_with_seg_removed(nerfacto, camera_ray_bundle, removed_coords):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = remove_density_fxn(nerfacto.density_fns[ix], removed_coords=removed_coords)
    nerfacto.field.get_density = remove_field_density(nerfacto.field, removed_coords=removed_coords)
    rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save('results/deletion.png')

def render_with_seg_duplicated(nerfacto, camera_ray_bundle, orig_coords):
    theta = .5
    transform = torch.tensor([[cos(theta), -sin(theta), 0, .1], [sin(theta), cos(theta), 0, .05], [0, 0, 1, .05]], device='cuda')
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = duplicate_density_fxn(nerfacto.density_fns[ix], orig_coords, transform)
    nerfacto.field.get_density = duplicate_field_density(nerfacto.field, orig_coords, transform)
    nerfacto.field.get_outputs = duplicate_rgb(nerfacto.field, orig_coords, transform)
    rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save('results/duplicate.png')

def render_with_affine_tx(nerfacto, camera_ray_bundle, target_coords):
    n_frames = 64
    for ix in range(n_frames):
        t = ix / n_frames * 2 * pi
        R = torch.tensor([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]], device='cuda')
        scale = torch.tensor([[1+t/5, 0, 0], [0, 1+t/5, 0], [0, 0, 1+t/2.5]], device='cuda')
        transform = torch.cat((torch.mm(R, scale), torch.tensor([0,0,-cos(t)/2], device='cuda')))
    pdb.set_trace()
    bbox = target_coords.min(dim=0).values, target_coords.max(dim=0).values
    nerfacto.field.get_density = move_density_fxn(nerfacto.field, target_coords=target_coords, scale=scale, shift=shift, angles=angles)
    transform = torch.eye(4)
    transform[:3,:3] *= scale
    transform[:3,3] += shift
    
    if scale > 1:
        new_bbox = modified_coords.min(dim=0).values * scale + shift, modified_coords.max(dim=0).values * scale + shift

def vis_seg(sims, in_range, class_labels, pix_embeddings):
    seg = torch.zeros(len(class_labels), *pix_embeddings.shape[:3], dtype=torch.long) - 1
    sims.clamp_min_(0) # don't bother showing negative
    seg[:, in_range] = sims * 255
    seg = seg.numpy().astype('uint8')
    for cls in range(len(class_labels)):
        Image.fromarray(seg[cls,0]).save(f'{class_labels[cls]}.png')

        # seg = torch.zeros(*pix_embeddings.shape[:3], dtype=torch.long) - 1
        # seg[in_range] = torch.max(sims, dim=0).indices