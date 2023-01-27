import os, pdb, torch
import wandb
import yaml

from torchvision.transforms.functional import pil_to_tensor
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
torch.backends.cudnn.benchmark = True

def animate_camera(camera_ray_bundle, reference, frac):
    t = frac * 2 * pi
    R = torch.tensor([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]], device='cuda')
    camera_ray_bundle.origins = torch.matmul(reference[0], R)
    camera_ray_bundle.directions = torch.matmul(reference[1], R)

def run_segmenter(args={}):
    # paths = args["paths"]
    dl_args = args["data"]
    scene_id = dl_args['scene_id']
    cutoff = args['seg_radius_cutoff']
    native_res = args['native_res']
    class_labels = args['class_labels']
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    cams = pipeline.datamanager.eval_dataset.cameras

    text_embeddings = TextEmbedder().cuda()(class_labels).T.cpu()
    feature_extractor = FeatureExtractor()
    
    all_coords = []
    all_segs = []
    cams.rescale_output_resolution(640/native_res)

    if args['task'] == 'render 2D segmentations':
        camera_ray_bundle = cams.generate_rays(0).to(pipeline.model.device)
        for label in class_labels:
            os.makedirs(f'results/segs/{label}', exist_ok=True)

        n_frames = 64
        orig = torch.clone(camera_ray_bundle.origins)
        dirs = torch.clone(camera_ray_bundle.directions)
        for frame in range(n_frames):
            animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            rgb = outputs['rgb']
            
            with torch.no_grad():
                pix_embeddings = feature_extractor([rgb], text_embeddings)

            world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']).unsqueeze(0)
            in_range = world_coords.norm(dim=-1) < cutoff
            pix_embeddings = pix_embeddings[in_range]
            sims = F.cosine_similarity(pix_embeddings.unsqueeze(0),
                text_embeddings.unsqueeze(1), dim=-1)
            
            seg = torch.zeros(1,640,640, dtype=torch.long) - 1
            seg[in_range] = torch.max(sims, dim=0).indices
            seg.squeeze_(0)
            for i in range(len(class_labels)):
                Image.fromarray(((seg == i)*255).cpu().numpy().astype('uint8')).save(f'results/segs/{class_labels[i]}/{frame:03d}.png')
            
        return

    for camera_index in range(len(args['num_views'])):
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
    
    folder = 'results/original'
    os.makedirs(folder, exist_ok=True)

    if args['task'] == 'render original scene':
        n_frames = 64
        orig = torch.clone(camera_ray_bundle.origins)
        dirs = torch.clone(camera_ray_bundle.directions)
        for ix in range(n_frames):
            animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
            
            rgb = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
            Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')
        return

    all_coords = torch.cat(all_coords)
    all_segs = torch.cat(all_segs)
    seg_coords = all_coords[all_segs == 1]
    seg_coords = point_cloud.select_largest_subset(seg_coords)
    del all_coords, all_segs, seg, rgb, pix_embeddings, text_embeddings, sims, outputs

    print('Starting render')
    render_with_seg_removed(pipeline.model, camera_ray_bundle, seg_coords)
    pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    render_with_seg_recolored(pipeline.model, camera_ray_bundle, seg_coords)
    render_with_seg_duplicated(pipeline.model, camera_ray_bundle, seg_coords)
    render_with_affine_tx(pipeline.model, camera_ray_bundle, seg_coords)
    render_with_procedural_texture(pipeline.model, camera_ray_bundle, seg_coords)

    texture_map = pil_to_tensor(Image.open('texture_map.jpeg')).cuda()
    render_with_texture_map(pipeline.model, camera_ray_bundle, seg_coords, texture_map)
    

def render_with_procedural_texture(nerfacto, camera_ray_bundle, coords):
    nerfacto.field.forward = forward_proc_texture(nerfacto.field, coords=coords)
    folder = 'results/proc_tex'
    os.makedirs(folder, exist_ok=True)
    
    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        nerfacto.field.t = ix / n_frames * 2 * pi
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_with_texture_map(nerfacto, camera_ray_bundle, coords, texture_map):
    nerfacto.field.forward = forward_uv_map(nerfacto.field, coords=coords, texture=texture_map)
    folder = 'results/texture_map'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        nerfacto.field.t = ix / n_frames * 2 * pi
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')


def render_with_seg_recolored(nerfacto, camera_ray_bundle, recolored_coords):
    nerfacto.field.forward = forward_recolor(nerfacto.field, recolored_coords=recolored_coords)
    folder = 'results/recolor'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        nerfacto.field.frame_frac = ix/n_frames
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_with_seg_removed(nerfacto, camera_ray_bundle, coords):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], coords)
    nerfacto.field.get_density = attenuate_field_density(nerfacto.field, coords)
    folder = 'results/deletion'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_with_seg_duplicated(nerfacto, camera_ray_bundle, orig_coords):
    theta = .5
    transform = torch.tensor([[cos(theta), -sin(theta), 0, .1], [sin(theta), cos(theta), 0, .05], [0, 0, 1, .05]], device='cuda')
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = duplicate_density_fxn(nerfacto.density_fns[ix], orig_coords, transform)
    nerfacto.field.get_density = duplicate_field_density(nerfacto.field, orig_coords, transform)
    nerfacto.field.get_outputs = duplicate_rgb(nerfacto.field, orig_coords, transform)
    folder = 'results/duplicate'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)

        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_reflective_material(nerfacto, camera_ray_bundle, coords):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], scale=0.1, coords=coords)
    nerfacto.field.get_density = attenuate_field_density(nerfacto.field, scale=0.1, coords=coords)
    nerfacto.field.get_outputs = reflect_rgb(nerfacto.field, coords)
    folder = 'results/reflection'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')

def render_with_affine_tx(nerfacto, camera_ray_bundle, orig_coords):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = animate_density_fxn(nerfacto.field, nerfacto.density_fns[ix], orig_coords)
    nerfacto.field.get_density = animate_field_density(nerfacto.field, orig_coords)
    nerfacto.field.get_outputs = animate_rgb(nerfacto.field, orig_coords)
    folder = 'results/animate'
    os.makedirs(folder, exist_ok=True)

    n_frames = 64
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for ix in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=ix/n_frames)
        
        t = ix/n_frames * 2 * pi
        R = torch.tensor([[cos(2*t), -sin(2*t), 0], [sin(2*t), cos(2*t), 0], [0, 0, 1]], device='cuda')
        scale = torch.tensor([[1-sin(t)*.4, 0, 0], [0, 1+sin(t)*.4, 0], [0, 0, 1-cos(t)*.4]], device='cuda')
        nerfacto.field.animation_transform = torch.cat((torch.mm(R, scale), torch.tensor([0,0,cos(t)*.07], device='cuda').unsqueeze(1)), dim=1)
        nerfacto.field.inv_tx = torch.inverse(nerfacto.field.animation_transform[:,:3])
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'{folder}/{ix:02d}.png')


def vis_seg(sims, in_range, class_labels, pix_embeddings):
    # os.makedirs('results/sing_view_segs', exist_ok=True)
    # seg = torch.zeros(1,640,640, dtype=torch.long) - 1
    # seg[in_range] = torch.max(sims, dim=0).indices
    # seg.squeeze_(0)
    # for i in range(len(class_labels)):
    #     Image.fromarray(((seg == i)*255).cpu().numpy().astype('uint8')).save(f'results/sing_view_segs/{class_labels[i]}.png')
    
    seg = torch.zeros(len(class_labels), *pix_embeddings.shape[:3], dtype=torch.long) - 1
    sims.clamp_min_(0) # don't bother showing negative
    seg[:, in_range] = sims * 255
    seg = seg.numpy().astype('uint8')
    for cls in range(len(class_labels)):
        Image.fromarray(seg[cls,0]).save(f'{class_labels[cls]}.png')

        # seg = torch.zeros(*pix_embeddings.shape[:3], dtype=torch.long) - 1
        # seg[in_range] = torch.max(sims, dim=0).indices