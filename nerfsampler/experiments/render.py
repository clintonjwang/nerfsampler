import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional

from nerfsampler.utils.camera import animate_camera
from nerfsampler.utils.edits import *
from nerfsampler.utils.textures import *
from nerfsampler.utils import mesh, point_cloud
from math import sin, cos, pi

from PIL import Image

def render_with_seg_fit_inpaint(nerfacto, camera_ray_bundle, inpainting, seg, n_frames=64):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], seg)
    nerfacto.field.get_density = attenuate_field_density(nerfacto.field, seg)
    outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    inpaint_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth'])
    in_range = inpaint_coords.norm(dim=-1) < 1
    rgb = outputs['rgb'][in_range]
    inpaint_coords = inpaint_coords[in_range]
    inpaint_scale = inpainting[in_range] / (rgb * 255)
    nerfacto.field.forward = forward_inpaint(nerfacto.field,
        inpaint_scale, inpaint_coords)
    
    folder = 'results/fit_inpaint'
    os.makedirs(folder, exist_ok=True)
    
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        nerfacto.field.t = frame / n_frames * 2 * pi
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_with_procedural_texture(nerfacto, camera_ray_bundle, seg, n_frames=64):
    nerfacto.field.forward = forward_proc_texture(nerfacto.field, seg=seg)
    folder = 'results/proc_tex'
    os.makedirs(folder, exist_ok=True)
    
    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        nerfacto.field.t = frame / n_frames * 2 * pi
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_with_texture_map(nerfacto, camera_ray_bundle, seg, texture_map, n_frames=64):
    nerfacto.field.forward = forward_uv_map(nerfacto.field, seg=seg, texture=texture_map)
    folder = 'results/texture_map'
    os.makedirs(folder, exist_ok=True)

    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        nerfacto.field.t = frame / n_frames * 2 * pi
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')


def render_with_seg_recolored(nerfacto, camera_ray_bundle, recolored_seg, n_frames=64):
    nerfacto.field.forward = forward_recolor(nerfacto.field, recolored_seg=recolored_seg)
    folder = 'results/recolor'
    os.makedirs(folder, exist_ok=True)

    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        nerfacto.field.frame_frac = frame/n_frames
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_with_seg_removed(nerfacto, camera_ray_bundle, seg, n_frames=64):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], seg)
    nerfacto.field.get_density = attenuate_field_density(nerfacto.field, seg)
    folder = 'results/deletion'
    os.makedirs(folder, exist_ok=True)

    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_with_seg_duplicated(nerfacto, camera_ray_bundle, orig_seg, n_frames=64):
    theta = .5
    transform = torch.tensor([[cos(theta), -sin(theta), 0, .1], [sin(theta), cos(theta), 0, .05], [0, 0, 1, .05]], device=camera_ray_bundle.origins.device)
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = duplicate_density_fxn(nerfacto.density_fns[ix], orig_seg, transform)
    nerfacto.field.get_density = duplicate_field_density(nerfacto.field, orig_seg, transform)
    nerfacto.field.get_outputs = duplicate_rgb(nerfacto.field, orig_seg, transform)
    folder = 'results/duplicate'
    os.makedirs(folder, exist_ok=True)

    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_reflective_material(nerfacto, camera_ray_bundle, seg, n_frames=64):
    # for ix in range(len(nerfacto.density_fns)):
    #     nerfacto.density_fns[ix] = attenuate_density_fxn(nerfacto.density_fns[ix], scale=0.1, seg=seg)
    # nerfacto.field.get_density = attenuate_field_density(nerfacto.field, scale=0.1, seg=seg)
    # nerfacto.field.get_outputs = reflect_rgb(nerfacto.field, seg)
    # folder = 'results/reflection'
    # os.makedirs(folder, exist_ok=True)
    pass
    # orig = torch.clone(camera_ray_bundle.origins)
    # dirs = torch.clone(camera_ray_bundle.directions)
    # for frame in range(n_frames):
    #     animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
    #     generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def render_with_affine_tx(nerfacto, camera_ray_bundle, orig_seg, n_frames=64):
    for ix in range(len(nerfacto.density_fns)):
        nerfacto.density_fns[ix] = animate_density_fxn(nerfacto.field, nerfacto.density_fns[ix], orig_seg)
    nerfacto.field.get_density = animate_field_density(nerfacto.field, orig_seg)
    nerfacto.field.get_outputs = animate_rgb(nerfacto.field, orig_seg)
    folder = 'results/animate'
    os.makedirs(folder, exist_ok=True)
    device = camera_ray_bundle.origins.device

    orig = torch.clone(camera_ray_bundle.origins)
    dirs = torch.clone(camera_ray_bundle.directions)
    for frame in range(n_frames):
        animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
        
        t = frame/n_frames * 2 * pi
        R = torch.tensor([[cos(2*t), -sin(2*t), 0], [sin(2*t), cos(2*t), 0], [0, 0, 1]], device=device)
        scale = torch.tensor([[1-sin(t)*.4, 0, 0], [0, 1+sin(t)*.4, 0], [0, 0, 1-cos(t)*.4]], device=device)
        nerfacto.field.animation_transform = torch.cat((
            torch.mm(R, scale), torch.tensor([0,0,cos(t)*.07], device=device).unsqueeze(1)), dim=1)
        nerfacto.field.inv_tx = torch.inverse(nerfacto.field.animation_transform[:,:3])
        generate_view(nerfacto, camera_ray_bundle, f'{folder}/{frame:02d}.png')

def generate_view(nerfacto, camera_ray_bundle, path):
    rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(path)


def vis_seg(sims, in_range, class_labels, pix_embeddings):
    # os.makedirs('results/sing_view_segs', exist_ok=True)
    # seg = torch.zeros(1,resolution,resolution, dtype=torch.long) - 1
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
        # seg[in_range] = torch.max(sims, dim=0).indices3