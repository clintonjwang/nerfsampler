import os, pdb, torch
import shutil
osp = os.path
nn = torch.nn
F = nn.functional

from torchvision.transforms.functional import pil_to_tensor
from nerfstudio.cameras.cameras import Cameras

from nerfsampler import CODE_DIR
from nerfsampler.utils.camera import animate_camera
from nerfsampler.networks.clip import TextEmbedder
from nerfsampler.networks.feature_extractor import FeatureExtractor
from nerfsampler.data.nerfacto import load_nerf_pipeline_for_scene
from nerfsampler.utils import point_cloud, mesh
from nerfsampler.experiments.render import *

import numpy as np
import os
from PIL import Image
torch.backends.cudnn.benchmark = True

def setup(args):
    # paths = args["paths"]
    dl_args = args["data"]
    scene_id = dl_args['scene_id']
    cutoff = args['seg_radius_cutoff']
    n_frames = args['n_frames']
    use_gt_segs = True
    resolution = 512 if use_gt_segs else 640
    base_frame = 44 if use_gt_segs else 0
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    if use_gt_segs:
        pipeline.datamanager.setup_train()
        cams = pipeline.datamanager.train_dataset.cameras
    else:
        pipeline.datamanager.train_camera_optimizer = None
        pipeline.datamanager.setup_eval()
        cams = pipeline.datamanager.eval_dataset.cameras
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

    if args['task'] == 'render RGB':
        cams.rescale_output_resolution(resolution/native_res)
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

    if use_gt_segs:
        frames = [0,25,48,61,99]#,131,164,189,210]
        seg_coords, seg_2d = load_gt_segs(pipeline.model, cams, args, frames)
        cams.rescale_output_resolution(512/native_res)

        if args['task'] == 'render segs':
            # for cls in range(1,7):
            folder = 'results/segs'
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)
            for ix,frame in enumerate(frames):
                Image.fromarray(np.array(seg_2d[ix])).save(f'{folder}/{frame:02d}.png')
            return

        camera_ray_bundle = cams.generate_rays(base_frame).to(pipeline.model.device)
    
        seg_coords = seg_coords[seg_coords.norm(dim=-1) < cutoff]
        seg_coords = point_cloud.select_largest_subset(seg_coords, k=5)
    else:
        class_labels = dl_args['class_labels']
        text_embeddings = TextEmbedder().cuda()(class_labels).T.cpu()
        feature_extractor = FeatureExtractor()
        
        cams.rescale_output_resolution(640/native_res)
                
        if args['task'] == 'render segs':
            folder = 'results/segs'
            shutil.rmtree(folder, ignore_errors=True)
            for label in class_labels:
                os.makedirs(f'{folder}/{label}')

            camera_ray_bundle = cams.generate_rays(0).to(pipeline.model.device)
            orig = torch.clone(camera_ray_bundle.origins)
            dirs = torch.clone(camera_ray_bundle.directions)

            for frame in range(n_frames):
                animate_camera(camera_ray_bundle, (orig, dirs), frac=frame/n_frames)
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                rgb = outputs['rgb']
                
                with torch.no_grad():
                    pix_embeddings = feature_extractor([rgb], text_embeddings)

                world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']).unsqueeze(0)
                in_range = world_coords.norm(dim=-1) < cutoff
                pix_embeddings = pix_embeddings[in_range]
                sims = F.cosine_similarity(pix_embeddings.unsqueeze(0),
                    text_embeddings.unsqueeze(1), dim=-1)
                
                seg = torch.zeros(1,resolution,resolution, dtype=torch.long) - 1
                seg[in_range] = torch.max(sims, dim=0).indices
                seg.squeeze_(0)
                for i in range(len(class_labels)):
                    Image.fromarray(((seg == i)*255).cpu().numpy().astype('uint8')).save(f'{folder}/{class_labels[i]}/{frame:03d}.png')
                
            return

        all_coords = []
        all_segs = []

        for camera_index in range(args['num_views']):
            camera_ray_bundle = cams.generate_rays(camera_index).to(pipeline.model.device)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            with torch.no_grad():
                pix_embeddings = feature_extractor([outputs['rgb']], text_embeddings)

            world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']).unsqueeze(0)
            in_range = world_coords.norm(dim=-1) < cutoff
            all_coords.append(world_coords[in_range])
            sims = F.cosine_similarity(pix_embeddings[in_range].unsqueeze(0),
                text_embeddings.unsqueeze(1), dim=-1)
            all_segs.append(torch.max(sims, dim=0).indices)
    
        all_coords = torch.cat(all_coords)
        all_segs = torch.cat(all_segs)
        seg_coords = all_coords[all_segs == 1]
        seg_coords = point_cloud.select_largest_subset(seg_coords)

    return pipeline.model, camera_ray_bundle, seg_coords

def run_segmenter(args={}):
    ret = setup(args)
    if ret is None:
        return
    nerfacto, base_camera_rays, seg_coords = ret
    # seg_mesh = mesh.pcd_to_mesh(seg_coords)
    
    # import matplotlib.pyplot as plt
    # point_cloud.plot_point_cloud(seg_coords.cpu())
    # plt.savefig('results/seg_coords.png')
    n_frames = args['n_frames']

    print('Starting render')
    if args['edit_task'] == 'delete':
        render_with_seg_removed(nerfacto, base_camera_rays, seg_coords, n_frames)
        # pipeline = load_nerf_pipeline_for_scene(scene_id=scene_id)
    elif args['edit_task'] == 'recolor':
        render_with_seg_recolored(nerfacto, base_camera_rays, seg_coords, n_frames)
    elif args['edit_task'] == 'duplicate':
        render_with_seg_duplicated(nerfacto, base_camera_rays, seg_coords, n_frames)
    elif args['edit_task'] == 'animate':
        render_with_affine_tx(nerfacto, base_camera_rays, seg_coords, n_frames)
    elif args['edit_task'] == 'procedural_texture':
        render_with_procedural_texture(nerfacto, base_camera_rays, seg_coords, n_frames)
    elif args['edit_task'] == 'texture_map':
        texture_map = pil_to_tensor(Image.open('texture_map.jpeg')).cuda()
        render_with_texture_map(nerfacto, base_camera_rays, seg_coords, texture_map, n_frames)
    
from skimage.morphology import binary_erosion
def load_gt_segs(nerfacto, cams, args, frames):
    dl_args = args["data"]
    scene_id = dl_args['scene_id']
    if scene_id.startswith('gso'):
        scene_num = int(scene_id[3:])
    base_dir = osp.expandvars(f'$DS_DIR/kubric/gso/{scene_num:04d}')
    
    segs_2d = []
    seg_3d = []
    for frame in frames:
        camera_ray_bundle = cams.generate_rays(frame).to(nerfacto.device)
        outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb = outputs['rgb']
        world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth'])
        img = (rgb * 255).cpu().numpy()
        gt = np.array(Image.open(osp.join(base_dir, f'rgba_{frame:04d}.png')).convert('RGB')).astype(float)
        diff = ((np.abs(img - gt))/(img+.1)).mean()
        ix = frame
        while diff > 0.2:
            ix += 1
            gt = np.array(Image.open(osp.join(base_dir, f'rgba_{ix:04d}.png')).convert('RGB')).astype(float)
            diff = ((np.abs(img - gt))/(img+.1)).mean()
        seg = np.array(Image.open(osp.join(base_dir, f'segmentation_{ix:04d}.png'))) == args['target_seg']
        seg = torch.tensor(binary_erosion(seg))
        seg_3d.append(world_coords[seg])
        segs_2d.append(seg)

    return torch.cat(seg_3d, dim=0), segs_2d
