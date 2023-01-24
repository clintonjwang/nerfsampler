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
# from nerfsampler.utils import point_cloud

from nerfstudio.model_components.renderers import RGBRenderer

import torch
import os
from PIL import Image
import yaml

def run_segmenter(args={}):
    # paths = args["paths"]
    # dl_args = args["data loading"]
    cutoff = 0.75
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=0)
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    cams = pipeline.datamanager.eval_dataset.cameras
    num_images = len(cams)#pipeline.datamanager.fixed_indices_eval_dataloader)

    class_labels = ['floating blue cube', 'floating orange cube', 'small orange ball', 'floating cyan cube', 'large gray ball', 'brown floor']
    text_embeddings = TextEmbedder().cuda()(class_labels).T.cpu()
    feature_extractor = FeatureExtractor()
    
    all_coords = []
    all_segs = []
    cams.rescale_output_resolution(640/256)
    for camera_index in [2,0,1]:#range(num_images):
        camera_ray_bundle = cams.generate_rays(camera_index).to('cuda')
        # for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
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
    
    rgb = outputs['rgb']
    Image.fromarray(rgb.cpu().numpy().astype('uint8')).save('results/rgb.png')

    save_segs = True
    if save_segs:
        seg = torch.zeros(1,640,640, dtype=torch.long) - 1
        seg[in_range] = torch.max(sims, dim=0).indices
        seg.squeeze_(0)
        Image.fromarray(((seg == 0)*255).cpu().numpy().astype('uint8')).save('results/seg0.png')
        Image.fromarray(((seg == 1)*255).cpu().numpy().astype('uint8')).save('results/seg1.png')
        Image.fromarray(((seg == 2)*255).cpu().numpy().astype('uint8')).save('results/seg2.png')
        Image.fromarray(((seg == 3)*255).cpu().numpy().astype('uint8')).save('results/seg3.png')
        Image.fromarray(((seg == 4)*255).cpu().numpy().astype('uint8')).save('results/seg4.png')

    all_coords = torch.cat(all_coords)
    all_segs = torch.cat(all_segs)
    seg_coords = all_coords[all_segs == 2]
    del all_coords, all_segs, seg, rgb, pix_embeddings, text_embeddings, sims, outputs
    
    render_with_seg_removed(pipeline.model, camera_ray_bundle, seg_coords)
    pipeline = load_nerf_pipeline_for_scene(scene_id=0)
    render_with_seg_recolored(pipeline.model, camera_ray_bundle, seg_coords)
    # pipeline = load_nerf_pipeline_for_scene(scene_id=0)
    pdb.set_trace()
    render_with_scale_shift(pipeline.model, camera_ray_bundle, seg_coords)
    render_with_texture(pipeline.model, camera_ray_bundle, seg_coords)
    
    return seg

def render_with_seg_recolored(nerfacto, camera_ray_bundle, recolored_coords, n_frames=32):
    nerfacto.field.forward = forward_recolor(nerfacto.field, recolored_coords=recolored_coords)
    os.makedirs('results/recolor_ball', exist_ok=True)
    for ix in range(n_frames):
        nerfacto.field.frame_frac = ix / n_frames
        rgb = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)['rgb']
        Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save(f'results/recolor_ball/{ix}.png')

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
def forward_recolor(self, recolored_coords, eps=0.05):
    bbox = recolored_coords.min(dim=0).values, recolored_coords.max(dim=0).values
    def forward(ray_samples: RaySamples, compute_normals: bool = False):
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        positions = ray_samples.frustums.get_positions()
        positions = self.spatial_distortion(positions)
        #in_range = (self._sample_locations > bbox[0]).min(dim=-1).values & (self._sample_locations < bbox[1]).min(dim=-1).values
        in_range = (positions > bbox[0]-eps).min(dim=-1).values & (positions < bbox[1]+eps).min(dim=-1).values
        rgb = field_outputs[FieldHeadNames.RGB]
        if self.frame_frac < 1/3:
            alpha = self.frame_frac * 3
            new_rgb = rgb * (1-alpha) + torch.stack((rgb[...,2], rgb[...,0], rgb[...,1]), dim=-1) * alpha
        elif self.frame_frac < 2/3:
            alpha = self.frame_frac * 3 - 1
            new_rgb = torch.stack((rgb[...,2], rgb[...,0], rgb[...,1]), dim=-1) * (1-alpha) + \
                    torch.stack((rgb[...,1], rgb[...,2], rgb[...,0]), dim=-1) * alpha
        else:
            alpha = self.frame_frac * 3 - 2
            new_rgb = torch.stack((rgb[...,1], rgb[...,2], rgb[...,0]), dim=-1) * (1-alpha) + \
                    rgb * alpha
        field_outputs[FieldHeadNames.RGB] = torch.where(in_range.unsqueeze(-1), new_rgb, rgb)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    return forward

def render_with_seg_removed(nerfacto, camera_ray_bundle, removed_coords):
    for ix in range(2): #len(nerfacto.density_fns)
        nerfacto.density_fns[ix] = modify_density_fxn(nerfacto.density_fns[ix], removed_coords=removed_coords)
    outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    rgb = outputs['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save('results/rm_orange_ball.png')

from torchtyping import TensorType
def modify_density_fxn(density_fxn, removed_coords, eps=0.05):
    bbox = removed_coords.min(dim=0).values, removed_coords.max(dim=0).values
    def new_fxn(positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        in_range = (positions > bbox[0]-eps).min(dim=-1).values & (positions < bbox[1]+eps).min(dim=-1).values
        # diffs = (removed_coords.view(1,1,-1,3) - positions.unsqueeze(-2))
        # in_range = diffs.norm(dim=-1).min(dim=2) < threshold
        return torch.where(in_range.unsqueeze(-1), 0, density_fxn(positions))
    return new_fxn

def render_with_scale_shift(nerfacto, camera_ray_bundle, modified_coords, scale, shift):
    pass

def render_with_texture(nerfacto, camera_ray_bundle, texture_coords):
    pass

def vis_seg(sims, in_range, class_labels, pix_embeddings):
    seg = torch.zeros(len(class_labels), *pix_embeddings.shape[:3], dtype=torch.long) - 1
    sims.clamp_min_(0) # don't bother showing negative
    seg[:, in_range] = sims * 255
    seg = seg.numpy().astype('uint8')
    for cls in range(len(class_labels)):
        Image.fromarray(seg[cls,0]).save(f'{class_labels[cls]}.png')

        # seg = torch.zeros(*pix_embeddings.shape[:3], dtype=torch.long) - 1
        # seg[in_range] = torch.max(sims, dim=0).indices