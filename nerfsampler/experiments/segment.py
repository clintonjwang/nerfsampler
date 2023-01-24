import os, pdb, torch
import wandb
import yaml

from nerfsampler.utils import tf_util
from nerfsampler.utils import jobs as job_mgmt
osp = os.path
nn = torch.nn
F = nn.functional
import matplotlib.pyplot as plt

from nerfsampler.utils import args as args_module
from nerfsampler import networks, RESULTS_DIR, DS_DIR
from nerfsampler.networks.clip import TextEmbedder
from nerfsampler.networks.feature_extractor import FeatureExtractor
from nerfsampler.data.nerfacto import load_nerf_pipeline_for_scene

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

    class_labels = ['floating blue cube', 'floating orange cube', 'floating orange ball', 'floating cyan cube', 'floating gray ball', 'brown floor']
    text_embedder = TextEmbedder().cuda()
    feature_extractor = FeatureExtractor().cuda()
    
    all_coords = []
    all_segs = []
    cams.rescale_output_resolution(640/256)
    for camera_index in range(3):#num_images):
        camera_ray_bundle = cams.generate_rays(camera_index).to('cuda')
        # for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb = outputs['rgb']

        with torch.autocast('cuda'):
            with torch.no_grad():
                text_embeddings = text_embedder(class_labels).T
                pix_embeddings = feature_extractor([rgb], text_embeddings)

        world_coords = (camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth']).unsqueeze(0)
        in_range = world_coords.norm(dim=-1) < cutoff
        world_coords = world_coords[in_range]
        all_coords.append(world_coords)
        pix_embeddings = pix_embeddings[in_range]
        sims = F.cosine_similarity(pix_embeddings.unsqueeze(0),
            text_embeddings.unsqueeze(1).cpu(), dim=-1)
        seg = torch.max(sims, dim=0).indices
        all_segs.append(seg)
    
    rgb = outputs['rgb']
    Image.fromarray(rgb.cpu().numpy().astype('uint8')).save('original.png')

    # seg = torch.zeros(1,640,640, dtype=torch.long) - 1
    # seg[in_range] = torch.max(sims, dim=0).indices
    # seg.squeeze_(0)
    # Image.fromarray(((seg == 0)*255).cpu().numpy().astype('uint8')).save('seg0.png')
    # Image.fromarray(((seg == 1)*255).cpu().numpy().astype('uint8')).save('seg1.png')
    # Image.fromarray(((seg == 2)*255).cpu().numpy().astype('uint8')).save('seg2.png')
    # Image.fromarray(((seg == 3)*255).cpu().numpy().astype('uint8')).save('seg3.png')
    # Image.fromarray(((seg == 4)*255).cpu().numpy().astype('uint8')).save('seg4.png')

    all_coords = torch.cat(all_coords)
    all_segs = torch.cat(all_segs)
        
    render_with_seg1_removed(pipeline.model, camera_ray_bundle, all_coords[all_segs == 2])
    pdb.set_trace()
    nerfacto = pipeline.model
    render_with_seg1_moved(camera_ray_bundle)
    render_with_seg1_recolored(camera_ray_bundle)
    
    return seg

def render_with_seg1_removed(nerfacto, camera_ray_bundle, removed_coords):
    for ix in range(2): #len(nerfacto.density_fns)
        nerfacto.density_fns[ix] = modify_density_fxn(nerfacto.density_fns[ix], removed_coords=removed_coords)
    outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    rgb = outputs['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save('rm_orange_ball.png')

from torchtyping import TensorType
def modify_density_fxn(density_fxn, removed_coords, threshold=0.05):
    bbox = removed_coords.min(dim=0).values, removed_coords.max(dim=0).values
    def new_fxn(positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
        # diffs = (removed_coords.view(1,1,-1,3) - positions.unsqueeze(-2))
        # in_range = diffs.norm(dim=-1).min(dim=2) < threshold
        return torch.where(in_range.unsqueeze(-1), 0, density_fxn(positions))
    return new_fxn

# class RemovalRenderer(RGBRenderer):
#     def __init__(self, coords):
#         super().__init__()
#         self.removed_coords = coords
    
#     def forward(self):

# from nerfstudio.cameras.rays import RayBundle
# from nerfstudio.field_components.field_heads import FieldHeadNames
# def get_outputs(self, ray_bundle: RayBundle):
#     ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
#     field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
#     weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
#     weights_list.append(weights)
#     ray_samples_list.append(ray_samples)

#     rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
#     depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
#     accumulation = self.renderer_accumulation(weights=weights)

#     outputs = {
#         "rgb": rgb,
#         "accumulation": accumulation,
#         "depth": depth,
#     }

#     if self.config.predict_normals:
#         outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
#         outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

#     for i in range(self.config.num_proposal_iterations):
#         outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

#     return outputs


def render_with_seg1_moved(camera_ray_bundle):
    pass

def render_with_seg1_recolored(nerfacto, camera_ray_bundle, recolored_coords):
    nerfacto.field.forward = forward_recolor(nerfacto.density_fns[ix], recolored_coords=recolored_coords)
    outputs = nerfacto.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    rgb = outputs['rgb']
    Image.fromarray((rgb * 255).cpu().numpy().astype('uint8')).save('rm_orange_ball.png')
    pass

def forward_recolor(nerfacto, recolored_coords):
    def forward(ray_samples: RaySamples, compute_normals: bool = False):
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = nerfacto.get_density(ray_samples)
        else:
            density, density_embedding = nerfacto.get_density(ray_samples)

        field_outputs = nerfacto.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = nerfacto.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    return forward()


def vis_seg(sims, in_range, class_labels, pix_embeddings):
    seg = torch.zeros(len(class_labels), *pix_embeddings.shape[:3], dtype=torch.long) - 1
    sims.clamp_min_(0) # don't bother showing negative
    seg[:, in_range] = sims * 255
    seg = seg.numpy().astype('uint8')
    for cls in range(len(class_labels)):
        Image.fromarray(seg[cls,0]).save(f'{class_labels[cls]}.png')


        # seg = torch.zeros(*pix_embeddings.shape[:3], dtype=torch.long) - 1
        # seg[in_range] = torch.max(sims, dim=0).indices