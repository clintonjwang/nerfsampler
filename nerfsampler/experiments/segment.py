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

import torch
import os
from PIL import Image
import yaml

def run_segmenter(args={}):
    # paths = args["paths"]
    # dl_args = args["data loading"]
    cutoff = 0.5
    
    pipeline = load_nerf_pipeline_for_scene(scene_id=0)
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    cams = pipeline.datamanager.eval_dataset.cameras
    num_images = len(cams)#pipeline.datamanager.fixed_indices_eval_dataloader)
    rgbs = []
    world_coords = []
    cams.rescale_output_resolution(640/256)
    for camera_index in range(num_images): 
        camera_ray_bundle = cams.generate_rays(camera_index).to('cuda')
        # for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgbs.append(outputs['rgb'])
        world_coords.append(camera_ray_bundle.origins + camera_ray_bundle.directions * outputs['depth'])
        break
    world_coords = torch.stack(world_coords, dim=0)
    
    class_labels = ['floating blue cube', 'floating orange cube', 'floating orange ball', 'floating teal cube']
    os.chdir(os.path.expandvars('$NFS/code/nerfsampler'))
    
    text_embedder = TextEmbedder().cuda()
    feature_extractor = FeatureExtractor().cuda()
    
    with torch.autocast('cuda'):
        with torch.no_grad():
            text_embeddings = text_embedder(class_labels).T
            pix_embeddings = feature_extractor(rgbs, text_embeddings)
    in_range = world_coords.norm(dim=-1) < cutoff
    output = torch.zeros_like(pix_embeddings).repeat(len(class_labels), 1, 1, 1)
    world_coords = world_coords[in_range]
    pix_embeddings = pix_embeddings[in_range]
    pdb.set_trace()
    sims = F.cosine_similarity(pix_embeddings.unsqueeze(0).cuda(),
        text_embeddings.unsqueeze(1), dim=-1)
    sims.clamp_min_(0) # don't bother showing negative
    output[:, in_range] = sims.cpu() * 255
    output = output.numpy().astype('uint8')
    for cls in range(len(class_labels)):
        Image.fromarray(output[cls,0]).save(f'{class_labels[cls]}.png')

    
    pix_embeddings, text_embeddings
    return pix_embeddings
    pdb.set_trace()
    features.shape
    text_embeddings.shape
    feature_size = text_embeddings.size(2)
    return features, text_embeddings


