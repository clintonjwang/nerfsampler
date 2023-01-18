import os, pdb, torch
import wandb
import yaml

from nerfsampler.utils import losses, util
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

def run_segmenter(args):
    # paths = args["paths"]
    # dl_args = args["data loading"]
    feature_extractor = FeatureExtractor().cuda()
    text_embedder = TextEmbedder().cuda()
    
    pipeline = load_nerf_pipeline_for_scene()
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
    rgb = []
    for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb.append(outputs['rgb'])

    class_labels = ['metallic', 'rubber']
    
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    with torch.autocast('cuda'):
        with torch.no_grad():
            features = [feature_extractor(im.permute(2,0,1)) for im in rgb]
            text_embeddings = torch.stack([text_embedder(label) for label in class_labels])
    return features, text_embeddings