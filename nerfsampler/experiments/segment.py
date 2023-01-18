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

from easydict import EasyDict

import numpy as np
import torch

from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

import os

from PIL import Image
from pprint import pprint
from scipy.special import softmax
import yaml

import tensorflow.compat.v1 as tf
import tensorflow as tf2
import ipywidgets as widgets

def run_segmenter(args={}):
    # paths = args["paths"]
    # dl_args = args["data loading"]
    pipeline = load_nerf_pipeline_for_scene()
    pipeline.datamanager.setup_train()
    pipeline.datamanager.setup_eval()
    num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
    rgb = []
    # with torch.autocast('cuda'):
    for camera_ray_bundle, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        rgb.append(outputs['rgb'])
    
    class_labels = ['metallic', 'rubber']
    os.chdir(os.path.expandvars('$NFS/code/nerfsampler'))
    
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = FeatureExtractor().cuda()
    text_embedder = TextEmbedder().cuda()
    
    with torch.autocast('cuda'):
        with torch.no_grad():
            text_embeddings = text_embedder(class_labels)
            features = [feature_extractor(im.permute(2,0,1).cuda(), text_embeddings) for im in rgb]
    pdb.set_trace()
    features.shape
    text_embeddings.shape
    feature_size = text_embeddings.size(2)
    return features, text_embeddings