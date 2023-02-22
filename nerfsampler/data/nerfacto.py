import os, json
import shutil
osp = os.path
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup
from nerfsampler import CODE_DIR
from nerfsampler.utils.util import glob2

def get_scene_ids():
    return os.listdir(f'{CODE_DIR}/kubric/outputs')

def load_nerf_pipeline_for_scene(scene_id=0):
    os.chdir(f'{CODE_DIR}/kubric')
    if not osp.exists(f"{CODE_DIR}/kubric/outputs/{scene_id}/nerfacto"):
        raise ValueError(f"{scene_id=} not found in {CODE_DIR}/kubric/outputs")
    path = glob2(f'{CODE_DIR}/kubric/outputs/{scene_id}/nerfacto/*/config.yml')[0]
    # path = glob2(f'{DS_DIR}/nerfacto/{scene_id}/nerfacto/*/config.yml')[0]
    config, pipeline, checkpoint_path = eval_setup(Path(path), test_mode="inference")
    pipeline.eval()
    return pipeline
