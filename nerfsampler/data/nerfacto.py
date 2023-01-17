import os, json
import shutil
osp = os.path
from pathlib import Path

from nerfsampler import CODE_DIR, DS_DIR
from nerfsampler.utils.util import glob2
from nerfstudio.utils.eval_utils import eval_setup

def load_nerfacto_for_scene(scene_id=0):
    path = glob2(f'{DS_DIR}/nerfacto/{scene_id}/nerfacto/*/config.yml')[0]
    config, pipeline, checkpoint_path = eval_setup(Path(path), test_mode="inference")
    return config, pipeline, checkpoint_path

# def clean_kubric():
#     """
#     Clean the kubric directory.
#     """
#     subdirs = glob2(CODE_DIR, "/kubric/output")
#     for subdir in subdirs:
#         if osp.isdir(subdir) and "rgba_00002.png" not in os.listdir(subdir):
#             shutil.rmtree(subdir)

# def tmp():
#     subdirs = glob2(CODE_DIR, "/kubric/output")
#     for subdir in subdirs:
#         if osp.isdir(subdir):
#             path = subdir+"/transforms.json"
#             if osp.exists(path):
#                 os.rename(path, path.replace("transforms.json",
#                     "transforms_train.json"))
    # src = CODE_DIR+"/kubric/output/39/transforms_test.json"
    # for ix in range(39):
    #     target = CODE_DIR+f"/kubric/output/{ix}/transforms_test.json"
    #     shutil.copy(src, target)

#import nerfsampler.models.kubric as K
#K.clean_kubric()
#K.tmp()
# def get_kubric_paths():
#     data_dir = DS_DIR+'/kubric-public/data'
#     train_pattern = data_dir+'/multiview_matting/*/train/*'
#     train_paths = glob2(train_pattern)
#     val_pattern = data_dir+'/multiview_matting/*/val/*'
#     val_paths = glob2(val_pattern)
#     return train_paths, val_paths

# def get_data():
#     train_paths, val_paths = get_kubric_paths()
#     for path in train_paths:
#         data_json = osp.join(path, 'metadata.json')
#         with open(data_json) as f:
#             data = json.load(f)
#         camera = data['camera']
#         break