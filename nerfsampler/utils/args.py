"""
Argument parsing
"""
import argparse
import os
import wandb
import yaml
from glob import glob
from nerfsampler import CONFIG_DIR

osp = os.path

def parse_args(args):
    """Command-line args for train.sh, infer.sh and sweep.sh"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-s', '--sweep_id', default=None)
    parser.add_argument('-e', '--edit_type', default=None)
    parser.add_argument('-d', '--device_id', default=0)
    cmd_args = parser.parse_args(args)
    os.environ['CUDA_VISIBLE_DEVICES']=str(cmd_args.device_id)

    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    configs = glob(osp.join(CONFIG_DIR, "*", config_name+".yaml"))
    if len(configs) == 0:
        raise ValueError(f"{config_name=} not found in {CONFIG_DIR}")
    main_config_path = configs[0]

    args = args_from_file(main_config_path, cmd_args)
    # paths = args["paths"]
    # for subdir in ("weights", "imgs"):
    #     shutil.rmtree(osp.join(paths["job output dir"], subdir), ignore_errors=True)
    # for subdir in ("weights", "imgs"):
    #     os.makedirs(osp.join(paths["job output dir"], subdir))

    return args

def get_wandb_config():
    wandb_sweep_dict = {
        'kernel_size_0': ['network', 'conv', 'k0'],
        'kernel_size_1': ['network', 'conv', 'k1'],
        'kernel_size_2': ['network', 'conv', 'k2'],
        'kernel_size_3': ['network', 'conv', 'k3'],
    }
    if wandb.config['sweep_id'] is not None:
        for k,subkeys in wandb_sweep_dict.items():
            if k in wandb.config:
                d = wandb.config
                for subk in subkeys[:-1]:
                    d = d[subk]
                d[subkeys[-1]] = wandb.config[k]
    wandb.config.persist()
    args = dict(wandb.config.items())
    yaml.safe_dump(args, open(osp.join(args['paths']["job output dir"], "config.yaml"), 'w'))
    return args

def merge_args(parent_args, child_args):
    """Merge parent config args into child configs."""
    if "_overwrite_" in child_args.keys():
        return child_args
    for k,parent_v in parent_args.items():
        if k not in child_args.keys():
            child_args[k] = parent_v
        else:
            if isinstance(child_args[k], dict) and isinstance(parent_v, dict):
                child_args[k] = merge_args(parent_v, child_args[k])
    return child_args


def args_from_file(path, cmd_args=None):
    """Create args dict from yaml."""
    if osp.exists(path):
        args = yaml.safe_load(open(path, 'r'))
    else:
        raise ValueError(f"bad config_name {cmd_args.config_name}")

    if cmd_args is not None:
        for param in ["job_id", "config_name", 'edit_type', 'sweep_id']:
            if hasattr(cmd_args, param):
                args[param] = getattr(cmd_args, param)

    while "parent" in args:
        if isinstance(args["parent"], str):
            config_path = glob(osp.join(CONFIG_DIR, "*", args.pop("parent")+".yaml"))[0]
            args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
        else:
            parents = args.pop("parent")
            for p in parents:
                config_path = glob(osp.join(CONFIG_DIR, "*", p+".yaml"))[0]
                args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
            if "parent" in args:
                raise NotImplementedError("cannot handle multiple parents each with other parents")

    config_path = osp.join(CONFIG_DIR, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)

    args["config path"] = path
    label_dict = yaml.safe_load(
        open(osp.join(CONFIG_DIR,'class_labels.yaml'), 'r'))
    if 'data' in args and 'class_labels' not in args['data'] and args['data']['scene_id'] in label_dict:
        args['data']['class_labels'] = label_dict[args['data']['scene_id']]

    return args