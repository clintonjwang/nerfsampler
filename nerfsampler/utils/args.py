"""
Argument parsing
"""
import argparse
import os
import shutil
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
    parser.add_argument('-t', '--target_job', default=None)
    parser.add_argument('-s', '--sweep_id', default=None)
    parser.add_argument('-w', '--no_wandb', action='store_true')
    cmd_args = parser.parse_args(args)

    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    configs = glob(osp.join(CONFIG_DIR, "*", config_name+".yaml"))
    if len(configs) == 0:
        # if config_name.startswith('seg_i3'):
        #     discs = extract(config_name)
        #     train_disc = config_name.split('i3')[1]
        #     configs = glob(osp.join(CONFIG_DIR, "*", "seg_i3.yaml"))
        raise ValueError(f"Config {config_name} not found in {CONFIG_DIR}")
    main_config_path = configs[0]

    args = args_from_file(main_config_path, cmd_args)
    # if config_name.startswith('seg_i3'):
    #     args["test discretization"]
    paths = args["paths"]
    for subdir in ("weights", "imgs"):
        shutil.rmtree(osp.join(paths["job output dir"], subdir), ignore_errors=True)
    for subdir in ("weights", "imgs"):
        os.makedirs(osp.join(paths["job output dir"], subdir))

    return args

def parse_args_fitting(args):
    """Command-line args for fit_inr.sh"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-s', '--sweep_id', default=None)
    parser.add_argument('-t', '--start_ix', default=0, type=int)
    cmd_args = parser.parse_args(args)

    if cmd_args.dataset in ('fetal', 'echonet'):
        args = args_from_file(CONFIG_DIR+f'/fit_video/{cmd_args.dataset}.yaml', cmd_args)
        os.makedirs(args['paths']["job output dir"], exist_ok=True)
    else:
        args = yaml.safe_load(open(CONFIG_DIR+'/fit_config.yaml', 'r'))[cmd_args.dataset]
        for param in ["job_id", 'start_ix', 'sweep_id']:
            args[param] = getattr(cmd_args, param)
    args['data loading']['dataset'] = cmd_args.dataset
    return args

def get_wandb_train_config():
    wandb_sweep_dict = {
        'learning_rate': ['optimizer', 'learning rate'],
        'batch_size': ['data loading', 'batch size'],
        'sample_points': ['data loading', 'sample points'],
        'discretization_type': ['data loading', 'discretization type'],
        'augment': ['data loading', 'augment'],
        'weight_decay': ['optimizer', 'weight decay'],
        'optimizer_type': ['optimizer', 'type'],

        'num_layers': ['network', 'depth'],
        'num_channels': ['network', 'channels'],
        # 'inter_grid_size': ['network', 'intermediate grid size'],
        
        'kernel_size_0': ['network', 'conv', 'k0'],
        'kernel_size_1': ['network', 'conv', 'k1'],
        'kernel_size_2': ['network', 'conv', 'k2'],
        'kernel_size_3': ['network', 'conv', 'k3'],
        'posenc_order': ['network', 'conv', 'posenc_order'],
        'pe_scale': ['network', 'conv', 'pe_scale'],
        'conv_mlp_type': ['network', 'conv', 'mlp_type'],
        'conv_N_bins': ['network', 'conv', 'N_bins'],
        'conv_N_ch': ['network', 'conv', 'mid_ch'],
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

def get_wandb_fit_config():
    wandb_sweep_dict = {
        'learning_rate': ['optimizer', 'learning rate'],
        'weight_decay': ['optimizer', 'weight decay'],
        'n_deform_layers': ['deformation', 'network_config', 'n_hidden_layers'],
        'n_static_features': ['static', 'encoding_config', 'nested', 0, 'n_features_per_level'],
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

    # if args['deformation_encoding'] == 'Composite':
    #     args['deformation']['encoding_config'] = {
    #         'otype': 'Composite',
    #         'nested': [0, 0]
    #     }
    #     if args.get('spatial_encoding', None) == 'OneBlob':
    #         args['deformation']['encoding_config']['nested'][0] = {
    #             'n_dims_to_encode': 3,
    #             'otype': 'OneBlob',
    #             'n_bins': 64
    #         }
    #     else:
    #         args['deformation']['encoding_config']['nested'][0] = {
    #             'n_dims_to_encode': 3,
    #             'otype': 'TriangleWave',
    #             'n_frequencies': 7
    #         }
    # if args['time_encoding'] == 'OneBlob':
    #     if 'nested' in args['deformation']['encoding_config']:
    #         args['deformation']['encoding_config']['nested'][1] = {
    #             'otype': 'OneBlob',
    #             'n_bins': 128
    #         }
    #     args['residual']['encoding_config']['nested'][1] = {
    #         'otype': 'OneBlob',
    #         'n_bins': 64
    #     }
    # else:
    #     if 'nested' in args['deformation']['encoding_config']:
    #         args['deformation']['encoding_config']['nested'][1] = {
    #             'otype': 'TriangleWave',
    #             'n_frequencies': 8
    #         }
    #     args['residual']['encoding_config']['nested'][1] = {
    #         'otype': 'TriangleWave',
    #         'n_frequencies': 7
    #     }

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
        # if cmd_args.config_name.endswith('1k'):
        #     path = path.replace('1k.yaml', '.yaml')
        #     if osp.exists(path):
        #         args = yaml.safe_load(open(path, 'r'))
        #         if 'data loading' in args: raise NotImplementedError
        #         args['data loading'] = {'N': 1000}
        #     else:
        #         raise ValueError(f"bad config_name {cmd_args.config_name}")
        # else:
        raise ValueError(f"bad config_name {cmd_args.config_name}")

    if cmd_args is not None:
        for param in ["job_id", "config_name", 'target_job', 'sweep_id', 'no_wandb']:
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
                raise NotImplementedError("need to handle case of multiple parents each with other parents")

    config_path = osp.join(CONFIG_DIR, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
    infer_missing_args(args)
    return args

def infer_missing_args(args):
    """Convert str to float, etc."""
    paths = args["paths"]
    paths["slurm output dir"] = osp.expandvars(paths["slurm output dir"])
    if args["job_id"].startswith("lg_") or args["job_id"].startswith("A6"):
        args["data loading"]["batch size"] *= 6
        args["optimizer"]["epochs"] //= 6
    paths["job output dir"] = osp.join(paths["slurm output dir"], args["job_id"])
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if isinstance(args["optimizer"][k], str) and args["optimizer"][k].startswith("1e"):
            args["optimizer"][k] = float(args["optimizer"][k])
