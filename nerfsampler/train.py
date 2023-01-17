"""
Entrypoint for training
"""
import sys
import torch, wandb, os
import numpy as np
from functools import partial

from nerfsampler.utils import args as args_module
# from nerfsampler.experiments.diffusion import train_diffusion_model
# from nerfsampler.experiments.depth import train_depth_model
from nerfsampler.experiments.sdf import train_nerf_to_sdf
# from nerfsampler.experiments.motion import train_nerf_motion
from nerfsampler.experiments.classify import train_classifier
from nerfsampler.experiments.segment import train_segmenter
from nerfsampler.experiments.generate import train_generator
from nerfsampler.experiments.warp import train_warp

def main():
    os.environ['WANDB_API_KEY'] = open(os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])
    method_dict = {
        # 'diffusion': train_diffusion_model,
        'classify': train_classifier,
        'sdf': train_nerf_to_sdf,
        # 'motion': train_nerf_motion,
        'segment': train_segmenter,
        'generate': train_generator,
        'warp': train_warp,
    }
    method = method_dict[args["network"]["task"]]
        
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='nerfsampler')
    else:
        method(args=args)

if __name__ == "__main__":
    main()
