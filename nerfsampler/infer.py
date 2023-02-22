"""Entrypoint for inference"""
import sys
import os
import numpy as np
sys.path.insert(0,os.path.expandvars('$NFS/code/controlnet/controlnet'))
from nerfsampler.utils import args as args_module

def main():
    os.environ['WANDB_API_KEY'] = open(os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    args = args_module.parse_args(sys.argv[1:])
    import torch
    from nerfsampler.experiments.segment import run_segmenter
    from nerfsampler.experiments.dreamfusion import run_dreamfusion

    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    method_dict = {
        'segment': run_segmenter,
        'edit': run_segmenter,
        'dreamfusion': run_dreamfusion,
    }
    if args["task"] in method_dict.keys():
        method_dict[args["task"]](args=args)
    else:
        run_segmenter(args=args)

if __name__ == "__main__":
    main()
