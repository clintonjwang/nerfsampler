"""
Entrypoint for training
"""
import sys
import torch, wandb, os
import numpy as np
from functools import partial

from nerfsampler.utils import args as args_module
from nerfsampler.experiments.segment import train_segmenter

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
        # 'segment': train_segmenter,
    }
    method = method_dict[args["network"]["task"]]
        
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='nerfsampler')
    else:
        method(args=args)

if __name__ == "__main__":
    main()
