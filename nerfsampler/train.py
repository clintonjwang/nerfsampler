"""
Entrypoint for training
"""
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import torch, wandb, os
import numpy as np
from functools import partial

from nerfsampler.utils import args as args_module
from nerfsampler.srt.train import train_srt

def main():
    os.environ['WANDB_API_KEY'] = open(os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    # if args["random seed"] >= 0:
    #     np.random.seed(args["random seed"])
    #     torch.manual_seed(args["random seed"])
    method_dict = {
        'srt': train_srt,
    }
    method = method_dict[args["task"]]
        
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='nerfsampler')
    else:
        method(args=args)

if __name__ == "__main__":
    main()