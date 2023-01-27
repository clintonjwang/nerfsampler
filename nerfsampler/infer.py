"""Entrypoint for inference"""
import sys
import torch
import numpy as np

from nerfsampler.utils import args as args_module
from nerfsampler.experiments.segment import run_segmenter

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    method_dict = {
        'segment': run_segmenter,
        'edit': run_segmenter,
    }
    method_dict[args["task"]](args=args)

if __name__ == "__main__":
    main()
