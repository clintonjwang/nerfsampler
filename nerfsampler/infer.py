"""Entrypoint for inference"""
import sys
import torch
import numpy as np

from nerfsampler.utils import args as args_module
from nerfsampler.experiments.segment import clip_segment
# from nerfsampler.experiments.sdf import test_nerf_to_sdf

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    method_dict = {
        # 'classify': test_inr_classifier,
        # 'sdf': train_nerf_to_sdf,
        'segment': clip_segment,
        # 'generate': test_inr_generator,
    }
    method_dict[args["network"]["task"]](args=args)

if __name__ == "__main__":
    main()
