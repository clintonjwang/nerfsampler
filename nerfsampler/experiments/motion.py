from dataclasses import fields
import gc
import pdb
import time
import wandb
import torch
from nerfsampler import inn
from nerfsampler.inn.fields import DiscretizedField
import nerfsampler.baselines.seg
from nerfsampler.data import dataloader
from nerfsampler.inn import point_set
from nerfsampler.utils import util
from nerfsampler.inn import fields
from nerfsampler.inn.nets import field2field


def load_model(args):
    ntype = args["network"]["type"]
    kwargs = dict(in_channels=3, out_channels=3)
    if hasattr(field2field, ntype):
        module = getattr(field2field, ntype)
        model = module(**kwargs)
    elif hasattr(nerfsampler.baselines.seg, ntype):
        module = getattr(nerfsampler.baselines.seg, ntype)
        model = module(**kwargs)
    elif ntype.startswith("Tx"):
        module = getattr(nerfsampler.baselines.seg, ntype[2:])
        base = module(**kwargs)
        img_shape = args["data loading"]["image shape"]
        model, _ = inn.conversion.translate_discrete_model(base.layers, img_shape,
                                                           extrema=((-1, 1), (-1, 1), (-1, 1)))
    else:
        raise NotImplementedError(f"Network type {ntype} not implemented")
        
    wandb.watch(model, log="all", log_freq=100)
    return model.cuda()

def train_nerf_motion(args: dict) -> None:
    dl_args = args["data loading"]
    global_step = 0
    data_loader = dataloader.get_inr_dataloader(dl_args)
    discretizations = point_set.get_discretizations_for_args(args)
    in_disc = discretizations['input']
    out_disc = discretizations['output']
    test_in_disc = discretizations['test_in']
    test_out_disc = discretizations['test_out']

    model = load_model(args).cuda()
    optimizer = util.get_optimizer(model, args)
    start_time = time.time()
    for frame_pairs in data_loader:
        global_step += 1
        # (B,N,4), (B,N,1)
        prev, next_gt = frame_pairs(in_disc.coords, out_disc.coords)
        if util.is_model_adaptive_nerfsampler(args):
            pass
        elif util.is_model_nerfsampler(args):
            prev_field = DiscretizedField(in_disc, values=prev)
            next_pred = model(prev_field, out_disc.coords).values
            loss = ((next_gt - next_pred)**2).mean()
        else:
            voxels = util.BNc_to_Bcdims(rgba, in_disc.shape)
            next_pred = model(voxels)
            loss = ((util.BNc_to_Bcdims(next_gt, out_disc.shape) - next_pred)**2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log({
            "next_mse": loss.item(),
            'mins_elapsed': (time.time() - start_time)/60,
        })
        print('.', end='')

        test_dtype = test_out_disc.type
        if global_step % 20 == 0:
            del rgba, next_gt, next_pred, loss
            gc.collect()
            torch.cuda.empty_cache()

            prev, _ = frame_pairs(test_in_disc.coords)
            prev_dense, next_gt = frame_pairs(test_out_disc.coords)
            with torch.no_grad():
                if util.is_model_adaptive_nerfsampler(args):
                    pass
                elif util.is_model_nerfsampler(args):
                    prev_field = DiscretizedField(test_in_disc, values=prev)
                    next_pred = model(prev_field, test_out_disc.coords).values
                    loss = ((next_gt - next_pred)**2).mean()
                else:
                    voxels = util.BNc_to_Bcdims(prev, test_in_disc.shape)
                    next_pred = model(voxels)
                    loss = ((util.BNc_to_Bcdims(next_gt, test_out_disc.shape) - next_pred)**2).mean()
            wandb.log({"val_next_mse": loss.item()})

            if test_dtype.startswith("grid"):
                if util.is_model_nerfsampler(args):
                    prev_dense = fields.reorder_grid_data(
                        prev_dense, test_out_disc)
                    next_pred = fields.reorder_grid_data(next_pred, test_out_disc)
                    next_gt = fields.reorder_grid_data(next_gt, test_out_disc)
                shape = test_out_disc.shape
                prev_dense = util.BNc_to_npy(prev_dense[:1], shape)
                next_pred = util.BNc_to_npy(next_pred[:1], shape)
                next_gt = util.BNc_to_npy(next_gt[:1], shape)
                if test_dtype == "grid":
                    prev_dense = prev_dense[:, :, shape[2]//2, :3]
                    next_pred = next_pred[:, :, shape[2]//2]
                    next_gt = next_gt[:, :, shape[2]//2]
                elif test_dtype == "grid_slice":
                    prev_dense = prev_dense[..., :3]

                wandb.log({
                    'prev': wandb.Image(prev_dense),
                    'next_pred': wandb.Image(next_pred),
                    'next_gt': wandb.Image(next_gt),
                })

        if global_step >= args["optimizer"]["max steps"]:
            break
