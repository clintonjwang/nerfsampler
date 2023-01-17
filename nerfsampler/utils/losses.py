import torch
from math import log
import numpy as np
import pdb
from torch import tensor

from nerfsampler.networks.fields import DiscretizedField, FieldBatch
from nerfsampler.networks.point_set import Discretization
nn = torch.nn
F = nn.functional

def tv_norm(img):
    """Computes the total variation norm of an image."""
    return (img[:, :, 1:] - img[:, :, :-1]).norm(dim=-1).mean() + \
        (img[:, 1:] - img[:, :-1]).norm(dim=-1).mean() + \
        (img[1:] - img[:-1]).norm(dim=-1).mean()

def psnr(preds, target, data_range=2., base=10, dim=None):
    """Computes the peak signal-to-noise ratio.
    Args:
        preds: estimated signal
        target: ground truth signal
        data_range: the range of the data (max-min)
        base: a base of a logarithm to use
        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.
    """
    if dim is None:
        sum_squared_error = torch.sum(torch.pow(preds - target, 2))
        n_obs = tensor(target.numel(), device=target.device)
    else:
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff, dim=dim)

        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        if not dim_list:
            n_obs = tensor(target.numel(), device=target.device)
        else:
            n_obs = tensor(target.size(), device=target.device)[dim_list].prod()
            n_obs = n_obs.expand_as(sum_squared_error)

    psnr_base_e = 2 * log(data_range) - torch.log(sum_squared_error / n_obs)
    psnr_vals = psnr_base_e * 10 / log(base)
    return psnr_vals.mean()



def ncc_loss(Ii, Ji, window=None):
    # [B, C, *dims]
    Ii = Ii.float()
    Ji = Ji.float()
    ndims = len(list(Ii.size())) - 2
    if window is None:
        window = [9] * ndims
    kwargs = {
        'weight': Ii.new_ones((1, 1, *window)) / np.prod(window),
        'padding': [w//2 for w in window]
    }
    conv_fn = getattr(F, 'conv%dd' % ndims)
    u_I = conv_fn(Ii, **kwargs)
    u_J = conv_fn(Ji, **kwargs)
    I2 = conv_fn(Ii * Ii, **kwargs)
    J2 = conv_fn(Ji * Ji, **kwargs)
    IJ = conv_fn(Ii * Ji, **kwargs)
    cross = IJ - u_J * u_I
    I_var = I2 - u_I * u_I
    J_var = J2 - u_J * u_J
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -cc.mean()

def mse_loss(pred,target):
    return (pred-target).pow(2).flatten(start_dim=1).mean(1)

def contrastive_loss(features, T=0.5):
    # features: [B,D,C] where D=2 is different discretizations of the same field
    B,D,C = features.shape
    z_i = features.reshape(B*D,1,C)
    z_j = features.reshape(1,B*D,C)
    s_ij = F.cosine_similarity(z_i, z_j)/T # pairwise similarities [BD, BD]
    xs_ij = torch.exp(s_ij)
    xs_i = xs_ij.sum(dim=1, keepdim=True)
    loss_ij = -s_ij + torch.log(xs_i - xs_ij)
    return loss_ij.mean()

def mean_iou(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg & gt_seg).sum() / pred_seg.size(0)
