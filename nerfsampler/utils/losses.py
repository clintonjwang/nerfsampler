import torch
from math import log
import numpy as np
import pdb
from torch import Tensor, tensor

from nerfsampler.inn.fields import DiscretizedField, FieldBatch
from nerfsampler.inn.point_set import Discretization
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

def adv_loss_fxns(loss_settings: dict):
    if "WGAN" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit.squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit - true_logit).squeeze()
        return G_fxn, D_fxn
    elif "standard" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit - torch.log1p(torch.exp(-fake_logit))#torch.log(1-torch.sigmoid(fake_logit)).squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit + torch.log1p(torch.exp(-fake_logit)) + torch.log1p(torch.exp(-true_logit))).squeeze()
        #-torch.log(1-fake_logit) - torch.log(true_logit)
        return G_fxn, D_fxn
    else:
        raise NotImplementedError

def gradient_penalty(real_img: torch.Tensor, generated_img: torch.Tensor,
    D: nn.Module):
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, 1, device='cuda')
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach())
    interp_logit = D(interp_img)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def gradient_penalty_inr(real_inr: DiscretizedField,
    generated_inr: DiscretizedField, D: nn.Module):
    real_img = real_inr.values
    generated_img = generated_inr.values
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, device='cuda')
    interp_vals = alpha*real_img + (1-alpha)*generated_img.detach()
    interp_vals.requires_grad = True
    disc = Discretization(real_inr.coords, real_inr.discretization_type)
    interp_logit = D(DiscretizedField(disc, interp_vals))

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_vals,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def mean_iou(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg & gt_seg).sum() / pred_seg.size(0)

# def CrossEntropy(N: torch.int16=128):
#     ce = nn.CrossEntropyLoss()
#     def ce_loss(pred: FieldBatch, class_ix: torch.Tensor):
#         coords = pred.generate_discretization(sample_size=N)
#         return ce(pred(coords), class_ix)
#     return ce_loss

# def L1_dist_inr(N: int=128):
#     def l1_qmc(pred: FieldBatch, target: FieldBatch):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).abs().mean()
#     return l1_qmc
# class L1_dist_inr(nn.Module):
#     def __init__(self, N=128):
#         self.N = N
#     def forward(pred,target):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).abs().mean()

# def L2_dist_inr(N: int=128):
#     def l2_qmc(pred: FieldBatch, target: FieldBatch):
#         coords = target.generate_discretization(sample_size=N)
#         return (pred(coords)-target(coords)).pow(2).mean()
#     return l2_qmc

# def L1_dist(inr, gt_values, coords: Discretization):
#     pred = inr(coords)
#     #pred = util.realign_values(pred, coords_gt=coords, inr=inr)
#     return (pred-gt_values).abs().mean()
