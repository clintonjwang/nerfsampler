from __future__ import annotations
import seaborn as sns
import monai.transforms as mtr
import matplotlib.pyplot as plt
from glob import glob
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nerfsampler.inn.fields import DiscretizedField

import os
import torch
import numpy as np
from nerfsampler import TMP_DIR

osp = os.path

rescale_clip = mtr.ScaleIntensityRangePercentiles(
    lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(
    lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)


def rgb2d_tensor_to_npy(x):
    if len(x.shape) == 4:
        x = x[0]
    return x.permute(1, 2, 0).detach().cpu().numpy()


def grayscale2d_tensor_to_npy(x):
    x.squeeze_()
    if len(x.shape) == 3:
        x = x[0]
    return x.detach().cpu().numpy()


def BNc_to_npy(x, dims):
    return x[0].reshape(*dims, -1).squeeze(-1).detach().cpu().numpy()

def BNc_to_Bcdims(x, dims):
    return x.permute(0, 2, 1).reshape(x.size(0), -1, *dims)

def Bcdims_to_BNc(x):
    return x.flatten(start_dim=2).transpose(2, 1)


def is_model_nerfsampler(args):
    return args["network"]['type'].lower().startswith('i') or \
        args["network"]['type'].startswith('Tx') or \
        args["network"]['type'].startswith('F2F')


def is_model_adaptive_nerfsampler(args):
    return args["network"]['type'].lower().startswith('a')


def meshgrid(*tensors, indexing='ij') -> torch.Tensor:
    try:
        return torch.meshgrid(*tensors, indexing=indexing)
    except TypeError:
        return torch.meshgrid(*tensors)


def get_optimizer(model, args: dict, lr=None):
    opt_settings = args["optimizer"]
    if lr is None:
        lr = opt_settings["learning rate"]
    if 'beta1' in opt_settings:
        betas = (opt_settings['beta1'], .999)
    else:
        betas = (.9, .999)
    if opt_settings['type'].lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=args["optimizer"]["weight decay"], betas=betas)
    elif opt_settings['type'].lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=args["optimizer"]["weight decay"], betas=betas)
    else:
        raise NotImplementedError


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def imshow(img):
    if isinstance(img, torch.Tensor):
        plt.imshow(rescale_noclip(
            img.detach().cpu().squeeze().permute(1, 2, 0).numpy()))
    else:
        plt.imshow(img)
    plt.axis('off')


def load_checkpoint(model, paths):
    if paths["pretrained model name"] is not None:
        init_weight_path = osp.join(TMP_DIR, paths["pretrained model name"])
        if not osp.exists(init_weight_path):
            raise ValueError(f"bad pretrained model path {init_weight_path}")

        checkpoint_sd = torch.load(init_weight_path)
        model_sd = model.state_dict()
        for k in model_sd.keys():
            if k in checkpoint_sd.keys() and checkpoint_sd[k].shape != model_sd[k].shape:
                checkpoint_sd.pop(k)

        model.load_state_dict(checkpoint_sd, strict=False)


def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]


def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]


def glob2(*paths):
    pattern = osp.expanduser(osp.join(*paths))
    if "*" not in pattern:
        pattern = osp.join(pattern, "*")
    return glob(pattern)


def flatten_list(collection):
    new_list = []
    for element in collection:
        new_list += list(element)
    return new_list


def format_float(x, n_decimals):
    if x == 0:
        return "0"
    elif np.isnan(x):
        return "NaN"
    if hasattr(x, "__iter__"):
        np.set_printoptions(precision=n_decimals)
        return str(np.array(x)).strip("[]")
    else:
        if n_decimals == 0:
            return ('%d' % x)
        else:
            return ('{:.%df}' % n_decimals).format(x)


def latex_mean_std(X=None, mean=None, stdev=None, n_decimals=1, percent=False, behaviour_if_singleton=None):
    if X is not None and len(X) == 1:
        mean = X[0]
        if not percent:
            return (r'{0:.%df}' % n_decimals).format(mean)
        else:
            return (r'{0:.%df}\%%' % n_decimals).format(mean*100)

    if stdev is None:
        mean = np.nanmean(X)
        stdev = np.nanstd(X)
    if not percent:
        return (r'{0:.%df}\pm {1:.%df}' % (n_decimals, n_decimals)).format(mean, stdev)
    else:
        return (r'{0:.%df}\%%\pm {1:.%df}\%%' % (n_decimals, n_decimals)).format(mean*100, stdev*100)


class MetricTracker:
    def __init__(self, name=None, intervals=None, function=None, weight=1.):
        self.name = name
        self.epoch_history = {"train": [], "val": []}
        if intervals is None:
            intervals = {"train": 1, "val": 1}
        self.intervals = intervals
        self.function = function
        self.minibatch_values = {"train": [], "val": []}
        self.weight = weight

    def __call__(self, *args, phase="val", **kwargs):
        loss = self.function(*args, **kwargs)
        self.update_with_minibatch(loss, phase=phase)
        # if np.isnan(loss.mean().item()):
        #     raise ValueError(f"{self.name} became NaN")
        return loss.mean() * self.weight

    def update_at_epoch_end(self):
        for phase in self.minibatch_values:
            if len(self.minibatch_values[phase]) != 0:
                self.epoch_history[phase].append(self.epoch_average(phase))
                self.minibatch_values[phase] = []

    def update_with_minibatch(self, value, phase="val"):
        if isinstance(value, torch.Tensor):
            if torch.numel(value) == 1:
                self.minibatch_values[phase].append(value.item())
            else:
                self.minibatch_values[phase] += list(
                    value.detach().cpu().numpy())
        elif isinstance(value, list):
            self.minibatch_values[phase] += value
        elif isinstance(value, np.ndarray):
            self.minibatch_values[phase] += list(value)
        elif not np.isnan(value):
            self.minibatch_values[phase].append(value)

    def epoch_average(self, phase="val"):
        if len(self.minibatch_values[phase]) != 0:
            return np.nanmean(self.minibatch_values[phase])
        elif len(self.epoch_history[phase]) != 0:
            return self.epoch_history[phase][-1]
        return np.nan

    def max(self, phase="val"):
        try:
            return np.nanmax(self.epoch_history[phase])
        except:
            return np.nan

    def min(self, phase="val"):
        try:
            return np.nanmin(self.epoch_history[phase])
        except:
            return np.nan

    def current_moving_average(self, window, phase="val"):
        return np.mean(self.minibatch_values[phase][-window:])

    def get_moving_average(self, window, interval=None, phase="val"):
        if interval is None:
            interval = window
        ret = np.cumsum(self.minibatch_values[phase], dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret = ret[window - 1:] / window
        return ret[::interval]

    def is_at_max(self, phase="val"):
        return self.max(phase) >= self.epoch_average(phase)

    def is_at_min(self, phase="val"):
        return self.min(phase) <= self.epoch_average(phase)

    def histogram(self, path=None, phase="val", epoch=None):
        if epoch is None:
            epoch = len(self.epoch_history[phase]) * self.intervals[phase]
        if len(self.minibatch_values[phase]) < 5:
            return
        plt.hist(self.minibatch_values[phase])
        plt.title(f"{self.name}_{phase}_{epoch}")
        if path is not None:
            plt.savefig(path)
            plt.clf()

    def plot_running_average(self, path, phase='val'):
        _, axis = plt.subplots()
        N = 20
        x = self.epoch_history[phase]
        if len(x) == 0:
            x = self.minibatch_values[phase]
        if len(x) <= N:
            return
        spacing = N//2
        y = np.convolve(x, np.ones(N)/N, mode='valid')[::spacing]
        sns.lineplot(x=range(0, len(y)*spacing, spacing),
                     y=y, ax=axis, label=phase)
        axis.set_ylabel(self.name)
        plt.savefig(path)
        plt.close()

    def lineplot(self, path=None):
        _, axis = plt.subplots()

        if "train" in self.intervals:
            dx = self.intervals["train"]
            values = self.epoch_history["train"]
            if len(values) >= 3:
                x_values = np.arange(0, dx*len(values), dx)
                sns.lineplot(x=x_values, y=values, ax=axis, label="train")

        if "val" in self.intervals:
            dx = self.intervals["val"]
            values = self.epoch_history["val"]
            if len(values) >= 3:
                x_values = np.arange(0, dx*len(values), dx)
                sns.lineplot(x=x_values, y=values, ax=axis, label="val")

        axis.set_ylabel(self.name)

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            return axis


def save_plots(trackers, root=None):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        path = osp.join(root, tracker.name+".png")
        tracker.lineplot(path=path)


def save_metric_histograms(trackers, epoch, root):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        for phase in ["train", "val"]:
            path = osp.join(root, f"{epoch}_{tracker.name}_{phase}.png")
            tracker.histogram(path=path, phase=phase, epoch=epoch)
