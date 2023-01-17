from nerfsampler.baselines.classifier import conv_bn_relu
from nerfsampler.inn.nets.nerfsampler import nerfsampler, Adaptivenerfsampler
from nerfsampler.inn.fields import DiscretizedField
from nerfsampler import inn
import pdb
import torch

from nerfsampler.inn.support import BoundingBox
nn = torch.nn
F = nn.functional


class A2D(Adaptivenerfsampler):
    def __init__(self, in_channels, out_channels, C=8, dims=3,
                 final_activation='tanh', **kwargs):
        coord_predictor = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 2, bias=False, **kwargs),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, dims, 1, bias=True),
        )
        # nn.init.kaiming_uniform_([cv.weight for cv in coord_predictor])
        coord_predictor = nn.Sequential(*coord_predictor)

        encoder = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.3, .3),
                                     **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.6, .6), down_ratio=.5, **kwargs),
            inn.ChannelMixer(C, C, bias=True),
        ]
        encoder = nn.Sequential(*encoder)

        kernel_support = BoundingBox.from_orthotope(dims=(.6, .6))
        decoder = [
            inn.MLPConv(C, out_channels,
                        kernel_support=kernel_support, **kwargs),
        ]
        if final_activation is not None:
            decoder.append(inn.get_activation_layer(final_activation))
        decoder = inn.blocks.Sequential(*decoder)

        super().__init__(coord_predictor=coord_predictor, encoder=encoder, decoder=decoder)


class AF2F_3d(Adaptivenerfsampler):
    def __init__(self, in_channels, out_channels, init_sampler=None, C=8, dims=3,
                 final_activation='tanh', **kwargs):
        
        coord_predictor = nn.Sequential(
            nn.Conv3d(in_channels, C, 3, padding=1, bias=False, **kwargs),
            nn.BatchNorm3d(C),
            nn.ReLU(inplace=True),
            nn.Conv3d(C, C, 2, bias=False, **kwargs),
            nn.BatchNorm3d(C),
            nn.ReLU(inplace=True),
            nn.Conv3d(C, dims, 1, bias=True),
            nn.Tanh(),
            inn.point_set.Sampler(),
        )
        # nn.init.kaiming_uniform_([cv.weight for cv in coord_predictor])
        coord_predictor = nn.Sequential(*coord_predictor)

        encoder = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.3, .3, .3),
                                     **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.6, .6, .6), down_ratio=.5, **kwargs),
            inn.ChannelMixer(C, C, bias=True),
        ]
        encoder = nn.Sequential(*encoder)

        kernel_support = BoundingBox.from_orthotope(dims=(.6, .6, .6))
        decoder = [
            inn.MLPConv(C, out_channels,
                        kernel_support=kernel_support, **kwargs),
        ]
        if final_activation is not None:
            decoder.append(inn.get_activation_layer(final_activation))
        decoder = inn.blocks.Sequential(*decoder)
        # if init_sampler is None:
        # init_sampler = inn.point_set.Sampler()
        super().__init__(coord_predictor=coord_predictor, encoder=encoder, decoder=decoder, init_sampler=init_sampler)


class F2F_3d(nerfsampler):
    def __init__(self, in_channels, out_channels, C=16,
                 final_activation='tanh', **kwargs):
        encoder = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.3, .3, .3),
                                     **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.6, .6, .6), down_ratio=.5, **kwargs),
            inn.ChannelMixer(C, C, bias=True),
            # inn.PositionalEncoding(N=C//8),
        ]
        encoder = nn.Sequential(*encoder)
        kernel_support = BoundingBox.from_orthotope(dims=(.6, .6, .6))
        decoder = [
            inn.MLPConv(C, out_channels,
                        kernel_support=kernel_support, **kwargs),
        ]
        if final_activation is not None:
            decoder.append(inn.get_activation_layer(final_activation))
        decoder = inn.blocks.Sequential(*decoder)
        super().__init__(encoder=encoder, decoder=decoder)


class F2F_3d_5(nerfsampler):
    def __init__(self, in_channels, out_channels, C=4, **kwargs):
        super().__init__()
        self.first = nn.Sequential(
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.2, .2, .2), **kwargs))
        layers = [
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.4, .4, .4), down_ratio=.5, **kwargs),
            inn.blocks.conv_norm_act(
                C, C*2, kernel_size=(.8, .8, .8), down_ratio=.5, **kwargs),
            inn.Upsample(4),
            inn.blocks.conv_norm_act(
                C*2, C, kernel_size=(.4, .4, .4), **kwargs),
        ]
        self.layers = nn.Sequential(*layers)
        self.last = nn.Sequential(
            inn.ChannelMixer(C, out_channels))

    def encode(self, inr: DiscretizedField) -> DiscretizedField | torch.Tensor:
        inr = self.first(inr)
        inr = inr + self.layers(inr)
        return self.last(inr)


class F2F_2d_3(nerfsampler):
    def __init__(self, in_channels, out_channels, sampler=None, C=16,
                 final_activation=None, **kwargs):
        layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.025, .05),
                                     **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.075, .15),
                                     **kwargs),
            inn.ChannelMixer(C*2, out_channels, bias=True),
        ]
        if final_activation is not None:
            layers.append(inn.get_activation_layer(final_activation))
        layers = nn.Sequential(*layers)
        super().__init__(sampler=sampler, layers=layers)


class F2F_2d_5(nerfsampler):
    def __init__(self, in_channels, out_channels, sampler=None, C=16, **kwargs):
        super().__init__(sampler=sampler)
        self.first = nn.Sequential(
                inn.blocks.conv_norm_act(
                    in_channels, C, kernel_size=(.03, .03), **kwargs),
                inn.PositionalEncoding(N=C//4)
            )
        layers = [
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.06, .06), down_ratio=.5, **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.1, .1), down_ratio=.5, **kwargs),
            inn.Upsample(4),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.06, .06), **kwargs),
        ]
        self.layers = nn.Sequential(*layers)
        self.last = nn.Sequential(
            inn.ChannelMixer(C, out_channels))

    def encode(self, inr: DiscretizedField) -> DiscretizedField | torch.Tensor:
        inr = self.first(inr)
        inr = inr + self.layers(inr.create_derived_inr())
        return self.last(inr)
