from __future__ import annotations
from nerfsampler import inn
import pdb
from nerfsampler.inn.support import BoundingBox
from nerfsampler.inn.fields import FieldBatch
from nerfsampler.inn.fields import DiscretizedField
from nerfsampler.inn.point_set import Discretization, Sampler, generate_discretization

import torch
nn = torch.nn


class nerfsampler(nn.Module):
    def __init__(self, layers=None, encoder=(), decoder=(), sampler: dict = None,
        return_features=False):
        """nerfsampler produces neural fields represented by an intermediate
        vector and a decoder.

        Attributes:
        - encoder: layers to produce an intermediate discretized field
        - decoder: layers to interpolate / decode the intermediate field
        - sampler: the discretization of the input nf
        """
        super().__init__()
        if layers:
            self.encoder = layers
        else:
            self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.return_features = return_features

    def __len__(self):
        return len(self.encoder) + len(self.decoder)

    def __iter__(self):
        return iter(self.encoder) + iter(self.decoder)

    def __getitem__(self, ix):
        N = len(self.encoder)
        return self.encoder[ix] if ix < N else self.decoder[ix-N]

    def discretize_inr(self, nf: FieldBatch,
        sampler: Sampler|Discretization|None = None) -> DiscretizedField:
        """Discretize an INRBatch into a DiscretizedINR."""
        if sampler is None:
            sampler = self.sampler
        if isinstance(sampler, Discretization):
            disc = sampler
        else:
            disc = generate_discretization(domain=nf.domain, sampler=sampler)
        return DiscretizedField(disc, values=nf(disc.coords), domain=nf.domain)

    def encode(self, nf: DiscretizedField) -> DiscretizedField | torch.Tensor:
        """Produces intermediate field at discretized points"""
        return self.encoder(nf)

    def decode(self, nf: DiscretizedField, out_coords: torch.Tensor | None) -> DiscretizedField | torch.Tensor:
        """Interpolates / decodes intermediate field"""
        if isinstance(self.decoder, tuple):
            raise ValueError("nerfsampler was not given a decoder")
        return self.decoder(nf, out_coords)

    def forward(self, nf: FieldBatch,
        out_coords: torch.Tensor | None = None,
        sampler=None) -> DiscretizedField:

        if not isinstance(nf, DiscretizedField):
            nf = self.discretize_inr(nf, sampler=sampler)
        elif not isinstance(nf, FieldBatch):
            raise ValueError(
                "nf must be INRBatch, but got {}".format(type(nf)))

        if self.return_features:
            feats = self.encode(nf)
            return self.decode(feats, out_coords), feats
        elif out_coords is None:
            return self.encode(nf)
        else:
            return self.decode(self.encode(nf), out_coords)


class Adaptivenerfsampler(nerfsampler):
    def __init__(self, coord_predictor, encoder=None, decoder=None, init_sampler: dict=None):
        """
        Args:
        - init_sampler: the initial discretization of the input NF
        - coord_predictor: a subnetwork to predict which coordinates to sample next
        """
        super().__init__(encoder=encoder, decoder=decoder, sampler=init_sampler)
        self.coord_predictor = coord_predictor
        self.init_coords = generate_discretization(sampler=init_sampler).coords

    def forward(self, nf: FieldBatch, init_coords=None,
        out_coords=None) -> DiscretizedField:
        """Forward pass of Adaptivenerfsampler.

        Args:
            nf (FieldBatch): input nf
            init_coords (optional):
            full_coords (optional):
            out_coords (optional):

        Returns:
            DiscretizedField:
        """
        if init_coords is None:
            init_coords = self.init_coords
        
        old_vals = DiscretizedField(init_coords, nf(init_coords))
        full_vals = DiscretizedField(full_coords, nf(full_coords))
        full_coords = torch.stack((init_coords, ))
        probs = self.coord_predictor(DiscretizedField(init_coords, old_vals))
        self.sampler(probs)
        masks = self.probs_to_masks(full_coords, probs) # (B,N_bins) -> (B,N_points)

        if out_coords is None:
            return self.encode(old_vals, full_vals, masks)
        else:
            return self.decode(self.encode(full_disc_nf), out_coords)

# class Adaptivenerfsampler(nerfsampler):
#     def __init__(self, init_sampler: dict, coord_predictor, encoder=None, decoder=None):
#         super().__init__(init_sampler, encoder, decoder)
#         self.coord_predictor = coord_predictor

#     def forward(self, nf: FieldBatch, init_coords=None, out_coords=None) -> DiscretizedField:
#         old_vals = DiscretizedField(init_coords, nf(init_coords))
#         new_coords = self.coord_predictor(
#             DiscretizedField(init_coords, old_vals))
#         new_vals = nf(new_coords)
#         full_disc_inr = DiscretizedField(torch.stack((init_coords, new_coords)),
#                                          torch.stack((old_vals, new_vals)))

#         if out_coords is None:
#             return self.encode(full_disc_inr)
#         else:
#             return self.decode(self.encode(full_disc_inr), out_coords)


def freeze_layer_types(nerfsampler, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in nerfsampler:
        if hasattr(m, '__iter__'):
            freeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = False


def unfreeze_layer_types(nerfsampler, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in nerfsampler:
        if hasattr(m, '__iter__'):
            unfreeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = True


def replace_conv_kernels(nerfsampler, k_type='mlp', k_ratio=1.5):
    if hasattr(nerfsampler, 'sequential'):
        return replace_conv_kernels(nerfsampler.sequential, k_ratio=k_ratio)
    elif hasattr(nerfsampler, 'layers'):
        return replace_conv_kernels(nerfsampler.layers, k_ratio=k_ratio)
    length = len(nerfsampler)
    for i in range(length):
        m = nerfsampler[i]
        if hasattr(m, '__getitem__'):
            replace_conv_kernels(m, k_ratio=k_ratio)
        elif isinstance(m, inn.SplineConv):
            try:
                nerfsampler[i] = replace_conv_kernel(m, k_ratio=k_ratio)
            except:
                pdb.set_trace()


def replace_conv_kernel(layer, k_type='mlp', k_ratio=1.5):
    # if k_type
    if isinstance(layer, inn.SplineConv):
        conv = inn.MLPConv(layer.in_channels, layer.out_channels,
                           kernel_support=BoundingBox.from_orthotope(
                               [k*k_ratio for k in layer.kernel_size]),
                           down_ratio=layer.down_ratio, groups=layer.groups)
        # conv.padded_extrema = layer.padded_extrema
        conv.bias = layer.bias
        return conv
    raise NotImplementedError
