import pdb
import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp

from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

def duplicate_rgb(self, orig_coords, transform, eps=.01):
    # transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    target_bbox = new_coords.min(dim=0).values-eps, new_coords.max(dim=0).values+eps
    
    def get_outputs(ray_samples: RaySamples, density_embedding=None):
        assert density_embedding is not None
        outputs = {}
        positions = ray_samples.frustums.get_positions()
        in_range = (positions > target_bbox[0]).min(dim=-1).values & (positions < target_bbox[1]).min(dim=-1).values
        directions = torch.where(in_range.unsqueeze(-1),
            torch.matmul(ray_samples.frustums.directions - transform[:,-1], inv_tx),
            ray_samples.frustums.directions)
        directions = (directions + 1.0) / 2.0
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if self.use_average_appearance_embedding:
            embedded_appearance = torch.ones(
                (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
            ) * self.embedding_appearance.mean(dim=0)
        else:
            embedded_appearance = torch.zeros(
                (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
            )
        if self.use_semantics or self.use_pred_normals:
            raise NotImplementedError
        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
    return get_outputs

def duplicate_density_fxn(orig_fxn, orig_coords, transform, eps=.01):
    assert transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    target_bbox = new_coords.min(dim=0).values-eps, new_coords.max(dim=0).values+eps
    def new_fxn(positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        in_range = (positions > target_bbox[0]).min(dim=-1).values & (positions < target_bbox[1]).min(dim=-1).values
        positions = torch.where(in_range.unsqueeze(-1), torch.matmul(positions - transform[:,-1], inv_tx), positions)
        return orig_fxn(positions)
    return new_fxn


def duplicate_field_density(self, orig_coords, transform, eps=.01):
    assert transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    target_bbox = new_coords.min(dim=0).values-eps, new_coords.max(dim=0).values+eps
    def get_density(ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            # WARNING: in_range check may need to be before spatial_distortion to be consistent with proposal network sampler
            in_range = (positions > target_bbox[0]).min(dim=-1).values & (positions < target_bbox[1]).min(dim=-1).values
            positions = torch.where(in_range.unsqueeze(-1), torch.matmul(positions - transform[:,-1], inv_tx), positions)
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            raise NotImplementedError
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out
    return get_density

def remove_field_density(self, removed_coords, eps=.01):
    bbox = removed_coords.min(dim=0).values-eps, removed_coords.max(dim=0).values+eps
    def get_density(ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            # WARNING: in_range check may need to be before spatial_distortion to be consistent with proposal network sampler
            in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            raise NotImplementedError
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        density = trunc_exp(density_before_activation.to(positions))
        return torch.where(in_range.unsqueeze(-1), 0, density), base_mlp_out
    return get_density

def remove_density_fxn(orig_fxn, removed_coords, eps=.01):
    bbox = removed_coords.min(dim=0).values-eps, removed_coords.max(dim=0).values+eps
    def new_fxn(positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
        # diffs = (removed_coords.view(1,1,-1,3) - positions.unsqueeze(-2))
        # in_range = diffs.norm(dim=-1).min(dim=2) < threshold
        return torch.where(in_range.unsqueeze(-1), 0, orig_fxn(positions))
    return new_fxn
