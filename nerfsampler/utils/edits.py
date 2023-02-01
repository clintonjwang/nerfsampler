import pdb
import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp

from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfsampler.utils.point_cloud import OccupancyTester

def duplicate_rgb(self, orig_coords, transform):
    # transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    ot = OccupancyTester(new_coords)
    
    def get_outputs(ray_samples: RaySamples, density_embedding=None):
        assert density_embedding is not None
        outputs = {}
        positions = ray_samples.frustums.get_positions()
        in_range = ot.get_mask_of_points_inside_pcd(positions)
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

def duplicate_density_fxn(orig_fxn, orig_coords, transform):
    # transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    ot = OccupancyTester(new_coords)
    def new_fxn(positions):
        in_range = ot.get_mask_of_points_inside_pcd(positions)
        positions = torch.where(in_range.unsqueeze(-1), torch.matmul(positions - transform[:,-1], inv_tx), positions)
        return orig_fxn(positions)
    return new_fxn


def duplicate_field_density(self, orig_coords, transform):
    # transform.shape == (3,4)
    new_coords = torch.matmul(orig_coords, transform[:,:3]) + transform[:,-1]
    inv_tx = torch.inverse(transform[:,:3])
    ot = OccupancyTester(new_coords)
    def get_density(ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            in_range = ot.get_mask_of_points_inside_pcd(positions)
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

def attenuate_field_density(self, coords, scale=0.):
    ot = OccupancyTester(coords)
    def get_density(ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            in_range = ot.get_mask_of_points_inside_pcd(positions)
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
        density[in_range] *= scale
        return density, base_mlp_out
    return get_density

def attenuate_density_fxn(orig_fxn, coords, scale=0.):
    ot = OccupancyTester(coords)
    def new_fxn(positions):
        out = orig_fxn(positions)
        in_range = ot.get_mask_of_points_inside_pcd(positions)
        out[in_range] *= scale
        return out
    return new_fxn

def animate_density_fxn(field, orig_fxn, orig_coords):
    old_ot = OccupancyTester(orig_coords)
    cog = orig_coords.mean(dim=0)

    def new_fxn(positions):
        transform = field.animation_transform
        new_coords = torch.matmul(orig_coords - cog, transform[:,:3]) + transform[:,-1] + cog
        new_ot = OccupancyTester(new_coords)

        in_new_range = new_ot.get_mask_of_points_inside_pcd(positions)
        in_old_range = old_ot.get_mask_of_points_inside_pcd(positions) & ~in_new_range
        positions = torch.where(in_new_range.unsqueeze(-1), 
            torch.matmul(positions - transform[:,-1] - cog, field.inv_tx) + cog, positions)
        return torch.where(in_old_range.unsqueeze(-1), 0, orig_fxn(positions))
    return new_fxn
    
def animate_field_density(self, orig_coords):
    old_ot = OccupancyTester(orig_coords)
    cog = orig_coords.mean(dim=0)

    def get_density(ray_samples: RaySamples):
        transform = self.animation_transform
        new_coords = torch.matmul(orig_coords - cog, transform[:,:3]) + transform[:,-1] + cog
        new_ot = OccupancyTester(new_coords)

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            in_new_range = new_ot.get_mask_of_points_inside_pcd(positions)
            in_old_range = old_ot.get_mask_of_points_inside_pcd(positions) & ~in_new_range
            positions = torch.where(in_new_range.unsqueeze(-1), 
                    torch.matmul(positions - cog - transform[:,-1], self.inv_tx) + cog, positions)
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
        return torch.where(in_old_range.unsqueeze(-1), 0, density), base_mlp_out
    return get_density


def animate_rgb(self, orig_coords):
    # orig_bbox = orig_coords.min(dim=0).values-eps, orig_coords.max(dim=0).values+eps
    cog = orig_coords.mean(dim=0)
    def get_outputs(ray_samples: RaySamples, density_embedding=None):
        assert density_embedding is not None
        transform = self.animation_transform
        new_coords = torch.matmul(orig_coords - cog + transform[:,-1], transform[:,:3]) + cog
        new_ot = OccupancyTester(new_coords)

        outputs = {}
        positions = ray_samples.frustums.get_positions()
        in_range = new_ot.get_mask_of_points_inside_pcd(positions)
        directions = torch.where(in_range.unsqueeze(-1),
            torch.matmul(ray_samples.frustums.directions - cog - transform[:,-1], self.inv_tx) + cog,
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


def reflect_rgb(self, coords):
    ot = OccupancyTester(coords)
    def get_outputs(ray_samples: RaySamples, density_embedding: TensorType):
        # predicted normals
        positions = ray_samples.frustums.get_positions()
        raise NotImplementedError
        in_range = None
        
        positions_flat = self.position_encoding(positions.view(-1, 3))
        pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
        N = self.field_head_pred_normals(x)
        V = ray_samples.frustums.directions
        R = V - 2 * (V * N).sum(dim=-1, keepdim=True) * N

        outputs = {}
        directions = (ray_samples.frustums.directions + 1)/2
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
