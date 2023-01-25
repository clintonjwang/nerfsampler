import pdb
import torch
import opensimplex
from sklearn.decomposition import PCA
F = torch.nn.functional
from math import sin, cos, pi

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

# @njit(cache=True, parallel=True)
def noise4a(points, t):
    rgbs = []
    for ix in range(len(points)):
        rgbs.append(opensimplex.noise4(points[ix,0], points[ix,1], points[ix,2], t))
    return rgbs

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

def forward_recolor(self, recolored_coords, eps=.01):
    bbox = recolored_coords.min(dim=0).values-eps, recolored_coords.max(dim=0).values+eps
    def forward(ray_samples: RaySamples, compute_normals: bool = False):
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        positions = ray_samples.frustums.get_positions()
        #in_range = (self._sample_locations > bbox[0]).min(dim=-1).values & (self._sample_locations < bbox[1]).min(dim=-1).values
        in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
        rgb = field_outputs[FieldHeadNames.RGB]
        if self.frame_frac < 1/3:
            alpha = self.frame_frac * 3
            new_rgb = torch.stack((rgb[...,1], rgb[...,2], rgb[...,0]), dim=-1) * (1-alpha) + \
                    rgb * alpha
        elif self.frame_frac < 2/3:
            alpha = self.frame_frac * 3 - 1
            new_rgb = rgb * (1-alpha) + torch.stack((rgb[...,2], rgb[...,0], rgb[...,1]), dim=-1) * alpha
        else:
            alpha = self.frame_frac * 3 - 2
            new_rgb = torch.stack((rgb[...,2], rgb[...,0], rgb[...,1]), dim=-1) * (1-alpha) + \
                    torch.stack((rgb[...,1], rgb[...,2], rgb[...,0]), dim=-1) * alpha
        field_outputs[FieldHeadNames.RGB] = torch.where(in_range.unsqueeze(-1), new_rgb, rgb)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    return forward


def forward_proc_texture(self, coords, eps=.01):
    bbox = coords.min(dim=0).values-eps, coords.max(dim=0).values+eps
    opensimplex.seed(324)
    def forward(ray_samples: RaySamples, compute_normals: bool = False):
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        positions = ray_samples.frustums.get_positions()
        in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
        if in_range.sum() > 0:
            positions = positions[in_range].cpu().numpy()
            R = noise4a(positions*51, sin(self.t)*6)
            G = noise4a(positions*37, cos(self.t)*5)
            B = noise4a(positions*18, sin(self.t)*3)
            rgbs = torch.cat((
                torch.tensor(R, device=density.device).unsqueeze(-1)-.1,
                torch.tensor(G, device=density.device).unsqueeze(-1)-.1,
                torch.tensor(B, device=density.device).unsqueeze(-1)+.1), dim=1)
            field_outputs[FieldHeadNames.RGB][in_range] = torch.clamp((rgbs + 0.9)/1.8, 0, 1)

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    return forward

def forward_uv_map(self, coords, texture, eps=.01):
    bbox = coords.min(dim=0).values-eps, coords.max(dim=0).values+eps
    def forward(ray_samples: RaySamples, compute_normals: bool = False):
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        positions = ray_samples.frustums.get_positions()
        in_range = (positions > bbox[0]).min(dim=-1).values & (positions < bbox[1]).min(dim=-1).values
        R = torch.tensor([[cos(self.t), -sin(self.t)],
            [sin(self.t), cos(self.t)]], device=in_range.device)
        if in_range.sum() > 0:
            positions = positions[in_range].cpu().numpy()
            pca = PCA(n_components=2)
            uv_coords = torch.matmul(torch.tensor(pca.fit_transform(positions), device=in_range.device), R)
            uv_coords -= uv_coords.min(dim=0).values
            uv_coords /= uv_coords.max(dim=0).values
            rgbs = F.grid_sample((texture/255.).unsqueeze(0), uv_coords.reshape(1,1,-1,2)*2-1, align_corners=False).squeeze().T
            field_outputs[FieldHeadNames.RGB][in_range] = rgbs

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    return forward
