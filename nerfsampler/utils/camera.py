import torch
from math import sin, cos, pi

def animate_camera(camera_ray_bundle, reference=None, frac=0.5, motion="rotate"):
    if reference is None:
        reference = camera_ray_bundle.origins, camera_ray_bundle.directions
    if motion == "rotate":
        t = frac * 2 * pi
        R = torch.tensor([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]], device=camera_ray_bundle.origins.device)
        camera_ray_bundle.origins = torch.matmul(reference[0], R)
        camera_ray_bundle.directions = torch.matmul(reference[1], R)
    # elif motion == "translate":
    else:
        raise NotImplementedError
