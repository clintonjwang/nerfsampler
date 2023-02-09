import open3d as o3d
import pdb, torch
import numpy as np
import matplotlib.pyplot as plt

from nerfsampler.utils import point_cloud

def pcd_to_mesh(pcd, filter_low_density=False):
    if isinstance(pcd, torch.Tensor):
        pcd = point_cloud.torch_to_o3d(pcd)
    pcd = pcd.voxel_down_sample(voxel_size=0.007)
    pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    if filter_low_density:
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    if not mesh.is_watertight():
        if mesh.is_self_intersecting():
            print("WARNING: mesh is self-intersecting")
        else:
            print("WARNING: mesh is not watertight")
    return o3d.t.geometry.TriangleMesh.from_legacy(mesh)#, densities

def visualize_mesh(mesh, path='mesh.png'):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    plt.savefig(path)