import open3d as o3d
import pdb, torch
import numpy as np

def pcd_to_mesh(pcd, filter_low_density=True):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    if filter_low_density:
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh, densities

def is_point_inside_mesh(point, mesh):
    return
