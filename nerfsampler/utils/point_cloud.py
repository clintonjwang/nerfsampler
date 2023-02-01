import open3d as o3d
import pdb, torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

from nerfsampler.utils import mesh

class OccupancyTester:
    # http://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html
    def __init__(self, pcd):
        self.mesh = mesh.pcd_to_mesh(pcd)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

    def get_mask_of_points_inside_pcd(self, query):
        pdb.set_trace()
        occupancy = self.scene.compute_occupancy(query)
        return occupancy == 1

class BBoxOccupancyTester:
    def __init__(self, pcd, eps=3e-4):
        self.bbox = pcd.min(dim=0).values-eps, pcd.max(dim=0).values+eps

    def get_mask_of_points_inside_pcd(self, query):
        return (query > self.bbox[0]).min(dim=-1).values & (query < self.bbox[1]).min(dim=-1).values

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

def plot_point_cloud(
    pc,
    grid_size: int = 1,
    fixed_bounds = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                pc = pc @ rotation

            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])

            if fixed_bounds is None:
                min_point = pc.min(0)
                max_point = pc.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

    return fig

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def torch_to_o3d(points: torch.Tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    return pcd

def downsample_pcd(points: torch.Tensor):
    pcd = torch_to_o3d(points)
    pcd, indices = pcd.voxel_down_sample_and_trace(voxel_size=0.02)
    return torch.tensor(np.asarray(pcd.points),
            device=points.device, dtype=points.dtype), indices
    
def select_largest_subset(points: torch.Tensor, k=30):
    # points (N,3) tensor
    p_npy = points.cpu().numpy()
    adj = kneighbors_graph(p_npy, k, include_self=False)
    n_components, labels = csgraph.connected_components(adj)
    if n_components == 1:
        return points
    ix = np.argmax((labels == i).sum() for i in range(n_components))
    mask = labels == ix
    return points[torch.tensor(mask, device=points.device)]

def refine_labels(points, init_labels):
    model = AgglomerativeClustering(
        linkage='single', connectivity=adj, n_clusters=k
    )
    model.fit(p_npy)
    labels = model.labels_
    ix = np.argmax((labels == i).sum() for i in range(k))
    mask = labels == ix
    pdb.set_trace()
    return points[torch.tensor(mask, device=points.device)]
