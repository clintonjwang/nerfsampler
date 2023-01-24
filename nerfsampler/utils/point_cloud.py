import open3d as o3d
import pdb, torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def select_largest_subset(points):
    # points (N,3)
    p_npy = points.cpu().numpy()
    adj = kneighbors_graph(p_npy, 30, include_self=False)
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

def downsample(points):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points.cpu().numpy())
    pcd, indices = pcd.voxel_down_sample_and_trace(voxel_size=0.02)
    return torch.tensor(np.asarray(pcd.points),
            device=points.device, dtype=points.dtype), indices