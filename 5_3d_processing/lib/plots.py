import k3d
import numpy as np

from .utils import pack_rgb


def plot_point_cloud(pcd, point_size=0.001, plot=None):
    if plot is None:
        plot = k3d.plot(camera_auto_fit=False)

    points = np.asarray(pcd.points, np.float32)
    colors = pack_rgb(np.asarray(pcd.colors))
    
    plot +=  k3d.points(points, colors, point_size=point_size,
                        shader="flat")
    return plot


def plot_mesh(mesh, plot=None):
    if plot is None:
        plot = k3d.plot(camera_auto_fit=False)

    vertices = np.array(mesh.vertices, np.float32)
    triangles = np.array(mesh.triangles, np.uint32)

    plot +=  k3d.mesh(vertices, triangles)

    return plot


def plot_voxel_grid(voxel_grid,  plot=None):
    if plot is None:
        plot = k3d.plot(camera_auto_fit=False)

    voxel_grid_inds = np.array([
        v.grid_index for v in voxel_grid.get_voxels()
    ])

    resolution = voxel_grid_inds.max() + 1
    voxels = np.zeros((resolution,)*3, np.uint8)

    i, j, k = voxel_grid_inds.T
    voxels[j, i, k] = 1
    
    plot += k3d.voxels(voxels, [0x00ff00])

    return plot