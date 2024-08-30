# mesh_calculations.py

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, Delaunay

def calculate_lost_volume(mesh_before, mesh_after):
    mesh_before.compute_vertex_normals()
    mesh_after.compute_vertex_normals()

    pcd_before = mesh_before.sample_points_uniformly(number_of_points=100000)
    pcd_after = mesh_after.sample_points_uniformly(number_of_points=100000)

    distances = pcd_after.compute_point_cloud_distance(pcd_before)
    distances = np.asarray(distances)

    reference_area = mesh_after.get_surface_area()
    volume_lost = np.mean(distances) * reference_area

    return volume_lost


def filter_unchangedpointson_mesh(mesh_before, mesh_after, threshold=0.1):
    points_before = np.asarray(mesh_before.vertices)
    points_after = np.asarray(mesh_after.vertices)

    kdtree_before = cKDTree(points_before)
    distances, indices = kdtree_before.query(points_after)

    unchanged_indices = np.where(distances < threshold)[0]
    changed_indices = np.where(distances >= threshold)[0]

    unchanged_vertices = points_after[unchanged_indices]
    changed_vertices = points_after[changed_indices]

    if len(unchanged_vertices) >= 3:
        delaunay_unchanged = Delaunay(unchanged_vertices[:, :2])
        unchanged_triangles = delaunay_unchanged.simplices
    else:
        unchanged_triangles = []

    if len(changed_vertices) >= 3:
        delaunay_changed = Delaunay(changed_vertices[:, :2])
        changed_triangles = delaunay_changed.simplices
    else:
        changed_triangles = []

    mesh_unchanged = o3d.geometry.TriangleMesh()
    mesh_unchanged.vertices = o3d.utility.Vector3dVector(unchanged_vertices)
    mesh_unchanged.triangles = o3d.utility.Vector3iVector(unchanged_triangles)

    mesh_changed = o3d.geometry.TriangleMesh()
    mesh_changed.vertices = o3d.utility.Vector3dVector(changed_vertices)
    mesh_changed.triangles = o3d.utility.Vector3iVector(changed_triangles)

    return mesh_unchanged, mesh_changed


def calculate_lost_thickness(mesh_before, changed_mesh_after, lost_volume):
    reference_area = changed_mesh_after.get_surface_area()
    lost_thickness = lost_volume / reference_area
    
    return lost_thickness


def compute_average_x(mesh):
    vertices = np.asarray(mesh.vertices)
    average_x = np.mean(vertices[:, 0])
    return average_x


def compute_average_y(mesh):
    vertices = np.asarray(mesh.vertices)
    average_y = np.mean(vertices[:, 1])
    return average_y


def compute_average_z(mesh):
    vertices = np.asarray(mesh.vertices)
    average_z = np.mean(vertices[:, 2])
    return average_z


def calculate_curvature(mesh):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    L = np.zeros(vertices.shape)
    area_weight = np.zeros(vertices.shape[0])

    for tri in triangles:
        i1, i2, i3 = tri
        v1 = vertices[i1]
        v2 = vertices[i2]
        v3 = vertices[i3]

        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3

        angle1 = np.arccos(np.dot(e1, -e3) / (np.linalg.norm(e1) * np.linalg.norm(e3)))
        angle2 = np.arccos(np.dot(e2, -e1) / (np.linalg.norm(e2) * np.linalg.norm(e1)))
        angle3 = np.pi - angle1 - angle2

        cot1 = 1 / np.tan(angle1)
        cot2 = 1 / np.tan(angle2)
        cot3 = 1 / np.tan(angle3)

        L[i1] += cot3 * (v3 - v2) + cot2 * (v2 - v3)
        L[i2] += cot1 * (v1 - v3) + cot3 * (v3 - v1)
        L[i3] += cot2 * (v2 - v1) + cot1 * (v1 - v2)

        area = np.linalg.norm(np.cross(e1, -e3)) / 2
        area_weight[i1] += area
        area_weight[i2] += area
        area_weight[i3] += area

    mean_curvature = np.linalg.norm(L, axis=1) / (2 * area_weight)
    overall_curvature = np.mean(np.abs(mean_curvature))

    return mean_curvature, overall_curvature


def calculate_point_density(mesh):
    num_points = len(mesh.vertices)
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_volume = bbox.volume()

    point_density = num_points / bbox_volume
    
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_resolution = np.mean(distances)

    return num_points, point_density, avg_resolution
