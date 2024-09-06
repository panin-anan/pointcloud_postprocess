import open3d as o3d
import numpy as np

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries(mesh, mesh_show_back_face=True)

def visualize_meshes_overlay(worn_meshes, desired_meshes):
    geometries = []
    
    for worn_mesh, desired_mesh in zip(worn_meshes, desired_meshes):
        # Add both worn and desired meshes to the list of geometries
        geometries.append(worn_mesh)
        geometries.append(desired_mesh)

    # Visualize the overlay of both sets of meshes
    o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)

def visualize_sub_section(sub_section):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors for each sub-curve
    geometries = []

    for idx, section in enumerate(sub_section):
        curve_pc = o3d.geometry.PointCloud()
        curve_pc.points = o3d.utility.Vector3dVector(np.asarray(section.points))
        curve_pc.paint_uniform_color(colors[idx % len(colors)])
        geometries.append(curve_pc)

    o3d.visualization.draw_geometries(geometries)


def project_worn_to_desired(worn_mesh, desired_mesh):
    """
    Project the worn mesh points onto the desired mesh and visualize the lost material.
    """
    worn_pcd = worn_mesh.sample_points_uniformly(number_of_points=10000)
    desired_pcd = desired_mesh.sample_points_uniformly(number_of_points=10000)

    distances = np.asarray(worn_pcd.compute_point_cloud_distance(desired_pcd))
    p2p_distances = o3d.geometry.KDTreeFlann(desired_pcd)

    worn_points = np.asarray(worn_pcd.points)
    projected_points = []

    for point in worn_points:
        _, idx, _ = p2p_distances.search_knn_vector_3d(point, 1)
        projected_points.append(desired_pcd.points[idx[0]])

    projected_points = np.asarray(projected_points)
    combined_points = np.vstack((worn_points, projected_points))

    lines = [[i, i + len(worn_points)] for i in range(len(worn_points))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(combined_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    max_distance = np.max(distances)
    colors = [[d / max_distance, 0, 1 - d / max_distance] for d in distances]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def visualize_lost_material(worn_meshes, desired_meshes):
    for i in range(len(worn_meshes)):
        lost_visualization = project_worn_to_desired(worn_meshes[i], desired_meshes[i])
        o3d.visualization.draw_geometries([worn_meshes[i], desired_meshes[i], lost_visualization])
