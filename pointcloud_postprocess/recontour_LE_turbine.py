import numpy as np
import open3d as o3d
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl

# -- Utility Functions for Point Cloud Processing --

def estimate_curvature(pcd, k_neighbors=30):
    """Estimate curvature for each point using eigenvalue decomposition."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    points = np.asarray(pcd.points)
    curvatures = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)
        neighbors = points[idx, :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

def detect_leading_edge_by_curvature(pcd, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=20, min_distance=40):
    """Detect leading edge points based on curvature and further refine them."""
    curvatures = estimate_curvature(pcd, k_neighbors=k_neighbors)
    lower_bound, upper_bound = curvature_threshold
    filtered_indices = np.where((curvatures >= lower_bound) & (curvatures <= upper_bound))[0]
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    refined_leading_edge_points = []
    for idx in filtered_indices:
        point = pcd.points[idx]
        [_, idx_neigh, _] = kdtree.search_radius_vector_3d(point, vicinity_radius)
        if len(idx_neigh) > 0:
            highest_curvature_idx = idx_neigh[np.argmax(curvatures[idx_neigh])]
            refined_leading_edge_points.append(pcd.points[highest_curvature_idx])
    
    # Remove points too close to each other
    filtered_leading_edge_points = []
    for point in refined_leading_edge_points:
        if len(filtered_leading_edge_points) == 0 or np.all(np.linalg.norm(np.array(filtered_leading_edge_points) - point, axis=1) >= min_distance):
            filtered_leading_edge_points.append(point)

    return np.array(filtered_leading_edge_points)

def fit_spline_to_leading_edge(leading_edge_points, smoothing_factor=1e-3):
    """Fit a spline to the detected leading edge points."""
    leading_edge_points = np.unique(np.asarray(leading_edge_points), axis=0)
    tck, u = splprep([leading_edge_points[:, 0], leading_edge_points[:, 1], leading_edge_points[:, 2]], s=smoothing_factor)
    return tck, u

def sample_spline(tck, num_points=100):
    """Sample points along the fitted spline."""
    u_fine = np.linspace(0, 1, num_points)
    sampled_points = splev(u_fine, tck)
    return np.vstack(sampled_points).T

# -- Visualization Functions --

def visualize_leading_edge_and_spline(pcd, leading_edge_points, spline_points):
    """Visualize the original point cloud, detected leading edge, and fitted spline."""
    leading_edge_pcd = o3d.geometry.PointCloud()
    leading_edge_pcd.points = o3d.utility.Vector3dVector(leading_edge_points)
    leading_edge_pcd.paint_uniform_color([1, 0, 0])

    spline_pcd = o3d.geometry.PointCloud()
    spline_pcd.points = o3d.utility.Vector3dVector(spline_points)
    spline_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd, leading_edge_pcd, spline_pcd])

def visualize_curvature_based_leading_edge(pcd, leading_edge_points):
    """Visualize original point cloud and detected leading edge points."""
    leading_edge_pcd = o3d.geometry.PointCloud()
    leading_edge_pcd.points = o3d.utility.Vector3dVector(leading_edge_points)
    leading_edge_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, leading_edge_pcd])

# -- Section Processing and Symmetry Adjustment --

def point_to_plane_distance(points, plane_point, plane_normal):
    """Calculate distance from points to a plane."""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    return np.abs(np.dot(points - plane_point, plane_normal))

def extract_points_on_plane(point_cloud, plane_point, plane_normal, threshold=0.4):
    """Extract points lying near a specified plane."""
    distances = point_to_plane_distance(np.asarray(point_cloud.points), plane_point, plane_normal)
    mask = distances < threshold
    points_on_plane = np.asarray(point_cloud.points)[mask]
    
    points_on_plane_cloud = o3d.geometry.PointCloud()
    points_on_plane_cloud.points = o3d.utility.Vector3dVector(points_on_plane)
    
    return points_on_plane_cloud

def slice_point_cloud_with_visualization(point_cloud, leading_edge_points, num_sections=10, threshold=0.1):
    """Slice the point cloud into sections using leading edge points."""
    vis_element = []
    sections = []
    
    for i in range(len(leading_edge_points) - 1):
        start_point = leading_edge_points[i]
        end_point = leading_edge_points[i + 1]
        for j in range(num_sections):
            t = j / num_sections
            section_point = (1 - t) * start_point + t * end_point
            flow_axis = end_point - start_point
            flow_axis /= np.linalg.norm(flow_axis)
            points_on_plane = extract_points_on_plane(point_cloud, section_point, flow_axis, threshold)
            if len(points_on_plane.points) > 0:
                points_on_plane.paint_uniform_color([0, 0, 0])
                vis_element.append(points_on_plane)
                sections.append(np.asarray(points_on_plane.points))

    o3d.visualization.draw_geometries(vis_element)
    return sections

def find_closest_leading_edge_point(section_points, leading_edge_points):
    """Find the closest point in section_points to any point in leading_edge_points."""
    min_distance = float('inf')
    closest_point = None
    for point in section_points:
        distances = np.linalg.norm(leading_edge_points - point, axis=1)
        closest_distance = np.min(distances)
        if closest_distance < min_distance:
            min_distance = closest_distance
            closest_point = point
    return closest_point

def adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, tolerance=1e-4, max_iterations=100):
    """Iteratively adjust the center and LE vector for symmetry."""
    center = initial_center
    LE_vector = leading_edge_point - center
    LE_vector /= np.linalg.norm(LE_vector)

    iteration = 0
    while iteration < max_iterations:
        perpendicular_vector = np.array([LE_vector[1], -LE_vector[0], 0])
        perpendicular_vector /= np.linalg.norm(perpendicular_vector)
        points_relative_to_LE = section_points - center
        distances_to_plane = np.dot(points_relative_to_LE, LE_vector)[:, None] * LE_vector
        projected_points = points_relative_to_LE - distances_to_plane

        distances_left = []
        distances_right = []
        for point, projected_point in zip(section_points, projected_points):
            direction = point - center
            if np.dot(direction, LE_vector) > 0:
                perpendicular_distance = np.dot(projected_point, perpendicular_vector)
                if perpendicular_distance > 0:
                    distances_right.append(perpendicular_distance)
                else:
                    distances_left.append(-perpendicular_distance)

        avg_left_distance = np.mean(distances_left) if distances_left else 0
        avg_right_distance = np.mean(distances_right) if distances_right else 0
        
        if abs(avg_left_distance - avg_right_distance) < tolerance:
            break

        offset = (avg_right_distance - avg_left_distance) / 2
        center += perpendicular_vector * offset
        LE_vector = leading_edge_point - center
        LE_vector /= np.linalg.norm(LE_vector)
        iteration += 1

    return center, LE_vector

def recontour_LE_sections(LE_sections, leading_edge_points, target_radius=2, tolerance=1e-4):
    """Recontour leading edge sections based on adjusted symmetry."""
    recontoured_sections = []
    for section_points in LE_sections:
        leading_edge_point = find_closest_leading_edge_point(section_points, leading_edge_points)
        initial_center = np.mean(section_points, axis=0)
        adjusted_center, LE_vector = adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, tolerance)
        
        recontoured_section = []
        for point in section_points:
            direction = point - adjusted_center
            if np.dot(direction, LE_vector) > 0:
                projection_onto_LE = np.dot(direction, LE_vector) * LE_vector
                projection_distance_squared = np.dot(projection_onto_LE, projection_onto_LE)
                if projection_distance_squared < target_radius**2:
                    arc_distance = np.sqrt(target_radius**2 - projection_distance_squared)
                    perpendicular_direction = direction - projection_onto_LE
                    perpendicular_direction /= np.linalg.norm(perpendicular_direction) or 1
                    new_point = adjusted_center + projection_onto_LE + perpendicular_direction * arc_distance
                    recontoured_section.append(new_point)
                else:
                    direction_length = np.linalg.norm(direction)
                    new_point = adjusted_center + (direction / direction_length) * target_radius if direction_length != 0 else adjusted_center
                    recontoured_section.append(new_point)
            else:
                recontoured_section.append(point)
        recontoured_sections.append(recontoured_section)
    
    # Visualization (original sections and recontoured sections)
    vis_elements = []
    for section_id, section_points in enumerate(LE_sections):
        original_points = o3d.geometry.PointCloud()
        original_points.points = o3d.utility.Vector3dVector(section_points)
        original_points.paint_uniform_color([1, 0, 0])  # Red for original points
        vis_elements.append(original_points)

        recontoured_points = o3d.geometry.PointCloud()
        recontoured_points.points = o3d.utility.Vector3dVector(recontoured_sections[section_id])
        recontoured_points.paint_uniform_color([0, 1, 0])  # Green for recontoured points
        vis_elements.append(recontoured_points)

    o3d.visualization.draw_geometries(vis_elements, window_name="Original and Recontoured Sections", width=800, height=600)


    return recontoured_sections

# -- Surface Mesh Generation --

def smooth_sections(sections):
    """Perform smoothing of the sections using spline interpolation."""
    smoothed_sections = []
    for section in sections:
        section = np.array(section)
        tck, u = splprep(section.T, s=0)
        u_fine = np.linspace(0, 1, len(section))
        smoothed_sections.append(np.array(splev(u_fine, tck)).T)
    
    return smoothed_sections

def match_points_between_sections(section_1, section_2):
    """Match points between two sections using nearest-neighbor search."""
    tree = cKDTree(section_2)
    distances, indices = tree.query(section_1)
    return [(i, indices[i]) for i in range(len(section_1))]

def create_surface_mesh_from_sections(sections):
    """Create a surface mesh from section lines using nearest-neighbor matching."""
    vertices = []
    triangles = []

    for i in range(len(sections) - 1):
        section_1, section_2 = sections[i], sections[i + 1]
        matched_pairs = match_points_between_sections(section_1, section_2)
        for j, (p1, p2) in enumerate(matched_pairs):
            next_p1 = (j + 1) % len(section_1)
            next_p2 = matched_pairs[next_p1][1] if next_p1 < len(matched_pairs) else 0
            v0, v1, v2, v3 = section_1[p1], section_1[next_p1], section_2[p2], section_2[next_p2]
            idx0 = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            triangles.extend([[idx0, idx0 + 1, idx0 + 2], [idx0 + 1, idx0 + 3, idx0 + 2]])

    surface_mesh = o3d.geometry.TriangleMesh()
    surface_mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    surface_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    surface_mesh.compute_vertex_normals()
    
    return surface_mesh


# Main
# Load mesh to mesh processor           comment one out depending on data type
mstore = MeshProcessor()
#mstore.mesh1 = mstore.load_mesh(1)
mstore.mesh1_pcl = mstore.load_mesh(1)


print("Mesh Loaded")
# Sample points
#mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=40000)

#For Cropping Point Cloud
'''
print("Press 'K' to go to selection mode, then use CTRL+LMB to *draw* a polygon for cropping (click multiple times).")
print("Press 'C' to crop and select only point in polygon, Press S and put name to save file")
vis = o3d.visualization.VisualizerWithEditing(-1, False, "")
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(mstore.mesh1_pcl)
# Run the visualizer, allowing the user to draw polygons and crop the point cloud
vis.run() 
vis.destroy_window()
# After drawing the polygon, the cropped points are saved in the visualizer's memory
cropped_indices = vis.get_picked_points()
# Extract the selected points
cropped_pcd = mstore.mesh1_pcl.select_by_index(cropped_indices)

# Save the cropped point cloud
o3d.io.write_point_cloud("cropped_point_cloud.ply", cropped_pcd)
'''


curvature_array = estimate_curvature(mstore.mesh1_pcl)
leading_edge_points = detect_leading_edge_by_curvature(mstore.mesh1_pcl, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=10, min_distance=20)
#tck, u = fit_spline_to_leading_edge(leading_edge_points)

# Sample points along the spline for visualization
#spline_points = sample_spline(tck, num_points=1000)

# Visualize the leading edge points and the spline
#visualize_leading_edge_and_spline(mstore.mesh1_pcl, leading_edge_points, spline_points)


LE_sections = slice_point_cloud_with_visualization(mstore.mesh1_pcl, leading_edge_points, num_sections=6, threshold=0.5)


recontoured_LE_sections = recontour_LE_sections(LE_sections, leading_edge_points, target_radius=14)
smoothed_sections = smooth_sections(recontoured_LE_sections)

# Example usage (assuming sections is a list of NumPy arrays representing cross-sections)

turbine_surface = create_surface_mesh_from_sections(recontoured_LE_sections)

o3d.visualization.draw_geometries([turbine_surface], window_name="Turbines", width=800, height=600)

