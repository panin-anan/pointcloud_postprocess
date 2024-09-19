import numpy as np
import open3d as o3d
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl, visualize_pcl_overlay

# -- Utility Functions for Point Cloud Processing --

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

def adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, vis_elements, tolerance=1e-3, max_iterations=5000):
    
    """Iteratively adjust the center and LE vector for symmetry."""
    
    center = initial_center
    LE_vector = leading_edge_point - center
    LE_vector /= np.linalg.norm(LE_vector)  # Normalize the leading edge vector

    initial_center_point = o3d.geometry.PointCloud()
    initial_center_point.points = o3d.utility.Vector3dVector([initial_center])
    initial_center_point.paint_uniform_color([0, 0, 0])  # Black for initial center
    vis_elements.append(initial_center_point)

    iteration = 0
    while iteration < max_iterations:
        # 1. Find two vectors orthogonal to LE_vector using Gram-Schmidt process
        arbitrary_vector = np.array([1, 0, 0]) if np.abs(LE_vector[0]) < 0.9 else np.array([0, 1, 0])
        
        # First orthogonal vector
        perp_vector1 = arbitrary_vector - np.dot(arbitrary_vector, LE_vector) * LE_vector
        perp_vector1 /= np.linalg.norm(perp_vector1)
        
        # Second orthogonal vector
        perp_vector2 = np.cross(LE_vector, perp_vector1)
        perp_vector2 /= np.linalg.norm(perp_vector2)
        
        points_relative_to_LE = section_points - center
        distances_to_plane = np.dot(points_relative_to_LE, LE_vector)[:, None] * LE_vector
        projected_points = points_relative_to_LE - distances_to_plane  # Projected points in the cross-section plane

        distances_left = []
        distances_right = []
        
        for point, projected_point in zip(section_points, projected_points):
            direction = point - center
            # Perpendicular distance to the cross-section plane
            perpendicular_distance = np.dot(projected_point, perp_vector2)

            # 3. Sort points into left and right based on their position relative to LE_vector
            if np.dot(direction, LE_vector) > 0:  # Points above the center
                if perpendicular_distance > 0:
                    distances_right.append(perpendicular_distance)
                elif perpendicular_distance < 0:
                    distances_left.append(-perpendicular_distance)

        # 4. Compute the average distances
        if distances_left and distances_right:
            avg_left_distance = np.mean(distances_left)
            avg_right_distance = np.mean(distances_right)
        else:
            avg_left_distance, avg_right_distance = 0, 0
        
        # 5. Check for symmetry and break if within tolerance
        distance_diff = avg_right_distance - avg_left_distance
        if abs(distance_diff) < tolerance:
            break

        # 6. Adjust the center based on the difference
        offset = distance_diff / 2
        # Adjust center along the single perpendicular vector in the cross-sectional plane
        center += perp_vector2 * offset
        
        # Recalculate LE_vector based on the updated center
        LE_vector = leading_edge_point - center
        LE_vector /= np.linalg.norm(LE_vector)

        iteration += 1

    final_center_point = o3d.geometry.PointCloud()
    final_center_point.points = o3d.utility.Vector3dVector([center])
    final_center_point.paint_uniform_color([0, 0, 1])  # Blue for final center
    vis_elements.append(final_center_point)

    LE_point = o3d.geometry.PointCloud()
    LE_point.points = o3d.utility.Vector3dVector([leading_edge_point])
    LE_point.paint_uniform_color([0, 0, 1])  # blue LE point
    vis_elements.append(LE_point)

    points = [center, leading_edge_point]  # Two points for the line
    lines = [[0, 1]]  # Single line connecting the two points
    colors = [[0, 0, 1]]  # Blue color for the line

    # Create a LineSet object for visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis_elements.append(line_set)


    return center, LE_vector, vis_elements


def recontour_LE_sections(LE_sections, leading_edge_points, initial_target_radius=0.01, tolerance=1e-3):
    """Recontour leading edge sections, ensuring the recontoured radius does not exceed the original distance from the adjusted center."""
    
    recontoured_sections = []
    vis_elements = []

    for section_points in LE_sections:
        # 1. Determine the leading edge vector and center
        leading_edge_point = find_closest_leading_edge_point(section_points, leading_edge_points)
        initial_center = np.mean(section_points, axis=0)
        adjusted_center, LE_vector, vis_elements = adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, vis_elements, tolerance)
        shift_factor = 0.2
        shift_down_length = shift_factor * np.linalg.norm(leading_edge_point - adjusted_center) 

        recontoured_section = []
        for point in section_points:
            # Calculate the vector from the center to the current point
            direction = point - adjusted_center  

            # 2. Determine the area of recontouring (points above the center along the LE vector)
            if np.dot(direction, LE_vector) > 0:
                # Perpendicular direction to the leading edge vector
                projection_onto_LE = np.dot(direction, LE_vector) * LE_vector
                perpendicular_direction = direction - projection_onto_LE
                original_distance = np.linalg.norm(perpendicular_direction)
                perpendicular_distance_squared = np.dot(perpendicular_direction, perpendicular_direction)

                # 3. Create new points following the parabola algorithm without exceeding original distance
                target_radius = initial_target_radius
                arc_distance = -target_radius * perpendicular_distance_squared

                # Normalize the perpendicular direction
                perpendicular_direction /= np.linalg.norm(perpendicular_direction)

                
                # Adjust the target radius if the new point exceeds the original distance
                while np.abs(arc_distance) > original_distance:
                    target_radius -= tolerance  # Reduce the radius to fit within the original distance
                    arc_distance = -target_radius * perpendicular_distance_squared

                    # Prevent target radius from becoming negative or too small
                    if target_radius <= 0:
                        arc_distance = original_distance
                        break
                
                # Create the new recontoured point
                new_point = adjusted_center + perpendicular_direction * arc_distance + projection_onto_LE
                new_point -= LE_vector * shift_down_length

                # 4. Remove old points above the new point profile
                # If the original point is higher than the new point (along the LE vector), discard it
                old_point_distance = np.linalg.norm(point - adjusted_center)
                new_point_distance = np.linalg.norm(new_point - adjusted_center)
                if np.dot((point - adjusted_center), LE_vector) > np.dot((new_point - adjusted_center), LE_vector) and new_point_distance <= old_point_distance:
                    recontoured_section.append(new_point)
                else:
                    recontoured_section.append(point)
            else:
                # Leave points below the adjusted center unchanged
                recontoured_section.append(point)
        
        recontoured_sections.append(recontoured_section)
    
    # Visualization (original sections and recontoured sections)
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

mstore.load_mesh(1)
#mstore.load_mesh(2)

if mstore.mesh1_pcl == None:
    mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=60000)
#if mstore.mesh2_pcl == None:
#    mstore.mesh2_pcl = mstore.mesh2.sample_points_poisson_disk(number_of_points=60000)

#curvature_array = mstore.estimate_curvature(mstore.mesh1_pcl)
mstore.mesh1_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh1_pcl)
#mstore.mesh2_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh2_pcl)
#tck, u = fit_spline_to_leading_edge(mstore.mesh1_LE_points)

# Sample points along the spline for visualization
#spline_points = sample_spline(tck, num_points=1000)

# Visualize the leading edge points and the spline
visualize_curvature_based_leading_edge(mstore.mesh1_pcl, mstore.mesh1_LE_points)


LE_sections_mesh1 = slice_point_cloud_with_visualization(mstore.mesh1_pcl, mstore.mesh1_LE_points, num_sections=1, threshold=0.8)
#LE_sections_mesh2 = slice_point_cloud_with_visualization(mstore.mesh2_pcl, mstore.mesh2_LE_points, num_sections=1, threshold=0.8)

#visualize_pcl_overlay(LE_sections_mesh1, LE_sections_mesh2)

recontoured_LE_sections = recontour_LE_sections(LE_sections_mesh1, mstore.mesh1_LE_points, initial_target_radius=3)


smoothed_sections = smooth_sections(recontoured_LE_sections)
turbine_surface = create_surface_mesh_from_sections(recontoured_LE_sections)

o3d.visualization.draw_geometries([turbine_surface], window_name="Turbines", width=800, height=600)

