import numpy as np
import open3d as o3d
import gc
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from mesh_processor import MeshProcessor
from mesh_visualizer import MeshVisualizer

# -- Section Processing and Symmetry Adjustment --

def point_to_plane_distance(points, plane_point, plane_normal):
    """Calculate distance from points to a plane."""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    return np.abs(np.dot(points - plane_point, plane_normal))

def extract_points_on_plane(point_cloud, plane_point, plane_normal, threshold=0.0004):
    """Extract points lying near a specified plane."""
    distances = point_to_plane_distance(np.asarray(point_cloud.points), plane_point, plane_normal)
    mask = distances < threshold
    points_on_plane = np.asarray(point_cloud.points)[mask]
    
    points_on_plane_cloud = o3d.geometry.PointCloud()
    points_on_plane_cloud.points = o3d.utility.Vector3dVector(points_on_plane)
    
    return points_on_plane_cloud

def filter_and_project_sections(LE_sections_mesh1, LE_sections_mesh2, threshold=0.050, point_threshold=0.005):
    """Filter sections that are not close to each other and project the ones close to the same plane using mesh1 as the base."""

    filtered_sections_mesh1 = []
    filtered_sections_mesh2 = []
    
    #vis_elements = []

    for sec1, sec2 in zip(LE_sections_mesh1, LE_sections_mesh2):
        # Calculate the mean point of each section
        mean_sec1 = np.mean(sec1, axis=0)
        mean_sec2 = np.mean(sec2, axis=0)
        
        # Calculate distance between the sections
        distance = np.linalg.norm(mean_sec1 - mean_sec2)
        
        # Filter sections based on the threshold
        if distance <= threshold:
            filtered_sec1 = []
            filtered_sec2 = []
            
            # Project sec2 points onto the plane of sec1
            sec1_plane_normal = calculate_plane_normal(sec1)
            projected_sec2 = project_points_to_plane(sec2, mean_sec1, sec1_plane_normal)
            
            tree_sec1 = cKDTree(sec1)
            tree_sec2 = cKDTree(projected_sec2)

            for point in sec1:
                dist, _ = tree_sec2.query(point)
                if dist <= point_threshold:
                    filtered_sec1.append(point)

            for point in projected_sec2:
                dist, _ = tree_sec1.query(point)
                if dist <= point_threshold:
                    filtered_sec2.append(point)

            filtered_sections_mesh1.append(np.array(filtered_sec1))
            filtered_sections_mesh2.append(np.array(filtered_sec2))
    
    return filtered_sections_mesh1, filtered_sections_mesh2


def calculate_plane_normal(section_points):
    """Calculate the normal vector of the plane defined by the section points."""
    pca = PCA(n_components=3)
    pca.fit(section_points)
    normal = pca.components_[-1]  # The normal is the last component
    return normal


def project_points_to_plane(points, plane_point, plane_normal):
    """Project points onto a plane defined by a point and a normal."""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projected_points = points - np.dot(points - plane_point, plane_normal)[:, None] * plane_normal
    return projected_points


def rotate_point_cloud(pcd, theta_x=0, theta_y=0, theta_z=0):
    """
    Rotate the point cloud using independent rotation angles for each axis.
    
    Parameters:
    - pcd: The input Open3D point cloud object.
    - theta_x: Rotation angle around the X-axis (in radians).
    - theta_y: Rotation angle around the Y-axis (in radians).
    - theta_z: Rotation angle around the Z-axis (in radians).
    
    Returns:
    - rotated_pcd: A new point cloud with the combined rotation applied.
    """
    # Rotation matrix for X-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Rotation matrix for Y-axis
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Rotation matrix for Z-axis
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: First rotate around X, then Y, then Z
    R = R_z @ R_y @ R_x
    
    # Apply the combined rotation to the point cloud
    rotated_pcd = pcd.rotate(R, center=(0, 0, 0))  # Rotate around the origin
    
    return rotated_pcd


# # # #  Turbine Section based on major axis  # # # #

def slice_point_cloud_along_axis(pcd, flow_axis = 'y', num_sections = 10, threshold=0.0002):
    """Slice the point cloud into sections using leading edge points."""
    vis_element = []

    points = np.asarray(pcd.points)

    # Map the flow axis to the appropriate index in the point array
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if flow_axis not in axis_map:
        raise ValueError("Invalid flow axis. Must be 'x', 'y', or 'z'.")
    
    flow_idx = axis_map[flow_axis]

    # Get the min and max values along the flow axis
    flow_min = np.min(points[:, flow_idx])
    flow_max = np.max(points[:, flow_idx])

    # Generate equally spaced slicing planes along the flow axis
    section_positions = np.linspace(flow_min, flow_max, num_sections)

    sections = []

    # Define a normal vector for each plane (based on the flow axis)
    normal = np.zeros(3)
    normal[flow_idx] = 1  # Normal is aligned with the flow axis (e.g., for 'y', normal = [0, 1, 0])

    for section_pos in section_positions:
        # Define the plane equation: ax + by + cz + d = 0, where d = -dot(normal, point_on_plane)
        point_on_plane = np.zeros(3)
        point_on_plane[flow_idx] = section_pos
        d = -np.dot(normal, point_on_plane)

        # Project points onto the plane and calculate distances
        distances = np.dot(points, normal) + d
        mask = np.abs(distances) < threshold  # Keep points close to the plane

        section_points = points[mask]
        if section_points.shape[0] > 0:
            sections.append(section_points)

    return sections


def detect_leading_edge_by_maxima(sections, leading_edge_axis='z'):
    """
    Find the leading edge point based on a section of the point cloud.
    The leading edge is assumed to be the point with the largest value along the specified axis.
    
    Parameters:
    - sections: List of 2D arrays, where each array is a section of the point cloud (Nx3).
    - leading_edge_axis: The axis to search for the maxima ('x', 'y', or 'z').
    
    Returns:
    - le_points: List of leading edge point coordinates (x, y, z) for each section.
    """
    # Determine which axis to use for leading edge detection
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if leading_edge_axis not in axis_map:
        raise ValueError("Invalid leading edge axis. Must be 'x', 'y', or 'z'.")
    
    idx_le = axis_map[leading_edge_axis]  # Get the axis index (0 for 'x', 1 for 'y', 2 for 'z')

    le_points = []

    # Loop over each section and detect the leading edge
    for section in sections:
        # Find the index of the point with the maximum value along the selected axis
        max_le_idx = np.argmax(section[:, idx_le])
        
        # The leading edge point is the point at this index
        le_point = section[max_le_idx]
        
        # Append the leading edge point to the list
        le_points.append(le_point)
    
    return le_points


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

def adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, vis_elements, tolerance=1e-7, max_iterations=5000):
    
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


def recontour_LE_sections(LE_sections, leading_edge_points, initial_target_parabolic_parameter=1000, tolerance=1e-0, max_iterations=1000):
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

                #TO DO: Change perpendicular distance algorithm to first obtain maximum at the center point and scale the value down parabolicaly from there

                # Perpendicular direction to the leading edge vector
                projection_onto_LE = np.dot(direction, LE_vector) * LE_vector
                perpendicular_direction = direction - projection_onto_LE
                original_distance = np.linalg.norm(perpendicular_direction)
                perpendicular_distance_squared = np.dot(perpendicular_direction, perpendicular_direction)

                # 3. Create new points following the parabola algorithm without exceeding original distance
                target_radius = initial_target_parabolic_parameter
                arc_distance = -target_radius * perpendicular_distance_squared
                print(arc_distance)
                # Normalize the perpendicular direction
                perpendicular_direction /= np.linalg.norm(perpendicular_direction)

                
                # Adjust the target radius if the new point exceeds the original distance
                iteration_count = 0
                while np.abs(arc_distance) > np.abs(original_distance):
                    target_radius -= tolerance  # Reduce the radius to fit within the original distance
                    arc_distance = -target_radius * perpendicular_distance_squared

                    iteration_count += 1
                    if iteration_count >= max_iterations:
                        print(f"Reached maximum iterations: {max_iterations}, breaking loop.")
                        break  # Stop if maximum iterations is reached
                
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


def main():
    # Load mesh to mesh processor           comment one out depending on data type
    mstore = MeshProcessor()
    mvis = MeshVisualizer()

    mstore.load_mesh(1)
    mstore.load_mesh(2)

    if mstore.mesh1_pcl == None:
        mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=60000)
    if mstore.mesh2_pcl == None:
        mstore.mesh2_pcl = mstore.mesh2.sample_points_poisson_disk(number_of_points=60000)

    scale_factor = 1.0
    thresholds = {
        "plane_threshold": 0.0001 * scale_factor,
        "curvature_threshold": 0.005,                   #manually input
        "project_tolerance": 0.0002 * scale_factor,
        "point_threshold": 0.005 * scale_factor,
        "vicinity_radius": 0.004 * scale_factor,
        "min_distance": 0.004 * scale_factor,
        "tolerance": 1e-8 * scale_factor,
    }

    theta_x = np.deg2rad(-15)  # degrees around X-axis
    theta_y = np.deg2rad(31)  # degrees around Y-axis
    theta_z = np.deg2rad(-12)  # degrees around Z-axis

    # Rotate the point cloud
    mstore.mesh1_pcl = rotate_point_cloud(mstore.mesh1_pcl, theta_x, theta_y, theta_z)
    mstore.mesh2_pcl = rotate_point_cloud(mstore.mesh2_pcl, theta_x, theta_y, theta_z)
    
    #Create LE sections
    LE_sections_mesh1 = slice_point_cloud_along_axis(mstore.mesh1_pcl, flow_axis = 'y', num_sections = 10, threshold=thresholds["project_tolerance"])
    LE_sections_mesh2 = slice_point_cloud_along_axis(mstore.mesh2_pcl, flow_axis = 'y', num_sections = 10, threshold=thresholds["project_tolerance"])

    mvis.visualize_pcl_overlay(LE_sections_mesh1, LE_sections_mesh2)

    mstore.mesh1_LE_points = detect_leading_edge_by_maxima(LE_sections_mesh1, leading_edge_axis='x')
    mstore.mesh2_LE_points = detect_leading_edge_by_maxima(LE_sections_mesh2, leading_edge_axis='x')


    mvis.visualize_sections_with_leading_edges(LE_sections_mesh1, mstore.mesh1_LE_points)

    '''
    LE_sections_mesh1, LE_sections_mesh2 = filter_and_project_sections(LE_sections_mesh1, LE_sections_mesh2, threshold=thresholds["project_tolerance"], point_threshold=thresholds["point_threshold"])
    mvis.visualize_pcl_overlay(LE_sections_mesh1, LE_sections_mesh2)
    '''

    recontoured_LE_sections = recontour_LE_sections(LE_sections_mesh1, mstore.mesh1_LE_points, initial_target_parabolic_parameter=1000)

    


if __name__ == "__main__":
    main()