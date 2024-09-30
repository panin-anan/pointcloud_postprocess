import numpy as np
import open3d as o3d
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from mesh_processor import MeshProcessor
from mesh_visualizer import MeshVisualizer
from grindparam_predictor import load_data, preprocess_data, train_svr, evaluate_model, create_grind_model, predict_grind_param

import os
import pandas as pd


# # # #  Turbine Section based on major axis  # # # #

def slice_point_cloud_along_axis(pcd, flow_axis = 'y', num_sections = 10, threshold=0.0002):
    """Slice the point cloud into sections using leading edge points."""
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
    section_length = (flow_max - flow_min) / num_sections
    section_positions = np.linspace(flow_min+(section_length/2), flow_max-(section_length/2), num_sections)

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

    return sections, section_length


#TO DO: Implement along leading edge sectioning instead of using major axes. (Can use below two functions)

def extract_points_on_plane(point_cloud, plane_point, plane_normal, threshold=0.0004):
    """Extract points lying near a specified plane."""
    distances = point_to_plane_distance(np.asarray(point_cloud.points), plane_point, plane_normal)
    mask = distances < threshold
    points_on_plane = np.asarray(point_cloud.points)[mask]
    
    points_on_plane_cloud = o3d.geometry.PointCloud()
    points_on_plane_cloud.points = o3d.utility.Vector3dVector(points_on_plane)
    
    return points_on_plane_cloud

def slice_point_cloud_along_leading_edge(point_cloud, leading_edge_points, num_sections=10, threshold=0.0004):
    """Slice the point cloud into sections using leading edge points."""
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

    #o3d.visualization.draw_geometries(vis_element)
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

def adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, vis_elements, tolerance=1e-6, max_iterations=5000):
    
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


def recontour_LE_sections(LE_sections, leading_edge_points, target_parabolic_parameter=3):
    """Recontour leading edge sections, ensuring the recontoured radius does not exceed the original distance from the adjusted center."""
    
    recontoured_sections = []
    area_removals = []
    vis_elements = []

    for section_index, section_points in enumerate(LE_sections):
        # 1. Determine the leading edge vector and center
        leading_edge_point = find_closest_leading_edge_point(section_points, leading_edge_points)
        initial_center = np.mean(section_points, axis=0)
        adjusted_center, LE_vector, vis_elements = adjust_center_and_le_for_symmetry(section_points, leading_edge_point, initial_center, vis_elements)
        shift_factor = 0.6
        peak_distance = np.linalg.norm(leading_edge_point - adjusted_center) 
        shift_down_length = shift_factor * peak_distance

        # Calculate the maximum perpendicular distance at the center
        max_perpendicular_distance_left = 0
        max_perpendicular_distance_right = 0

        for point in section_points:
            direction = point - adjusted_center
            if np.dot(direction, LE_vector) > 0:
                projection_onto_LE = np.dot(direction, LE_vector) * LE_vector
                perpendicular_direction = direction - projection_onto_LE
                perpendicular_distance = np.linalg.norm(perpendicular_direction)

                # Determine if the point is on the left or right side of the LE vector
                # Cross product of LE_vector and perpendicular_direction gives a vector whose sign indicates left or right
                cross_product = np.cross(LE_vector, perpendicular_direction)
                
                if cross_product[2] > 0:  # Assume the z-component determines left/right
                    max_perpendicular_distance_left = max(max_perpendicular_distance_left, perpendicular_distance)
                else:
                    max_perpendicular_distance_right = max(max_perpendicular_distance_right, perpendicular_distance)

        #use the lower value between the two
        max_perpendicular_distance = min(max_perpendicular_distance_left, max_perpendicular_distance_right)


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
                distance_along_LE = np.linalg.norm(projection_onto_LE)
    
                original_distance = np.linalg.norm(perpendicular_direction)
                perpendicular_distance_squared = np.dot(perpendicular_direction, perpendicular_direction)

                scaling_factor = (1 - (distance_along_LE / peak_distance) ** 2)
                perpendicular_distance_new = target_parabolic_parameter * scaling_factor * max_perpendicular_distance

                # Normalize the perpendicular direction
                if np.linalg.norm(perpendicular_direction) > 0:
                    perpendicular_direction /= np.linalg.norm(perpendicular_direction)

                
                perp_shift = 0.05 * max_perpendicular_distance

                # If the new perpendicular distance exceeds the original distance, adjust inward
                if np.abs(perpendicular_distance_new) > np.abs(original_distance):
                    # Move the point inward by reducing the perpendicular distance
                    perpendicular_distance_new = original_distance - perp_shift
                
                # Create the new recontoured point
                new_point = adjusted_center + perpendicular_distance_new * perpendicular_direction + projection_onto_LE
                new_point -= LE_vector * shift_down_length

                '''
                # 4. Remove old points above the new point profile
                # If the original point is higher than the new point (along the LE vector), discard it
                old_point_distance = np.linalg.norm(point - adjusted_center)
                new_point_distance = np.linalg.norm(new_point - adjusted_center)
                if np.dot((point - adjusted_center), LE_vector) > np.dot((new_point - adjusted_center), LE_vector) and new_point_distance <= old_point_distance:
                    recontoured_section.append(new_point)
                else:
                    recontoured_section.append(point)
                '''
                recontoured_section.append(new_point)
            else:
                # Leave points below the adjusted center unchanged
                recontoured_section.append(point)
        
        recontoured_sections.append(recontoured_section)

        # 3. Calculate the area between the original section and the recontoured section
        # Assuming LE_vector is the direction to project onto
        area, vis_elements = calculate_area_between_points(np.array(recontoured_section), np.array(section_points), LE_vector, adjusted_center, vis_elements)
        area_removals.append({
            'section_index': section_index,
            'areas': area  # This is already a dictionary with sub_section_idx_1 and sub_section_idx_2
        })
    
    # Visualization (original sections and recontoured sections)
    for section_id, section_points in enumerate(LE_sections):
        
        '''
        original_points = o3d.geometry.PointCloud()
        original_points.points = o3d.utility.Vector3dVector(section_points)
        original_points.paint_uniform_color([1, 0, 0])  # Red for original points
        vis_elements.append(original_points)
        
        recontoured_points = o3d.geometry.PointCloud()
        recontoured_points.points = o3d.utility.Vector3dVector(recontoured_sections[section_id])
        recontoured_points.paint_uniform_color([0, 1, 0])  # Green for recontoured points
        vis_elements.append(recontoured_points)
        '''

    o3d.visualization.draw_geometries(vis_elements, window_name="Original and Recontoured Sections", width=800, height=600)


    return recontoured_sections, np.array(area_removals)


def separate_sides(points, LE_vector, adjusted_center, vis_elements):
    """Separate points into left and right sides based on the dot product with a perpendicular vector."""
    # Find a vector perpendicular to the leading edge vector (LE_vector)
    arbitrary_vector = np.array([1, 0, 0]) if np.abs(LE_vector[0]) < 0.9 else np.array([0, 1, 0])
    perp_vector = np.cross(LE_vector, arbitrary_vector)
    perp_vector /= np.linalg.norm(perp_vector)  # Normalize the perpendicular vector

    # Compute dot product of points with the perpendicular vector
    directions = points - adjusted_center
    dot_products = np.dot(directions, perp_vector)

     # Use a tolerance to avoid missing points near the boundary (where dot_product ≈ 0)
    left_side = points[dot_products <= 0]
    right_side = points[dot_products >= 0]

    # Create point cloud for the left side and color it red
    left_side_cloud = o3d.geometry.PointCloud()
    left_side_cloud.points = o3d.utility.Vector3dVector(left_side)
    left_side_cloud.paint_uniform_color([1, 0, 0])  # Red for left side
    vis_elements.append(left_side_cloud)

    # Create point cloud for the right side and color it blue
    right_side_cloud = o3d.geometry.PointCloud()
    right_side_cloud.points = o3d.utility.Vector3dVector(right_side)
    right_side_cloud.paint_uniform_color([0, 0, 1])  # Blue for right side
    vis_elements.append(right_side_cloud)

    return left_side, right_side, vis_elements

def compute_area_between_points(recontoured_side, original_side, LE_vector, adjusted_center):
    """Compute the area between corresponding points on the green and red side."""
    center_projection = np.dot(adjusted_center, LE_vector)

    # Project points onto the leading edge vector
    recontour_point_proj = np.dot(recontoured_side, LE_vector)
    original_point_proj = np.dot(original_side, LE_vector)
    
    # Filter points above the center (where projection > center projection)
    recontoured_above = recontoured_side[recontour_point_proj > center_projection]
    original_above = original_side[original_point_proj > center_projection]

    # Ensure both sides have the same number of points
    min_length = min(len(recontoured_above), len(original_above))
    recontoured_above = recontoured_above[:min_length]  # Trim or handle boundary cases
    original_above = original_above[:min_length]  # Trim or handle boundary cases

    # Project the filtered points onto the leading edge vector again
    recontour_point_proj = np.dot(recontoured_above, LE_vector)
    original_point_proj = np.dot(original_above, LE_vector)

     # Calculate the "distances" as the difference between corresponding points along the LE vector
    distances = np.abs(recontour_point_proj - original_point_proj)
    
    # Calculate the "widths" (projection differences) along the perpendicular axis
    perp_vector = np.cross(LE_vector, np.array([1, 0, 0]) if np.abs(LE_vector[0]) < 0.9 else np.array([0, 1, 0]))
    perp_vector /= np.linalg.norm(perp_vector)  # Normalize the perpendicular vector

    # Project points onto the perpendicular vector
    recontour_point_perp_proj = np.dot(recontoured_above, perp_vector)
    original_point_perp_proj = np.dot(original_above, perp_vector)

    # Calculate the differences along the perpendicular axis (these are the widths for trapezoidal integration)
    projection_diffs = np.abs(recontour_point_perp_proj - original_point_perp_proj)

    # Apply the trapezoidal rule: sum of (distance_i + distance_(i+1)) * (width between the points) / 2
    area = np.sum((distances[:-1] + distances[1:]) * projection_diffs[:-1] / 2.0)

    return area

def calculate_area_between_points(recontoured_points, original_points, LE_vector, adjusted_center, vis_elements):
    """Calculate the total area between green and red points by separating into left and right sides."""
    
    # Separate the green and red points into left and right sides
    recontoured_left, recontoured_right, vis_elements = separate_sides(recontoured_points, LE_vector, adjusted_center, vis_elements)
    original_left, original_right, vis_elements = separate_sides(original_points, LE_vector, adjusted_center, vis_elements)
    
    # Compute the area for the left side
    area_left = compute_area_between_points(recontoured_left, original_left, LE_vector, adjusted_center)
    
    # Compute the area for the right side
    area_right = compute_area_between_points(recontoured_right, original_right, LE_vector, adjusted_center)
    
    return {
        'sub_section_idx_1': area_left,  # Left side (sub_section idx 1)
        'sub_section_idx_2': area_right  # Right side (sub_section idx 2)
    }, vis_elements


def calculate_lost_volumes(area_removals, constant_width):
    """
    Multiply the areas by the constant width and print the lost volumes for each section.

    Parameters:
    - area_removals: List of dictionaries containing section index and areas for left and right sides.
    - constant_width: The constant width to multiply with the area values.
    """
    lost_volumes = []

    for entry in area_removals:
        section_index = entry['section_index']
        areas = entry['areas']

        # Calculate the volumes for left (sub_section_idx_1) and right (sub_section_idx_2)
        volume_left = areas['sub_section_idx_1'] * constant_width
        volume_right = areas['sub_section_idx_2'] * constant_width

        # Append the volumes along with section and sub-section indices
        lost_volumes.append({
            'section_idx': section_index,
            'sub_section_idx': 1,  # sub_section_idx_1 corresponds to 1
            'lost_volume': volume_left
        })
        lost_volumes.append({
            'section_idx': section_index,
            'sub_section_idx': 2,  # sub_section_idx_2 corresponds to 2
            'lost_volume': volume_right
        })

    for entry in lost_volumes:
        section_idx = entry['section_idx']
        sub_section_idx = entry['sub_section_idx']
        volume = entry['lost_volume']
        
        print(f"Section {section_idx}, Sub-section {sub_section_idx}: Volume = {volume}")


    return lost_volumes


def main():
    # Load mesh to mesh processor           comment one out depending on data type
    mstore = MeshProcessor()
    mvis = MeshVisualizer()

    mstore.load_mesh(1)

    if mstore.mesh1_pcl == None:
        mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=60000)


    scale_factor = 1.0
    thresholds = {
        "project_tolerance": 0.0002 * scale_factor,
    }

    theta_x = np.deg2rad(-15)  # degrees around X-axis
    theta_y = np.deg2rad(31)  # degrees around Y-axis
    theta_z = np.deg2rad(-12)  # degrees around Z-axis

    # Rotate the point cloud
    mstore.mesh1_pcl = mstore.rotate_point_cloud(mstore.mesh1_pcl, theta_x, theta_y, theta_z)
    
    #Create LE sections
    LE_sections_mesh1, LE_sections_mesh1_length = slice_point_cloud_along_axis(mstore.mesh1_pcl, flow_axis = 'y', num_sections = 10, threshold=thresholds["project_tolerance"])

    mvis.visualize_pcl_overlay(mstore.mesh1_pcl, LE_sections_mesh1)

    mstore.mesh1_LE_points = detect_leading_edge_by_maxima(LE_sections_mesh1, leading_edge_axis='x')

    mvis.visualize_sections_with_leading_edges(LE_sections_mesh1, mstore.mesh1_LE_points)

    recontoured_LE_sections, area_removals = recontour_LE_sections(LE_sections_mesh1, mstore.mesh1_LE_points, target_parabolic_parameter=3)

    mstore.lost_volumes = calculate_lost_volumes(area_removals, LE_sections_mesh1_length)


    # Create model
    # TO DO: Store model somewhere so we dont need to train it everytime we run this code
    create_grind_model(mstore)

    # Predict RPM and Force
    # TO DO: think about how should we do this. What are the inputs (lost volume, curvature, or more?)
    #                                           What are the outputs (RPM, Force, Feed rate, or fix some variables?)

    # Fixed feed rate
    feed_rate = 20
    predict_grind_param(mstore, feed_rate)


if __name__ == "__main__":
    main()