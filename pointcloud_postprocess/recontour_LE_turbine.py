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

def filter_and_project_sections(LE_sections_mesh1, LE_sections_mesh2, threshold=50, point_threshold=5):
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

    #o3d.visualization.draw_geometries(vis_element)
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


def shift_mesh2_and_leading_edge(mesh1_section, mesh2_section, leading_edge_point_mesh2, LE_vector):
    """Shift mesh2 section and its leading edge point into the envelope of mesh1."""
    LE_vector_norm = LE_vector / np.linalg.norm(LE_vector)

    # Create KD trees for nearest neighbor search
    tree_mesh1 = cKDTree(mesh1_section)
    tree_mesh2 = cKDTree(mesh2_section)
    
    # Find nearest neighbors between mesh1_section and mesh2_section
    distances1, _ = tree_mesh1.query(mesh2_section)
    distances2, _ = tree_mesh2.query(mesh1_section)
    
    # Maximum distance between nearest neighbors
    max_distance = max(np.max(distances1), np.max(distances2))
    
    # Shift mesh2 along the LE_vector by the maximum distance
    shifted_mesh2_section = mesh2_section - (max_distance * LE_vector_norm)
    
    # Shift the leading edge point of mesh2 along the same LE_vector
    shifted_leading_edge_point_mesh2 = leading_edge_point_mesh2 - (max_distance * LE_vector_norm)

    return shifted_mesh2_section, shifted_leading_edge_point_mesh2


def recontour_with_shift_and_projection(mesh1_sections, mesh2_sections, leading_edge_points_mesh2, tolerance=1e-3):
    """Recontour mesh1 sections to match mesh2 sections, ensuring the contour fits within the original mesh1 boundary section by section."""
    
    recontoured_sections = []
    vis_elements = []
    adjusted_centers = []

    for mesh1_section, mesh2_section in zip(mesh1_sections, mesh2_sections):
        # 1. Determine the leading edge vector and center for mesh2 section
        leading_edge_point_mesh2 = find_closest_leading_edge_point(mesh2_section, leading_edge_points_mesh2)
        initial_center_mesh2 = np.mean(mesh2_section, axis=0)

        # Use your method to adjust center and LE vector for symmetry
        adjusted_center_mesh2, LE_vector, vis_elements = adjust_center_and_le_for_symmetry(
            mesh2_section, leading_edge_point_mesh2, initial_center_mesh2, vis_elements, tolerance
        )
        
        adjusted_centers.append(adjusted_center_mesh2)

        # 2. Shift mesh2 section into the envelope of mesh1 section
        shifted_mesh2_section, shifted_leading_edge_point_mesh2 = shift_mesh2_and_leading_edge(
            mesh1_section, mesh2_section, leading_edge_point_mesh2, LE_vector
        )

        # 3. Recontour mesh1 section to fit within the shifted mesh2 section envelope
        recontoured_section = recontour_section_to_fit_within_bounds(
            mesh1_section, shifted_mesh2_section, adjusted_center_mesh2, LE_vector, shifted_leading_edge_point_mesh2, tolerance
        )

        
        recontoured_sections.append(recontoured_section)


        
        mesh1_point_cloud = o3d.geometry.PointCloud()
        mesh1_point_cloud.points = o3d.utility.Vector3dVector(mesh1_section)
        vis_elements.append(mesh1_point_cloud)
        
        '''
        mesh2_point_cloud = o3d.geometry.PointCloud()
        mesh2_point_cloud.points = o3d.utility.Vector3dVector(mesh2_section)
        mesh2_point_cloud.paint_uniform_color([1, 0, 0]) 
        vis_elements.append(mesh2_point_cloud)
        '''
        '''
        shifted_mesh2_pcl = o3d.geometry.PointCloud()
        shifted_mesh2_pcl.points = o3d.utility.Vector3dVector(shifted_mesh2_section)
        shifted_mesh2_pcl.paint_uniform_color([0, 1, 0]) 
        vis_elements.append(shifted_mesh2_pcl)
        '''

    
    point_cloud = o3d.geometry.PointCloud()
    all_points = np.vstack(recontoured_sections)  # Stack all recontoured sections into a single array
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.paint_uniform_color([0, 0, 1]) 
    vis_elements.append(point_cloud)
    
    o3d.visualization.draw_geometries(vis_elements, window_name="PCL Overlay", width=800, height=600)

    return recontoured_sections, point_cloud


def recontour_section_to_fit_within_bounds(mesh1_section, shifted_mesh2_section, adjusted_center, LE_vector, leading_edge_point, tolerance=1e-3):
    """Recontour mesh1 section to fit within the bounds of the shifted mesh2 section."""

    #TO DO: Improve fit within bound to be smooth and has camber line aligned with original mesh 2 pcl

    tree = cKDTree(shifted_mesh2_section)  # Nearest-neighbor search in shifted mesh2 section
    new_mesh1_section = []
    max_perp_distance_left = 0
    max_perp_distance_right = 0
    min_perp_distance_right = 0
    min_perp_distance_left = 0
    min_LE_vector = 0

    # First, calculate the maximum perpendicular distance from the adjusted center for points below the center
    for point in mesh1_section:
        dist, idx = tree.query(point)
        target_point = shifted_mesh2_section[idx]

        direction = point - adjusted_center
        direction_target = target_point - adjusted_center
        projection_on_LE_vector = np.dot(direction, LE_vector)
        projection_on_LE_vector_target = np.dot(direction_target, LE_vector)

        if projection_on_LE_vector <= 0:  # Points below or at the center
            # Project the point onto the plane perpendicular to the LE_vector
            perpendicular_vector = direction - projection_on_LE_vector * LE_vector
            perpendicular_distance = np.linalg.norm(perpendicular_vector)
            perpendicular_vector_target = direction_target - projection_on_LE_vector_target * LE_vector
            perpendicular_distance_target = np.linalg.norm(perpendicular_vector_target)
            min_LE_vector = min(min_LE_vector, projection_on_LE_vector)
            # Track the maximum perpendicular distance
            if perpendicular_distance > 0:
                    max_perp_distance_right = max(max_perp_distance_right, perpendicular_distance)
            elif perpendicular_distance < 0:
                    perpendicular_distance = -perpendicular_distance
                    max_perp_distance_left = max(max_perp_distance_left, perpendicular_distance)

            if perpendicular_distance_target > 0:
                    min_perp_distance_right = min(min_perp_distance_right, perpendicular_distance_target)
            elif perpendicular_distance_target < 0:
                    perpendicular_distance_target = -perpendicular_distance_target
                    min_perp_distance_left = min(min_perp_distance_left, perpendicular_distance_target)


    for point in mesh1_section:
        # Find the closest point in shifted mesh2 section
        dist, idx = tree.query(point)
        target_point = shifted_mesh2_section[idx]

        # Compute the adjustment vector
        adjustment_vector = target_point - point
        new_point = point + adjustment_vector

        # Calculate the original distance from the leading edge point
        original_distance = np.linalg.norm(point - leading_edge_point)

        
        #Slope gap between target and mesh1_section
        direction = point - adjusted_center
        projection_on_LE_vector = np.dot(direction, LE_vector)
        '''
        if projection_on_LE_vector <= 0:  # Points below or at the center
            perpendicular_vector = direction - projection_on_LE_vector * LE_vector
            perpendicular_distance = np.linalg.norm(perpendicular_vector)
            perpendicular_direction = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        

            if np.dot(perpendicular_vector, perpendicular_vector) > 0:
                perp_adjust_right = (projection_on_LE_vector / min_LE_vector) * (max_perp_distance_right - min_perp_distance_right)
                new_point = new_point + perp_adjust_right * perpendicular_direction
            else:
                perp_adjust_left = (projection_on_LE_vector / min_LE_vector) * (max_perp_distance_left - min_perp_distance_left)
                new_point = new_point + perp_adjust_left * perpendicular_direction
        '''
        
        # Delete new points that exceed the original distance from the leading edge
        new_distance = np.linalg.norm(new_point - leading_edge_point)
        if new_distance <= original_distance:
            new_mesh1_section.append(new_point)
        elif projection_on_LE_vector <= 0 and projection_on_LE_vector > min_LE_vector:
            new_mesh1_section.append(point)



    return np.array(new_mesh1_section)

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

    #curvature_array = mstore.estimate_curvature(mstore.mesh1_pcl)
    mstore.mesh1_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh1_pcl)
    mstore.mesh2_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh2_pcl)
    #tck, u = fit_spline_to_leading_edge(mstore.mesh1_LE_points)

    # Sample points along the spline for visualization
    #spline_points = sample_spline(tck, num_points=1000)

    # Visualize the leading edge points and the spline
    mvis.visualize_pcl_overlay(mstore.mesh1_pcl, mstore.mesh1_LE_points)

    #mesh1 is damaged blade, mesh2 is undamaged / ideal blade

    LE_sections_mesh1 = slice_point_cloud_with_visualization(mstore.mesh1_pcl, mstore.mesh1_LE_points, num_sections=2, threshold=0.8)
    LE_sections_mesh2 = slice_point_cloud_with_visualization(mstore.mesh2_pcl, mstore.mesh2_LE_points, num_sections=2, threshold=0.8)
    mvis.visualize_pcl_overlay(LE_sections_mesh1, LE_sections_mesh2)

    LE_sections_mesh1, LE_sections_mesh2 = filter_and_project_sections(LE_sections_mesh1, LE_sections_mesh2, threshold=60, point_threshold=15)
    mvis.visualize_pcl_overlay(LE_sections_mesh1, LE_sections_mesh2)



    recontoured_sections, recontoured_sections_pcl = recontour_with_shift_and_projection(
        LE_sections_mesh1, LE_sections_mesh2, mstore.mesh2_LE_points, tolerance=1e-3
    )
    print("final recontoured compared with ideal shape")


    turbine_surface = mstore.create_mesh_from_pcl(recontoured_sections_pcl)

    o3d.visualization.draw_geometries([turbine_surface], window_name="Turbines", width=800, height=600)


if __name__ == "__main__":
    main()