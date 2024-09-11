import numpy as np
import open3d as o3d
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl


from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl

# Preprocess Point Cloud 
def fit_spline_to_leading_edge(leading_edge_points, smoothing_factor=1e-3):
    
    leading_edge_points = np.asarray(leading_edge_points)

    # Remove duplicate points (if any)
    leading_edge_points = np.unique(leading_edge_points, axis=0)
    print(f"Number of leading edge points: {leading_edge_points.shape[0]}")

    # Fit the spline to the leading edge points
    try:
        # Introduce a small smoothing factor to handle noisy or closely packed points
        tck, u = splprep([leading_edge_points[:, 0], leading_edge_points[:, 1], leading_edge_points[:, 2]], s=smoothing_factor)
        return tck, u
    except Exception as e:
        print(f"Error in fitting spline: {e}")
        raise


def visualize_leading_edge_and_spline(pcd, leading_edge_points, spline_points):
    """
    Visualize the original point cloud, detected leading edge points, and the fitted spline.
    
    Parameters:
    - pcd: Original point cloud (open3d.geometry.PointCloud).
    - leading_edge_points: Detected leading edge points as a numpy array.
    - spline_points: Sampled points along the fitted spline.
    """
    # Convert leading edge points and spline points to open3d point clouds
    leading_edge_pcd = o3d.geometry.PointCloud()
    leading_edge_pcd.points = o3d.utility.Vector3dVector(leading_edge_points)
    leading_edge_pcd.paint_uniform_color([1, 0, 0])  # Red color for leading edge points

    spline_pcd = o3d.geometry.PointCloud()
    spline_pcd.points = o3d.utility.Vector3dVector(spline_points)
    spline_pcd.paint_uniform_color([0, 1, 0])  # Green color for spline points

    # Visualize the original point cloud, leading edge points, and the spline
    
    o3d.visualization.draw_geometries([pcd, leading_edge_pcd, spline_pcd])
    o3d.visualization.draw_geometries([leading_edge_pcd])



def estimate_curvature(pcd, k_neighbors=30):
    """
    Estimate curvature for each point in the point cloud.
    
    Parameters:
    - pcd: Input point cloud (open3d.geometry.PointCloud)
    - k_neighbors: Number of neighbors to use for curvature estimation.
    
    Returns:
    - curvatures: Numpy array of curvature values for each point.
    """
    # Compute normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    
    # Initialize KDTree for fast neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    points = np.asarray(pcd.points)
    curvatures = []

    # Compute curvature for each point
    for i in range(len(points)):
        # Search for k nearest neighbors
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)

        # Use the neighbors to calculate the covariance matrix
        neighbors = points[idx, :]
        covariance_matrix = np.cov(neighbors.T)

        # Perform eigenvalue decomposition to get principal curvatures
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)

        # Curvature is proportional to the smallest eigenvalue
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)


def detect_leading_edge_by_curvature(pcd, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=20, min_distance=40):
    """
    Detect leading edge points based on curvature estimation and further refine them by selecting 
    the point with the highest curvature in the vicinity.

    Parameters:
    - pcd: Input point cloud (open3d.geometry.PointCloud)
    - curvature_threshold: Tuple specifying the (lower, upper) bound for curvature.
    - k_neighbors: Number of neighbors for curvature estimation.
    - vicinity_radius: Radius around each point within which to look for the point with the highest curvature.

    Returns:
    - leading_edge_points: Numpy array of refined leading edge points.
    """
    # Estimate curvature for each point in the point cloud
    curvatures = estimate_curvature(pcd, k_neighbors=k_neighbors)

    # Filter points that fall within the curvature range
    lower_bound, upper_bound = curvature_threshold
    filtered_indices = np.where((curvatures >= lower_bound) & (curvatures <= upper_bound))[0]

    # KDTree for fast neighborhood search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize list to hold refined leading edge points
    refined_leading_edge_points = []

    # Process each filtered point
    for idx in filtered_indices:
        point = pcd.points[idx]

        # Find the neighboring points within the vicinity_radius
        [_, idx_neigh, _] = kdtree.search_radius_vector_3d(point, vicinity_radius)

        # Find the index of the point with the highest curvature within the neighborhood
        if len(idx_neigh) > 0:
            highest_curvature_idx = idx_neigh[np.argmax(curvatures[idx_neigh])]
            refined_leading_edge_points.append(pcd.points[highest_curvature_idx])
    
    # Further remove points that are too close to each other
    filtered_leading_edge_points = []

    for point in refined_leading_edge_points:
        if len(filtered_leading_edge_points) == 0:
            filtered_leading_edge_points.append(point)
        else:
            # Check the distance between the current point and all previously selected points
            distances = np.linalg.norm(np.array(filtered_leading_edge_points) - point, axis=1)
            if np.all(distances >= min_distance):  # Only add if it's far enough from existing points
                filtered_leading_edge_points.append(point)

    return np.array(filtered_leading_edge_points)


def visualize_curvature_based_leading_edge(pcd, leading_edge_points):
    """
    Visualize the original point cloud and the detected leading edge points based on curvature.
    
    Parameters:
    - pcd: Original point cloud (open3d.geometry.PointCloud)
    - leading_edge_points: Detected leading edge points as a numpy array.
    """
    # Create point cloud from leading edge points
    leading_edge_pcd = o3d.geometry.PointCloud()
    leading_edge_pcd.points = o3d.utility.Vector3dVector(leading_edge_points)
    leading_edge_pcd.paint_uniform_color([1, 0, 0])  # Red color for leading edge points

    # Visualize original point cloud and leading edge points
    o3d.visualization.draw_geometries([pcd, leading_edge_pcd])

def sample_spline(tck, num_points=100):
    """
    Sample points along the fitted spline for visualization.
    
    Parameters:
    - tck: Spline parameters.
    - num_points: Number of points to sample along the spline.
    
    Returns:
    - sampled_points: Sampled points along the spline as a numpy array.
    """
    u_fine = np.linspace(0, 1, num_points)
    sampled_points = splev(u_fine, tck)
    return np.vstack(sampled_points).T

# Main
# Load mesh to mesh processor
mstore = MeshProcessor()
mstore.mesh1 = mstore.load_mesh(1)

print("Mesh Loaded")
# Sample points
mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=40000)

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
leading_edge_points = detect_leading_edge_by_curvature(mstore.mesh1_pcl)
tck, u = fit_spline_to_leading_edge(leading_edge_points)

# Sample points along the spline for visualization
spline_points = sample_spline(tck, num_points=1000)

# Visualize the leading edge points and the spline
visualize_leading_edge_and_spline(mstore.mesh1_pcl, leading_edge_points, spline_points)





def point_to_plane_distance(points, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize the plane normal
    distances = np.dot(points - plane_point, plane_normal)
    return np.abs(distances)

def extract_points_on_plane(point_cloud, plane_point, plane_normal, threshold=0.1):

    distances = point_to_plane_distance(np.asarray(point_cloud.points), plane_point, plane_normal)
    mask = distances < threshold
    points_on_plane = np.asarray(point_cloud.points)[mask]
    
    # Create a PointCloud for the points on the plane
    points_on_plane_cloud = o3d.geometry.PointCloud()
    points_on_plane_cloud.points = o3d.utility.Vector3dVector(points_on_plane)
    
    return points_on_plane_cloud

def slice_point_cloud_with_visualization(point_cloud, leading_edge_points, num_sections=10, threshold=0.1):
    """
    Slice a point cloud into sections using leading edge points and visualize the point cloud, planes, 
    and points lying on the planes.
    Args:
        point_cloud: The input point cloud.
        leading_edge_points: A list of 3D points representing the leading edge along the blade.
        num_sections: Number of sections to divide the blade into.
        threshold: Distance threshold for points to be considered on the plane.
    """
    vis_element =[]
    sections = []

    for i in range(len(leading_edge_points) - 1):
        start_point = leading_edge_points[i]
        end_point = leading_edge_points[i + 1]

        # Interpolate along the line segment between leading edge points
        for j in range(num_sections):
            t = j / num_sections
            section_point = (1 - t) * start_point + t * end_point
            
            # Define the section plane normal as the local flow axis
            flow_axis = end_point - start_point
            flow_axis /= np.linalg.norm(flow_axis)

            # Create the plane for visualization
            plane = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=0.01)
            plane.translate(section_point - np.array([50, 50, 0]))  # Adjust position
            plane.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.cross([0, 0, 1], flow_axis)))
            #vis_element.append(plane)

            # Extract points from the point cloud that lie on this plane
            points_on_plane = extract_points_on_plane(point_cloud, section_point, flow_axis, threshold)

            # Color the points on the plane for visualization
            if len(points_on_plane.points) > 0:
                points_on_plane.paint_uniform_color([0, 0, 0])  # Color the points red
                vis_element.append(points_on_plane)

                # Convert the point cloud to NumPy array and store in sections
                section_points_array = np.asarray(points_on_plane.points)
                sections.append(section_points_array)

    # Visualize everything (point cloud, planes, and points on planes)
    o3d.visualization.draw_geometries(vis_element, window_name="Point Cloud Sections with Points on Planes", width=800, height=600)

    return sections


LE_sections = slice_point_cloud_with_visualization(mstore.mesh1_pcl, leading_edge_points, num_sections=3, threshold=0.5)


def recontour_LE_sections(LE_sections, target_radius=2):

    recontoured_sections = []

    for section_id, section_points in enumerate(LE_sections):
        center = np.mean(section_points, axis=0)
        recontoured_section = []
        for point in section_points:
            direction = point - center
            direction /= np.linalg.norm(direction)
            new_point = center + direction * target_radius
            recontoured_section.append(new_point)

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

recontoured_LE_sections = recontour_LE_sections(LE_sections, target_radius=10)

from scipy.spatial import cKDTree

def match_points_between_sections(section_1, section_2):
    """
    Match points between two sections using nearest-neighbor search.
    Args:
        section_1: Nx3 NumPy array representing points of the first section.
        section_2: Mx3 NumPy array representing points of the second section.
    Returns:
        matched_pairs: List of tuples where each tuple contains indices of matched points (from section_1, section_2).
    """
    # Create KD-tree for the second section
    tree = cKDTree(section_2)
    
    # Query the nearest neighbors for each point in section_1
    distances, indices = tree.query(section_1)
    
    # Return the matched pairs (index in section_1, index in section_2)
    matched_pairs = [(i, indices[i]) for i in range(len(section_1))]
    
    return matched_pairs

def create_surface_mesh_from_sections(sections):
    """
    Create a surface mesh from section lines (cross-sections) by matching points using nearest-neighbor search.
    Args:
        sections: List of NumPy arrays representing cross-sections of the turbine blade.
    Returns:
        surface_mesh: Open3D TriangleMesh object representing the surface mesh.
    """
    vertices = []
    triangles = []

    num_sections = len(sections)

    # Loop through each pair of consecutive sections
    for i in range(num_sections - 1):
        section_1 = sections[i]
        section_2 = sections[i + 1]

        # Match points between section_1 and section_2
        matched_pairs = match_points_between_sections(section_1, section_2)

        for j, (p1, p2) in enumerate(matched_pairs):
            # Get the next point in the section to form triangles
            next_p1 = (j + 1) % len(section_1)  # Wrap around for last point in section
            next_p2 = (matched_pairs[next_p1][1] if next_p1 < len(matched_pairs) else 0)  # Wrap around in section 2

            # Vertices for the triangles
            v0 = section_1[p1]
            v1 = section_1[next_p1]
            v2 = section_2[p2]
            v3 = section_2[next_p2]

            # Add the vertices to the vertex list
            idx0 = len(vertices)
            vertices.append(v0)
            vertices.append(v1)
            vertices.append(v2)
            vertices.append(v3)

            # Add two triangles connecting the points from consecutive sections
            triangles.append([idx0, idx0 + 1, idx0 + 2])  # Triangle 1
            triangles.append([idx0 + 1, idx0 + 3, idx0 + 2])  # Triangle 2

    # Convert to Open3D mesh
    vertices = np.array(vertices)
    triangles = np.array(triangles)

    surface_mesh = o3d.geometry.TriangleMesh()
    surface_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    surface_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    surface_mesh.compute_vertex_normals()  # Compute normals for better visualization

    return surface_mesh

# Example usage (assuming sections is a list of NumPy arrays representing cross-sections)

turbine_surface = create_surface_mesh_from_sections(recontoured_LE_sections)

o3d.visualization.draw_geometries([turbine_surface], window_name="Turbines", width=800, height=600)

