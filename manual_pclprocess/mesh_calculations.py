# mesh_calculations.py

import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.spatial import cKDTree, Delaunay
import time
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_mesh(mesh_number):
    path = filedialog.askopenfilename(title=f"Select the mesh file for Mesh {mesh_number}",
                                      filetypes=[("PLY files", "*.ply"), ("All Files", "*.*")])
    if path:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            return mesh
        else:
            mesh = o3d.io.read_point_cloud(path)
            if len(mesh.points) > 0:
                return mesh
            else:
                messagebox.showwarning("Warning", f"Mesh {mesh_number} contains no data.")
    else:
        messagebox.showwarning("Warning", f"No file selected for Mesh {mesh_number}")
    return None


def segment_leading_edge_by_y_distance(input_data, num_segments=3, mid_ratio=0.4, use_bounds=None):
    # Check if the input is a triangle mesh or point cloud
    if isinstance(input_data, o3d.geometry.TriangleMesh):
        points = np.asarray(input_data.vertices)
        is_mesh = True
    elif isinstance(input_data, o3d.geometry.PointCloud):
        points = np.asarray(input_data.points)
        is_mesh = False
    else:
        raise TypeError("Input must be either an Open3D TriangleMesh or PointCloud.")

    # Sort points based on the Y-coordinate (second index)
    sorted_indices = np.argsort(points[:, 1])  # Sort by Y-coordinate
    sorted_points = points[sorted_indices]

    if use_bounds is None:
        # Compute the Y-axis bounds for segmentation based on the mid_ratio
        min_y = sorted_points[0, 1]
        max_y = sorted_points[-1, 1]
        total_y_range = max_y - min_y

        # Allocate a larger Y-range for the middle section
        middle_y_range = total_y_range * mid_ratio
        side_y_range = (total_y_range - middle_y_range) / 2

        # Define boundaries for each segment
        bounds = [
            min_y,                        # Start of the first segment
            min_y + side_y_range,          # End of the first segment and start of the middle segment
            min_y + side_y_range + middle_y_range,  # End of the middle segment and start of the third segment
            max_y                         # End of the third segment
        ]
    else:
        # Use predefined Y-boundaries
        bounds = use_bounds


    sub_sections = []

    # Divide the points into 3 segments based on the adjusted Y-boundaries
    for i in range(3):
        lower_bound = bounds[i]
        upper_bound = bounds[i + 1]

        # Create a mask for points that fall within the current Y-distance range
        mask = (sorted_points[:, 1] >= lower_bound) & (sorted_points[:, 1] < upper_bound)

        # Extract the points/vertices for this segment
        segment_indices = sorted_indices[mask]
        
        if is_mesh:
            # If it's a triangle mesh, create a new sub-mesh based on the selected vertices
            sub_mesh = input_data.select_by_index(segment_indices, vertex_only=True)
            sub_sections.append(sub_mesh)
        else:
            # If it's a point cloud, create a new sub-point cloud
            sub_pcd = input_data.select_by_index(segment_indices)
            sub_sections.append(sub_pcd)

    return sub_sections, bounds


def joggle_points(pcd, scale=1e-6):
    points = np.asarray(pcd.points)
    jitter = np.random.normal(scale=scale, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + jitter)


def calculate_lost_volume_from_changedpcl(mesh_missing, fixed_thickness):
    reference_area = mesh_missing.get_surface_area()
    volume_lost = fixed_thickness * reference_area

    return volume_lost

def create_mesh_from_point_cloud(pcd):
    points = np.asarray(pcd.points)
    jitter = np.random.normal(scale=1e-6, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + jitter) #add jitter if points coincide
    print('estimating normals')
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    print('meshing')
    
    # Alpha Shape
    # Iteratively adjust radii until a suitable mesh is created
    iteration = 0
    max_iterations = 10
    step = 1.2
    alpha = 0.002  # Adjust this parameter for alpha shape detail
    direction = "multiply"  # Start by multiplying alpha
    previous_surface_area = 0
    '''
    while iteration < max_iterations:
        try:
            # Try Alpha Shape for mesh creation with the current alpha value
            #print(f'Attempting mesh creation with alpha: {alpha:.2g}')
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

            # Compute the surface area of the mesh
            current_surface_area = mesh.get_surface_area()
            #print(f"Current surface area: {current_surface_area}")

            # Check if the surface area is below the threshold or has decreased
            if current_surface_area > previous_surface_area:
                # Surface area has increased, update previous_surface_area
                previous_surface_area = current_surface_area

            else:
                # If surface area decreased or did not improve, switch the direction
                if direction == "multiply":
                    direction = "divide"
                else:
                    direction = "multiply"

            # Update alpha based on the current direction
            if direction == "multiply":
                alpha *= step
            else:
                alpha /= step


        except Exception as e:
            print(f"Alpha shape failed on iteration {iteration} with error: {e}")

        # Increment iteration counter
        iteration += 1
    '''
    
    #uncomment this part if dont want iterative mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    current_surface_area = mesh.get_surface_area()
    print(f"Mesh created successfully with surface area {current_surface_area} and alpha {alpha:.2g}")
    
    '''
    #ball pivoting
    current_factor = 1.0
    #distances = pcd.compute_nearest_neighbor_distance()
    #avg_dist = np.mean(distances)
    initial_radii = [0.0001, 0.00025, 0.0005, 0.0075, 0.001, 0.002] 
    radii = [r * current_factor for r in initial_radii]
    r = o3d.utility.DoubleVector(radii)

    while iteration < max_iterations:
        # Dynamically scale radii based on the current factor
        radii = [r * current_factor for r in initial_radii]  # Scale all radii by the current factor
        print(f'Attempting mesh creation with radii: {[f"{r:.2g}" for r in radii]}')
        
        # Convert radii to Open3D-compatible DoubleVector format
        r = o3d.utility.DoubleVector(radii)
        
        try:
            # Try ball pivoting for mesh creation
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, r)
            current_vertices = len(mesh.vertices)
            print(f"Mesh created with {current_vertices} vertices.")
            
            # Check if the mesh has enough triangles/vertices to be considered "good"
            if current_vertices > vertice_quality_threshold:
                print(f"Mesh created successfully with {current_vertices} vertices.")
                return mesh  # Return the successfully created mesh

            # Check if the number of vertices decreased from the previous iteration
            if iteration > 0 and current_vertices < previous_vertices:
                # If the number of vertices decreases, switch direction (multiply -> divide or divide -> multiply)
                if direction == "multiply":
                    direction = "divide"
                else:
                    direction = "multiply"

            # Update current factor based on the current direction
            if direction == "multiply":
                current_factor *= step
            else:
                current_factor /= step

            # Store the number of vertices for the next iteration comparison
            previous_vertices = current_vertices

        except Exception as e:
            print(f"Ball pivoting failed on iteration {iteration} with error: {e}")

        # Increment iteration counter
        iteration += 1  
    '''

    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, r)

    return mesh

def fit_plane_to_pcd_pca(pcd):
        """Fit a plane to a cluster of points using PCA."""
        points = np.asarray(pcd.points)

        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(points)

        # Get the normal to the plane (third principal component)
        plane_normal = pca.components_[2]  # The normal to the plane (least variance direction)
        pca_basis = pca.components_  # Shape (3, 3)

        # The mean of the data gives the centroid
        centroid = pca.mean_  # Shape (3,)

        return pca_basis, centroid

def transform_to_local_pca_coordinates(pcd, pca_basis, centroid):
    points = np.asarray(pcd.points)
    centered_points = points - centroid
    local_points = centered_points @ pca_basis.T
    return local_points

def project_points_onto_plane(points, plane_normal, plane_point):
    """Project points onto the plane defined by the normal and a point."""
    vectors = points - plane_point  # Vector from point to plane_point

    distances = np.dot(vectors, plane_normal)  # Project onto the normal

    projected_points = points - np.outer(distances, plane_normal)  # Subtract projection along the normal

    return projected_points

def filter_project_points_by_plane(point_cloud, distance_threshold=0.001):
    # Fit a plane to the point cloud using RANSAC
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    [a, b, c, d] = plane_model
    
    # Select points that are close to the plane (within the threshold)
    inlier_cloud = point_cloud.select_by_index(inliers)
 
    pca_basis, plane_centroid = fit_plane_to_pcd_pca(inlier_cloud)
    points = np.asarray(inlier_cloud.points)

    projected_points = project_points_onto_plane(points, pca_basis[2], plane_centroid)

    # Create a new point cloud with the projected points
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    projected_pcd.paint_uniform_color([1, 0, 0])  # Color projected points in red

    # Color the inlier points (on the original RANSAC plane) in green
    inlier_cloud.paint_uniform_color([0, 1, 0])  # Color inlier points in green

    # Visualize both the original point cloud, inliers, and projected points
    #o3d.visualization.draw_geometries([projected_pcd])

    return projected_pcd, pca_basis, plane_centroid

def create_mesh_from_clusters(pcd, eps=0.005, min_points=30, remove_outliers=True):
    # Step 1: Segment point cloud into clusters using DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # Number of clusters (label -1 indicates noise)
    num_clusters = labels.max() + 1
    print(f"PointCloud has {num_clusters} clusters")

    # Step 2: Iterate over each cluster and create a mesh
    meshes = []
    for cluster_idx in range(num_clusters):
        # Select points belonging to the current cluster
        cluster_pcd = pcd.select_by_index(np.where(labels == cluster_idx)[0])

        # Step 3: Apply mesh creation process for each cluster
        if len(cluster_pcd.points) > 3:  # Ensure there are enough points
            # Fit a plane to the cluster using PCA
            plane_normal, plane_centroid = fit_plane_to_cluster_pca(cluster_pcd)

            # Project the points onto the plane
            points = np.asarray(cluster_pcd.points)
            projected_points = project_points_onto_plane(points, plane_normal, plane_centroid)

            # Create a new point cloud for the projected points
            cluster_pcd.points = o3d.utility.Vector3dVector(projected_points)


            mesh = create_mesh_from_point_cloud(cluster_pcd)
            meshes.append(mesh)

    # Step 4: Merge all the cluster meshes into one
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        combined_mesh += mesh

    return combined_mesh

'''
def filter_changedpointson_mesh(mesh_before, mesh_after, threshold=0.001, neighbor_threshold=5):
    # Convert points from Open3D mesh to numpy arrays
    points_before = np.asarray(mesh_before.points)
    points_after = np.asarray(mesh_after.points)

    # Create KDTree for the points in mesh_after
    kdtree_after = cKDTree(points_after)
    
    # Query KDTree to find distances and indices of nearest neighbors in mesh_after for points in mesh_before
    distances, indices = kdtree_after.query(points_before)
    
    # Filter points in mesh_before that are not within the threshold distance in mesh_after
    missing_indices = np.where(distances >= threshold)[0]
    missing_vertices = points_before[missing_indices]
    
    # Create a new point cloud with points that are missing in mesh_after
    mesh_missing = o3d.geometry.PointCloud()
    mesh_missing.points = o3d.utility.Vector3dVector(missing_vertices)
    
    # Now, filter out points in mesh_missing that have fewer than 20 neighbors within the threshold distance
    missing_vertices_np = np.asarray(mesh_missing.points)
    
    # Create KDTree for the points in mesh_missing
    kdtree_missing = cKDTree(missing_vertices_np)
    
    # Query neighbors within the threshold distance for each point in mesh_missing
    neighbor_counts = kdtree_missing.query_ball_point(missing_vertices_np, r=threshold)
    
    # Only keep points that have at least 20 neighbors in their vicinity
    valid_indices = [i for i, neighbors in enumerate(neighbor_counts) if len(neighbors) >= neighbor_threshold]
    valid_vertices = missing_vertices_np[valid_indices]
    
    # Create a new point cloud with the filtered points
    filtered_mesh_missing = o3d.geometry.PointCloud()
    filtered_mesh_missing.points = o3d.utility.Vector3dVector(valid_vertices)
    
    return filtered_mesh_missing
'''

def filter_missing_points_by_xy(mesh_before, mesh_after, x_threshold=0.0003, y_threshold=0.0001):
    # Convert points from Open3D mesh to numpy arrays
    points_before = np.asarray(mesh_before.points)
    points_after = np.asarray(mesh_after.points)
    
    # Create a KDTree for the points in mesh_after
    kdtree_after = cKDTree(points_after)
    
    # Query KDTree to find distances and indices of nearest neighbors in mesh_after for points in mesh_before
    _, indices = kdtree_after.query(points_before)
    
    # Get the y and z coordinates from both meshes
    x_before = points_before[:, 0]
    y_before = points_before[:, 1]
    x_after = points_after[indices, 0]  # Nearest neighbors' y-coordinates
    y_after = points_after[indices, 1]  # Nearest neighbors' z-coordinates
    
    # Calculate the absolute differences in the y and z coordinates
    x_diff = np.abs(x_before - x_after)
    y_diff = np.abs(y_before - y_after)
    
    # Create a mask to find points in mesh_before where either the y or z axis difference
    # with the corresponding point in mesh_after exceeds the respective thresholds
    xy_diff_mask = (x_diff >= x_threshold) | (y_diff >= y_threshold)
    
    # Select the points from mesh_before where the y or z axis difference is larger than the threshold
    missing_points = points_before[xy_diff_mask]
    
    # Create a new point cloud with the points that have significant y or z axis differences
    mesh_missing = o3d.geometry.PointCloud()
    mesh_missing.points = o3d.utility.Vector3dVector(missing_points)
    
    return mesh_missing

def calculate_lost_thickness(mesh_before, changed_mesh_after, lost_volume):
    reference_area = changed_mesh_after.get_surface_area()
    lost_thickness = lost_volume / reference_area
    
    return lost_thickness


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

def create_bbox_from_pcl(pcl):
    # Step 1: Convert point cloud to numpy array
    points = np.asarray(pcl.points)

    # Step 2: Since data is planar, project points to a 2D plane (ignore one axis, e.g., Z-axis)
    xy_points = points[:, 0:2]  # Take X and Y coordinates (planar in XY plane)

    # Step 3: Get the 2D Axis-Aligned Bounding Box (AABB) for the planar points (XY plane)
    min_bound = np.min(xy_points, axis=0)
    max_bound = np.max(xy_points, axis=0)

    # Step 4: Calculate width and height of the 2D bounding box (in XY plane)
    width = max_bound[0] - min_bound[0]  # X-axis difference
    height = max_bound[1] - min_bound[1]  # Y-axis difference

    # Step 5: Calculate the 2D area (in XY plane)
    area = width * height

    print(f"2D Bounding Box in XY plane - Width: {width}, Height: {height}, Area: {area}")
    
    # Step 6: Create the bounding box as a LineSet for visualization
    centroid = np.mean(points, axis=0)  # Get the centroid of the point cloud

    # Define the 3D bounding box corners (aligned to XY plane)
    bbox_corners = np.array([
        [min_bound[0], min_bound[1], centroid[2]],  # Min XY point
        [max_bound[0], min_bound[1], centroid[2]],  # Max X, Min Y
        [max_bound[0], max_bound[1], centroid[2]],  # Max XY
        [min_bound[0], max_bound[1], centroid[2]],  # Min X, Max Y
    ])

    # Define the lines connecting the corners
    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bounding box edges
    ]

    # Create the LineSet for the bounding box
    bbox_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_corners),
        lines=o3d.utility.Vector2iVector(bbox_lines)
    )
    bbox_lineset.paint_uniform_color([1, 0, 0])  # Red for the bounding box

    # Step 7: Create a small sphere at the centroid for visualization
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0,0,0])
    axes.translate(centroid)

    return width, height, area, bbox_lineset, axes

def create_bbox_from_pcl_axis_aligned(pcl):
    # Step 1: Convert point cloud to numpy array
    points = np.asarray(pcl.points)

    # Step 2: Since data is planar, project points to a 2D plane (ignore one axis, e.g., Z-axis)
    xy_points = points[:, 0:2]  # Take X and Y coordinates (planar in XY plane)

    # Step 3: Get the 2D Axis-Aligned Bounding Box (AABB) for the planar points (XY plane)
    min_bound = np.min(xy_points, axis=0)
    max_bound = np.max(xy_points, axis=0)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.append(min_bound, 0), 
                                               max_bound=np.append(max_bound, 0))
    bbox = bbox.get_minimal_oriented_bounding_box()

    width = max_bound[0] - min_bound[0]
    height = max_bound[1] - min_bound[1]
    area = width * height

    centroid = bbox.get_center()
    # Step 7: Create a small sphere at the centroid for visualization
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0,0,0])
    axes.translate(centroid)
    o3d.visualization.draw_geometries([pcl, bbox, axes],
                                      zoom=0.5,
                                      front=[-1, 0, 0],
                                      lookat=centroid,
                                      up=[0, 0, 1])

    return width, height, area, bbox, axes

def compute_convex_hull_area_xy(point_cloud):
    # Step 1: Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    
    # Step 2: Project the points onto the XY plane
    xy_points = points[:, 0:2]  # Extract only the X and Y coordinates
    
    # Step 3: Compute the convex hull using scipy's ConvexHull on the projected XY points
    hull_2d = ConvexHull(xy_points)
    
    # Step 4: The area of the convex hull (in the XY plane)
    area = hull_2d.area

    # Step 4: Create a new point cloud that includes the convex hull points for visualization
    hull_points_3d = points[hull_2d.vertices, :]  # Get the 3D points corresponding to the convex hull
    
    # Step 5: Create Open3D point cloud for the hull and original points
    hull_pcd = o3d.geometry.PointCloud()
    hull_pcd.points = o3d.utility.Vector3dVector(hull_points_3d)
    
    # Step 6: Create a LineSet to represent the convex hull
    lines = []
    for simplex in hull_2d.simplices:
        lines.append([simplex[0], simplex[1]])
        
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optional: Add colors to the lines
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for hull lines
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return area, hull_pcd, line_set

def sort_plate_cluster(pcd, eps=0.0005, min_points=20, remove_outliers=False, use_downsampling=False, downsample_voxel_size=0.0002):
    if use_downsampling and downsample_voxel_size > 0:
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    else:
        downsampled_pcd = pcd

    # Step 2: Segment downsampled point cloud into clusters using DBSCAN
    labels = np.array(downsampled_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # Number of clusters (label -1 indicates noise)
    num_clusters = labels.max() + 1
    if num_clusters == 0:
        return o3d.geometry.PointCloud()  # Return empty point cloud if no clusters are found

    # Step 3: Find the largest cluster in the downsampled point cloud
    max_cluster_size = 0
    largest_cluster_indices = None

    for cluster_idx in range(num_clusters):
        # Get the indices of the points that belong to the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        # If this cluster is the largest we've found, update the largest cluster info
        if len(cluster_indices) > max_cluster_size:
            max_cluster_size = len(cluster_indices)
            largest_cluster_indices = cluster_indices

    if largest_cluster_indices is None:
        return o3d.geometry.PointCloud()  # Return empty point cloud if no largest cluster is found

    # Step 4: Map the largest cluster back to the original point cloud
    largest_cluster_pcd = downsampled_pcd.select_by_index(largest_cluster_indices)

    # Find corresponding points in the original high-resolution point cloud
    distances = pcd.compute_point_cloud_distance(largest_cluster_pcd)
    original_cluster_indices = np.where(np.asarray(distances) < downsample_voxel_size*10)[0]  # Tolerance to find nearest neighbors
    high_res_largest_cluster_pcd = pcd.select_by_index(original_cluster_indices)

    # Optionally remove outliers from the largest cluster
    if remove_outliers and high_res_largest_cluster_pcd is not None:
        high_res_largest_cluster_pcd, _ = high_res_largest_cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return high_res_largest_cluster_pcd

def sort_plate_cluster_centroid(pcd, distance_threshold=0.05, remove_outliers = True):
    # Step 1: Calculate the centroid of the point cloud
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    print(f"Centroid: {centroid}")
    
    # Step 2: Calculate the distance of each point from the centroid
    distances = np.linalg.norm(np.asarray(pcd.points) - centroid, axis=1)
    
    # Step 3: Filter points based on the distance threshold
    inlier_indices = np.where(distances < distance_threshold)[0]
    
    # Select inliers (points within the distance threshold)
    filtered_pcd = pcd.select_by_index(inlier_indices)

    # Optionally: Remove statistical outliers
    if remove_outliers and filtered_pcd is not None:
        filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Color the points (filtered points will be colored, rest will be black)
    colors = np.zeros((len(pcd.points), 3))  # Initialize all points to black
    colors[inlier_indices] = [1, 0, 0]  # Color inliers red
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with colored inliers
    o3d.visualization.draw_geometries([pcd])

    return filtered_pcd



def sort_largest_cluster(pcd, eps=0.005, min_points=30, remove_outliers=True):
    # Step 1: Segment point cloud into clusters using DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # Number of clusters (label -1 indicates noise)
    num_clusters = labels.max() + 1
    print(f"PointCloud has {num_clusters} clusters")

    colors = plt.get_cmap("tab20")(labels / (num_clusters if num_clusters > 0 else 1))
    colors[labels == -1] = [0, 0, 0, 1]  # Color noise points black
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

    # Step 2: Find the largest cluster
    max_cluster_size = 0
    largest_cluster_pcd = None

    for cluster_idx in range(num_clusters):
        # Get the indices of the points that belong to the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        
        # If this cluster is the largest we've found, update the largest cluster info
        if len(cluster_indices) > max_cluster_size:
            max_cluster_size = len(cluster_indices)
            largest_cluster_pcd = pcd.select_by_index(cluster_indices)

    print(f"Largest cluster has {max_cluster_size} points")

    # Optionally: Remove outliers (if remove_outliers is set to True)
    #if remove_outliers and largest_cluster_pcd is not None:
    #    largest_cluster_pcd, _ = largest_cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if largest_cluster_pcd is None:
        largest_cluster_pcd = o3d.geometry.PointCloud()

    return largest_cluster_pcd

def transform_to_local_pca_coordinates(pcd, pca_basis, centroid):
    points = np.asarray(pcd.points)
    centered_points = points - centroid
    local_points = centered_points @ pca_basis.T
    local_pcl = o3d.geometry.PointCloud()
    local_pcl.points = o3d.utility.Vector3dVector(local_points)
    return local_pcl

def transform_to_global_coordinates(local_pcl, pca_basis, centroid):
    local_points = np.asarray(local_pcl.points)
    
    # Step 1: Apply the inverse PCA transformation (PCA basis transpose, since it's orthonormal)
    global_points = local_points @ pca_basis
    
    # Step 2: Add the centroid back to translate the points back to the original coordinate system
    global_points += centroid
    
    # Step 3: Create a new PointCloud with global coordinates
    global_pcl = o3d.geometry.PointCloud()
    global_pcl.points = o3d.utility.Vector3dVector(global_points)
    
    return global_pcl