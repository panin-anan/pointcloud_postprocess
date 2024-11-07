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
import copy
from concave_hull import concave_hull, concave_hull_indexes

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

def create_mesh_from_point_cloud(pcd, alpha = 0.01):
    #points = np.asarray(pcd.points)
    #jitter = np.random.normal(scale=1e-6, size=points.shape)
    #pcd.points = o3d.utility.Vector3dVector(points + jitter) #add jitter if points coincide
    print('estimating normals')
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(pcd.get_center() + np.array([0.1, 0, 0]))
    print('meshing')
    
    #ball pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    return mesh

def shift_y_fordiagonal(pcd):
    # Extract points
    points = np.asarray(pcd.points)

    # Define y range for applying the shift
    min_y = 0.82
    max_y = min_y + 0.006
    max_shift = 0.002

    # Calculate shift factor for points within the specified y range
    mask = (points[:, 1] > min_y) & (points[:, 1] <= max_y)
    points[mask, 0] += max_shift * (points[mask, 1] - min_y) / (max_y - min_y)

    # Update point cloud with modified points
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

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

def filter_project_points_by_plane(point_cloud, distance_threshold=0.0008):
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

    # Visualize both the inliers, projected points, and nearby points
    #o3d.visualization.draw_geometries([projected_pcd, inlier_cloud, nearby_cloud])

    return projected_pcd, pca_basis, plane_centroid

def filter_points_by_plane_nearbycloud(point_cloud, distance_threshold=0.0008, nearby_distance=0.01):
    # Fit a plane to the point cloud using RANSAC
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    [a, b, c, d] = plane_model
    
    # Select points that are close to the plane (within the threshold)
    inlier_cloud = point_cloud.select_by_index(inliers)
    pca_basis, plane_centroid = fit_plane_to_pcd_pca(inlier_cloud)
    points = np.asarray(inlier_cloud.points)

    # Select additional points within the specified nearby distance to the plane
    distances = np.abs(np.dot(np.asarray(point_cloud.points), [a, b, c]) + d) / np.linalg.norm([a, b, c])
    nearby_indices = np.where(distances <= nearby_distance)[0]
    nearby_cloud = point_cloud.select_by_index(nearby_indices)

    # Color the inlier points (on the original RANSAC plane) in green
    inlier_cloud.paint_uniform_color([0, 1, 0])  # Color inlier points in green
    nearby_cloud.paint_uniform_color([0, 0, 1])  # Color nearby points in blue

    # Visualize both the inliers, projected points, and nearby points
    #o3d.visualization.draw_geometries([projected_pcd, inlier_cloud, nearby_cloud])

    return nearby_cloud, pca_basis, plane_centroid

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


def filter_changedpoints_onNormaxis(mesh_before, mesh_after, x_threshold=0.0003, y_threshold=0.0001, x_threshold_after=0.00001, neighbor_threshold=5):
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
    x_after = points_after[indices, 0]  # Nearest neighbors' x-coordinates
    y_after = points_after[indices, 1]  # Nearest neighbors' y-coordinates
    
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
    
    # Now find points in mesh_after that are outside the x_threshold distance from mesh_missing
    kdtree_missing_x = cKDTree(missing_points[:, [0]])  # KDTree with only x-coordinates
    distances, _ = kdtree_missing_x.query(points_after[:, [0]])
    
    # Mask to filter points in mesh_after that are outside the x_threshold distance in x-axis
    change_mask = distances > x_threshold_after
    changed_points = points_after[change_mask]
    
    # Create mesh_change with points in mesh_after that are outside the x-axis threshold distance from mesh_missing
    mesh_change = o3d.geometry.PointCloud()
    mesh_change.points = o3d.utility.Vector3dVector(changed_points)
    
    return mesh_missing, mesh_change

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
    dummy_pcl = copy.deepcopy(pcl)
    
    # Step 2: Initialize variables to store the minimum width and corresponding bounding box
    min_width = float('inf')
    min_height = float('inf')
    best_bbox_corners = None
    best_axes = None
    best_angle = 0
    
    # Step 3: Iterate over angles to find the orientation with minimal bounding box width
    for angle in np.linspace(0, 2*np.pi, 100):  # 100 steps from 0 to 360 degrees
        # Step 4: Create the 2D rotation matrix around the Z-axis
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Step 5: Rotate points around the Z-axis in the XY plane
        xy_points = points[:, 0:2]  # Take X and Y coordinates (planar in XY plane)
        rotated_points = np.dot(xy_points, rotation_matrix.T)
        
        # Step 6: Get the Axis-Aligned Bounding Box (AABB) for the rotated points
        min_bound = np.min(rotated_points, axis=0)
        max_bound = np.max(rotated_points, axis=0)
        
        # Calculate width and height of the bounding box
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        
        # Step 7: If this rotation gives a smaller width, update the minimum width and bbox
        if width < min_width:
            min_width = width
            min_height = height
            min_bound_3d = np.array([min_bound[0], min_bound[1], np.mean(points[:, 2])])  # Set Z at the centroid
            max_bound_3d = np.array([max_bound[0], max_bound[1], np.mean(points[:, 2])])  # Set Z at the centroid
            
            # Define the bounding box corners in 3D
            best_bbox_corners = np.array([
                [min_bound_3d[0], min_bound_3d[1], min_bound_3d[2]],  # Min XY point
                [max_bound_3d[0], min_bound_3d[1], min_bound_3d[2]],  # Max X, Min Y
                [max_bound_3d[0], max_bound_3d[1], min_bound_3d[2]],  # Max XY
                [min_bound_3d[0], max_bound_3d[1], min_bound_3d[2]],  # Min X, Max Y
            ])
            best_angle = angle

    # Step 8: Rotate the point cloud back to the orientation with the smallest width
    final_rotation_matrix = np.array([
        [np.cos(best_angle), -np.sin(best_angle), 0],
        [np.sin(best_angle), np.cos(best_angle), 0],
        [0, 0, 1]
    ])
    rotated_pcl = dummy_pcl.rotate(final_rotation_matrix, center=(0, 0, 0))
    
    # Step 9: Create a LineSet for the bounding box edges
    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bounding box edges
    ]
    bbox_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(best_bbox_corners),
        lines=o3d.utility.Vector2iVector(bbox_lines)
    )
    bbox_lineset.paint_uniform_color([1, 0, 0])  # Red for the bounding box

    # Step 10: Create a small coordinate frame at the centroid for visualization
    centroid = np.mean(best_bbox_corners, axis=0)  # Get the centroid of the bounding box
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0, 0, 0])
    axes.translate(centroid)

    # Step 11: Visualize the result
    o3d.visualization.draw_geometries([rotated_pcl, bbox_lineset, axes],
                                      zoom=0.5,
                                      front=[-1, 0, 0],
                                      lookat=centroid,
                                      up=[0, 0, 1])

    # Step 12: Calculate the area of the bounding box
    area = min_width * min_height
    
    # Return the best bbox parameters
    return min_width, min_height, area, bbox_lineset, axes


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

def compute_concave_hull_area_xy(point_cloud, hull_convex_2d, concave_resolution=0.0005):
    points = np.asarray(point_cloud.points)
    #plt.scatter(points[:, 0], points[:, 1], s=0.5, color='b', alpha=0.5)
    idxes = concave_hull_indexes(
        points[:, :2],
        length_threshold=concave_resolution,
    )
    # you can get coordinates by `points[idxes]`
    assert np.all(points[idxes] == concave_hull(points, length_threshold=concave_resolution))

    for f, t in zip(idxes[:-1], idxes[1:]):  # noqa
        seg = points[[f, t]]
        #plt.plot(seg[:, 0], seg[:, 1], "r-", alpha=0.5)
    # plt.savefig('hull.png')
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()

    # Calculate the area using the Shoelace formula
    hull_points = points[idxes]
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    hull_cloud = o3d.geometry.PointCloud()
    hull_cloud.points = o3d.utility.Vector3dVector(hull_points)

    return area, hull_cloud

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


def calculate_volume_with_projected_boundaries_convex(pcd1, pcd2, num_slices=3):
    """
    Calculate the volume between two irregular surfaces by integrating cross-sectional areas along the y-axis.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # Define full x, y, and z bounds based on the two surfaces
    x_min = min(points1[:, 0].min(), points2[:, 0].min())
    x_max = max(points1[:, 0].max(), points2[:, 0].max())
    y_min = min(points1[:, 1].min(), points2[:, 1].min())
    y_max = max(points1[:, 1].max(), points2[:, 1].max())
    z_min = min(points1[:, 2].min(), points2[:, 2].min())
    z_max = max(points1[:, 2].max(), points2[:, 2].max())
    
    # Calculate the area of the xz-plane bounding box
    bounding_area_xz = (x_max - x_min) * (z_max - z_min)
    
    # Define y-axis slices
    y_slices = np.linspace(y_min, y_max, num_slices)
    total_volume = 0

    # Loop through each slice position
    for i in range(len(y_slices) - 1):
        y_start, y_end = y_slices[i], y_slices[i + 1]
        slice_thickness = y_end - y_start
        y_mid = (y_start + y_end) / 2 

        # Filter points within the current y-slice for each surface
        slice_points1 = points1[(points1[:, 1] >= y_start) & (points1[:, 1] < y_end)]
        slice_points2 = points2[(points2[:, 1] >= y_start) & (points2[:, 1] < y_end)]
        
        # Combine points from both surfaces for this slice
        combined_slice_points = np.vstack((slice_points1, slice_points2))
        
        if len(combined_slice_points) >= 3:
            # Project combined points onto the xz-plane at y_mid
            xz_combined_points = np.column_stack((combined_slice_points[:, 0], np.full(combined_slice_points.shape[0], y_mid), combined_slice_points[:, 2]))
            
            # Calculate the Convex Hull for the combined set
            hull = ConvexHull(xz_combined_points[:, [0, 2]])  # Use only x and z coordinates
            cross_sectional_area = hull.volume
            
            # Visualize Convex Hull in Open3D
            hull_points = o3d.geometry.PointCloud()
            hull_points.points = o3d.utility.Vector3dVector(xz_combined_points)
            
            hull_lines = [[hull.vertices[j], hull.vertices[(j + 1) % len(hull.vertices)]] for j in range(len(hull.vertices))]
            
            hull_line_set = o3d.geometry.LineSet()
            hull_line_set.points = hull_points.points
            hull_line_set.lines = o3d.utility.Vector2iVector(hull_lines)
            
            # Display the convex hull with Open3D
            o3d.visualization.draw_geometries([hull_points, hull_line_set], window_name=f"Slice {i+1} (y_mid = {y_mid:.2f})")
            
        else:
            # Use the bounding area if there are insufficient points to form a polygon
            cross_sectional_area = bounding_area_xz

        # Estimate volume for this slice
        slice_volume = cross_sectional_area * slice_thickness

        # Add the slice volume to the total volume
        total_volume += slice_volume

    # Return the total volume
    return total_volume


def calculate_volume_with_projected_boundaries_concave(pcd1, pcd2, num_slices=3, concave_resolution=0.002):
    """
    Calculate the volume between two irregular surfaces by integrating cross-sectional areas along the y-axis.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # Define full x, y, and z bounds based on the two surfaces
    x_min = min(points1[:, 0].min(), points2[:, 0].min())
    x_max = max(points1[:, 0].max(), points2[:, 0].max())
    y_min = min(points1[:, 1].min(), points2[:, 1].min())
    y_max = max(points1[:, 1].max(), points2[:, 1].max())
    z_min = min(points1[:, 2].min(), points2[:, 2].min())
    z_max = max(points1[:, 2].max(), points2[:, 2].max())
    
    # Calculate the area of the xz-plane bounding box
    bounding_area_xz = (x_max - x_min) * (z_max - z_min)
    
    # Define y-axis slices
    y_slices = np.linspace(y_min, y_max, num_slices)
    total_volume = 0

    for i in range(len(y_slices) - 1):
        y_start, y_end = y_slices[i], y_slices[i + 1]
        slice_thickness = y_end - y_start
        y_mid = (y_start + y_end) / 2

        # Filter points within the current y-slice for each surface
        slice_points1 = points1[(points1[:, 1] >= y_start) & (points1[:, 1] < y_end)]
        slice_points2 = points2[(points2[:, 1] >= y_start) & (points2[:, 1] < y_end)]
        
        # Combine points from both surfaces for this slice
        combined_slice_points = np.vstack((slice_points1, slice_points2))
        
        if len(combined_slice_points) >= 3:
            # Project combined points onto the xz-plane at y_mid
            xz_combined_points = np.column_stack((combined_slice_points[:, 0], np.full(combined_slice_points.shape[0], y_mid), combined_slice_points[:, 2]))
            xz_points_display = o3d.geometry.PointCloud()
            xz_points_display.points = o3d.utility.Vector3dVector(xz_combined_points)


            # Compute the concave hull indices
            idxes = concave_hull_indexes(xz_combined_points[:, [0, 2]], length_threshold=concave_resolution)
            hull_points = xz_combined_points[idxes]

            # Calculate the area of the concave hull using the Shoelace formula
            x = hull_points[:, 0]
            z = hull_points[:, 2]
            cross_sectional_area = 0.5 * np.abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
            
            # Prepare visualization of Concave Hull in Open3D
            hull_cloud = o3d.geometry.PointCloud()
            hull_cloud.points = o3d.utility.Vector3dVector(hull_points)
            
            # Create lines to connect hull points in sequence and close the loop
            hull_lines = [[j, (j + 1) % len(idxes)] for j in range(len(idxes))]
            
            hull_line_set = o3d.geometry.LineSet()
            hull_line_set.points = hull_cloud.points
            hull_line_set.lines = o3d.utility.Vector2iVector(hull_lines)
            
            # Display the concave hull with Open3D
            o3d.visualization.draw_geometries([xz_points_display, hull_line_set], window_name=f"Slice {i+1} (y_mid = {y_mid:.2f})")
            
        else:
            # Use the bounding area if there are insufficient points to form a polygon
            cross_sectional_area = bounding_area_xz

        # Estimate volume for this slice
        slice_volume = cross_sectional_area * slice_thickness

        # Add the slice volume to the total volume
        total_volume += slice_volume

    # Return the total volume
    return total_volume
