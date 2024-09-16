import open3d as o3d
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk
import random

class MeshProcessor:
    def __init__(self):
        self.mesh1 = None
        self.mesh2 = None
        self.mesh1_pcl = None
        self.mesh2_pcl = None
        self.mesh1_LE_points = None
        self.mesh2_LE_points = None
        self.worn_mesh_sections = []
        self.desired_mesh_sections = []
        self.lost_volumes = []
        self.y_bounds = None
        self.model = None
        self.scaler = None

    def load_mesh(self, mesh_number):
        path = filedialog.askopenfilename(title=f"Select the mesh file for Mesh {mesh_number}",
                                          filetypes=[("PLY files", "*.ply"), ("All Files", "*.*")])

        if path:
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.triangles) > 0:
                print(f"Mesh {mesh_number} loaded as Triangle Mesh.")
                if mesh_number == 1:
                    self.mesh1 = mesh
                elif mesh_number == 2:
                    self.mesh2 = mesh
            else:
                mesh_pcl = o3d.io.read_point_cloud(path)
                print(f"Mesh {mesh_number} loaded as Point Cloud.")
                if mesh_number == 1:
                    self.mesh1_pcl = mesh_pcl
                elif mesh_number == 2:
                    self.mesh2_pcl = mesh_pcl
                else:
                    messagebox.showwarning("Warning", f"Mesh {mesh_number} contains no data.")
        else:
            messagebox.showwarning("Warning", f"No file selected for Mesh {mesh_number}")
        

    def estimate_curvature(self, pcd, k_neighbors=30):
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

    def detect_leading_edge_by_curvature(self, pcd, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=20, min_distance=40):
        """Detect leading edge points based on curvature and further refine them."""
        curvatures = self.estimate_curvature(pcd, k_neighbors=k_neighbors)
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

        #visualize
        leading_edge_pcd = o3d.geometry.PointCloud()
        leading_edge_pcd.points = o3d.utility.Vector3dVector(filtered_leading_edge_points)
        leading_edge_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, leading_edge_pcd])

        return np.array(filtered_leading_edge_points)


    #TO DO: now only work with y-axis, to be axis independent and follow LE spline    
    def segment_pcd(self, input_pcd, num_segments=3, axis='z'):
        # Convert point cloud to a numpy array
        points = np.asarray(input_pcd.points)

        # Determine which axis to use for segmentation
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_dict[axis]

        # Get min and max values along the chosen axis
        axis_min, axis_max = points[:, axis_idx].min(), points[:, axis_idx].max()
        segment_size = (axis_max - axis_min) / num_segments
        segmented_point_clouds = []

        # Define a list of colors for each segment
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [0, 1, 1],  # Cyan
            [1, 0, 1],  # Magenta
        ]

        # Segment the point cloud along the specified axis
        for i in range(num_segments):
            lower_bound = axis_min + i * segment_size
            upper_bound = axis_min + (i + 1) * segment_size

            # Find the indices of points within the current segment
            segment_indices = np.where((points[:, axis_idx] >= lower_bound) & (points[:, axis_idx] < upper_bound))[0]

            # Create a new point cloud for this segment
            segment_pcd = o3d.geometry.PointCloud()
            segment_pcd.points = o3d.utility.Vector3dVector(points[segment_indices])

            # Assign color to each point based on the current segment
            color = colors[i % len(colors)]  # Cycle through the colors if num_segments > len(colors)
            segment_colors = np.tile(color, (len(segment_indices), 1))  # Repeat the color for all points
            segment_pcd.colors = o3d.utility.Vector3dVector(segment_colors)

            # Store the segmented point cloud
            segmented_point_clouds.append(segment_pcd)

        # Optionally visualize the segments with colors
        o3d.visualization.draw_geometries(segmented_point_clouds, 
                                          window_name="Segmented Point Cloud Visualization", 
                                          width=800, 
                                          height=600)

        print(f"Point cloud segmentation completed along {axis}-axis with colors!")

        return segmented_point_clouds


    def segment_turbine_pcd(self, input_pcd, leading_edge_points):
        # Convert point cloud to a numpy array
        points = np.asarray(input_pcd.points)

        # segmenting
        num_segments = len(leading_edge_points) - 1 
        print(f"Number of segments set to: {num_segments}")

        def project_point_to_leading_edge(point, leading_edge_points):
            """
            Project a point onto the leading edge and find the closest segment.
            """
            leading_edge = np.array(leading_edge_points)
            differences = leading_edge - point
            distances = np.linalg.norm(differences, axis=1)
            closest_idx = np.argmin(distances)

            # Determine which segment (between two leading edge points) the point belongs to
            if closest_idx == len(leading_edge) - 1:
                return closest_idx - 1  # If it's the last point, it belongs to the last segment
            else:
                return closest_idx

        # Define a list of colors for each segment
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [0, 1, 1],  # Cyan
            [1, 0, 1],  # Magenta
        ]

        if num_segments > len(colors):
            colors.extend(np.random.rand(num_segments - len(colors), 3).tolist())

        segments = [[] for _ in range(num_segments)]

        # Assign each point to a segment based on its projection onto the leading edge
        for point in points:
            segment_idx = project_point_to_leading_edge(point, leading_edge_points)
            if segment_idx < num_segments:  # Ensure index is within bounds
                segments[segment_idx].append(point)

        # Create separate point clouds for each segment
        segmented_point_clouds = []
        for idx, segment in enumerate(segments):
            if len(segment) > 0:
                segment_pcd = o3d.geometry.PointCloud()
                segment_pcd.points = o3d.utility.Vector3dVector(segment)
                color_array = np.tile(colors[idx % len(colors)], (len(segment), 1))  # Repeat color for all points
                segment_pcd.colors = o3d.utility.Vector3dVector(color_array)
                segmented_point_clouds.append(segment_pcd)

        
        # Optionally visualize the segments with colors
        o3d.visualization.draw_geometries(segmented_point_clouds, 
                                          window_name="Segmented Point Cloud Visualization", 
                                          width=800, 
                                          height=600)

        return segmented_point_clouds



    #TO DO: develop this more as it is now only 3 segments and cannot handle complex shape
    def section_leading_edge(self, input_segment, num_sections=3, mid_ratio=0.4, use_bounds=None, axis=0):
        points = np.asarray(input_segment.vertices if isinstance(input_segment, o3d.geometry.TriangleMesh) else input_segment.points)
    
        # Sort by the specified axis
        sorted_indices = np.argsort(points[:, axis])  # Sort by the specified axis (0=X, 1=Y, 2=Z)
        sorted_points = points[sorted_indices]

        if use_bounds is None:
            # Compute the bounds for segmentation based on the specified axis and mid_ratio
            min_val = sorted_points[0, axis]
            max_val = sorted_points[-1, axis]
            total_range = max_val - min_val

            # Allocate a larger range for the middle section
            middle_range = total_range * mid_ratio
            side_range = (total_range - middle_range) / 2

            # Define boundaries for each segment
            bounds = [
                min_val,                        # Start of the first segment
                min_val + side_range,          # End of the first segment and start of the middle segment
                min_val + side_range + middle_range,  # End of the middle segment and start of the third segment
                max_val                         # End of the third segment
            ]
        else:
            # Use predefined boundaries
            bounds = use_bounds

        sub_sections = []

        # Divide the points into sections based on the adjusted axis boundaries
        for i in range(num_sections):
            lower_bound = bounds[i]
            upper_bound = bounds[i + 1]

            mask = (sorted_points[:, axis] >= lower_bound) & (sorted_points[:, axis] < upper_bound)
            segment_indices = sorted_indices[mask]

            if isinstance(input_segment, o3d.geometry.TriangleMesh):
                sub_section = input_segment.select_by_index(segment_indices, vertex_only=True)
            else:
                sub_section = input_segment.select_by_index(segment_indices)

            sub_sections.append(sub_section)

        return sub_sections, bounds

    def section_leading_edge_on_segmentedPCL(self, segmented_point_clouds, leading_edge_points, num_sections=3, mid_ratio=0.4, use_bounds=None):
        all_sub_sections = []
        vis_element = []
    
        for i, segmented_pcd in enumerate(segmented_point_clouds):
            print(f"Processing segmented point cloud {i+1}/{len(segmented_point_clouds)}")

            # Apply the section_leading_edge function to each segmented point cloud
            sub_sections, bounds = self.section_leading_edge(segmented_pcd, num_sections=num_sections, mid_ratio=mid_ratio, use_bounds=use_bounds)
            
            # Assign color
            for sub_section in sub_sections:
                #print(f"Processing subsection {i+1}/{len(sub_sections)}")
                color = self.random_color()
                num_points = np.asarray(sub_section.points).shape[0]
                sub_section.colors = o3d.utility.Vector3dVector(np.tile(color, (num_points, 1)))

                vis_element.append(sub_section)
 
            all_sub_sections.append({'segment_id': i, 'sub_sections': sub_sections, 'bounds': bounds})

        o3d.visualization.draw_geometries(vis_element, 
                                            window_name="Sub-sections Visualization", 
                                            width=800, 
                                            height=600)
    
        return all_sub_sections, bounds

    def joggle_points(self, pcd, scale=1e-6):
        points = np.asarray(pcd.points)
        jitter = np.random.normal(scale=scale, size=points.shape)
        pcd.points = o3d.utility.Vector3dVector(points + jitter)

    def create_mesh_from_pcl(self, pcd):
        self.joggle_points(pcd) 
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)

        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [0.1 * avg_dist, 0.4 * avg_dist, 0.7 * avg_dist, 1 * avg_dist, 1.5 * avg_dist, 2 * avg_dist, 3 * avg_dist] #can reduce to reduce computation
        r = o3d.utility.DoubleVector(radii)

        #ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, r)

        return mesh

    def calculate_lost_volume(self, mesh_before, mesh_after):
        mesh_before.compute_vertex_normals()
        mesh_after.compute_vertex_normals()

        pcd_before = mesh_before.sample_points_poisson_disk(number_of_points=30000)
        pcd_after = mesh_after.sample_points_poisson_disk(number_of_points=30000)

        distances = pcd_after.compute_point_cloud_distance(pcd_before)
        distances = np.asarray(distances)

        reference_area = mesh_after.get_surface_area()
        volume_lost = np.mean(distances) * reference_area

        return volume_lost

    def random_color(self):
        """Generate a random RGB color."""
        return [random.random(), random.random(), random.random()]
