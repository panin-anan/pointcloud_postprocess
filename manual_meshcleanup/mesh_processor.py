import open3d as o3d
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk
import random
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class MeshProcessor:
    def __init__(self):
        self.mesh1 = None
        self.mesh2 = None
        self.mesh1_pcl = None
        self.mesh2_pcl = None
        self.mesh1_LE_points = None
        self.mesh2_LE_points = None
        self.mesh1_segments = None                  #not in use for axis-based 
        self.mesh2_segments = None                  #not in use for axis-based
        self.mesh1_sections = None
        self.mesh2_sections = None
        self.worn_mesh_sections = []                #not in use for axis-based
        self.desired_mesh_sections = []             #not in use for axis-based
        self.lost_volumes = []
        self.grind_params = []
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

        normalized_curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))
    
        # Assign colors based on normalized curvature
        colors = plt.get_cmap('jet')(normalized_curvatures)[:, :3]  # Use colormap 'jet' and ignore alpha channel
        pcd.colors = o3d.utility.Vector3dVector(colors)

        '''
        # Visualize the point cloud with curvature-based colors
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Curvature Visualization', width=800, height=600)
        vis.add_geometry(pcd)

        vis.run()  # Open the visualizer window
        vis.destroy_window()  # Close the window once interaction is done

        # After running, the selected points are stored in the VisualizerWithEditing object
        picked_indices = vis.get_picked_points()

        # Print curvature values for the selected points
        for idx in picked_indices:
            print(f"Curvature at point {idx}: {curvatures[idx]}")
        '''

        return np.array(curvatures)

    #Tuning parameter for testing CAD blade: curvature_threshold=(0.005, 0.04), k_neighbors=30, vicinity_radius=20, min_distance=40

    def detect_leading_edge_by_curvature(self, pcd, curvature_threshold=(0.01, 0.2), k_neighbors=30, vicinity_radius=0.004, min_distance=0.004):
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
        
        '''
        #visualize
        leading_edge_pcd = o3d.geometry.PointCloud()
        leading_edge_pcd.points = o3d.utility.Vector3dVector(filtered_leading_edge_points)
        leading_edge_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, leading_edge_pcd])
        '''

        filtered_leading_edge_points = np.array(filtered_leading_edge_points)
        #filtered_leading_edge_points = self.remove_outliers(filtered_leading_edge_points)

        return filtered_leading_edge_points

    def remove_outliers(self, points):
        """Remove outliers based on 2 standard deviation rule."""

        tree = cKDTree(points)

        distances = []
        for point in points:
            dist, _ = tree.query(point, k=2)  # k=2 to avoid self-distance
            distances.append(dist[1])  # Second nearest neighbor distance

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Use 2 standard deviations to filter outliers
        inliers = []
        for i, point in enumerate(points):
            if np.abs(distances[i] - mean_distance) <= 2 * std_distance:
                inliers.append(point)

        return np.array(inliers)

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


    def compute_orthogonal_vectors(self, input_vector, plane_normal):
        """Compute two vectors orthogonal to the input_vector, ensuring one lies along the cross-section plane."""
        # Ensure the input vector is normalized
        input_vector /= np.linalg.norm(input_vector)

        # Adjust arbitrary vector selection
        arbitrary_vector = np.array([1, 0, 0]) if np.abs(input_vector[0]) < 0.9 else np.array([0, 1, 0])

        # Compute the first orthogonal vector, ensuring it's in the plane of the cross-section
        perp_vector1 = np.cross(plane_normal, input_vector)
        perp_vector1 /= np.linalg.norm(perp_vector1)

        # Compute the second orthogonal vector which will lie in the cross-section plane
        perp_vector2 = np.cross(input_vector, perp_vector1)
        perp_vector2 /= np.linalg.norm(perp_vector2)

        return input_vector, perp_vector1, perp_vector2

    def visualize_vectors(self, origin, vectors, scale=1.0, colors=None):
        """Visualize vectors as lines in Open3D."""
        if colors is None:
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Default to red, green, blue

        lines = [[0, 1], [0, 2], [0, 3]]  # Connecting origin to the ends of the vectors
        points = [origin]  # Add origin point

        # Add scaled vectors as points
        for vector in vectors:
            points.append(origin + vector * scale)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )

        # Assign colors to each vector
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    #TO DO: develop this more as it is now only 3 segments and cannot handle complex shape
    def section_leading_edge(self, input_segment, num_sections=3, mid_ratio=0.4, use_bounds=None, axis=0):
        points = np.asarray(input_segment.vertices if isinstance(input_segment, o3d.geometry.TriangleMesh) else input_segment.points)
        vis_elements = []
        # Sort by the specified axis
        leading_edge_points = self.detect_leading_edge_by_curvature(input_segment, curvature_threshold=(0.01, 0.2), k_neighbors=40, vicinity_radius=0.0004, min_distance=0.0004)

        cross_section = self.slice_point_cloud_mid(input_segment, leading_edge_points, num_sections=1, threshold=0.0001)

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

        leading_edge_point_sec = find_closest_leading_edge_point(cross_section, leading_edge_points)

        # Calculate the leading edge vector
        initial_center = np.mean(cross_section, axis=0)
        leading_edge_vector = leading_edge_point_sec - initial_center
        leading_edge_vector /= np.linalg.norm(leading_edge_vector)

        plane_normal = leading_edge_points[-1] - leading_edge_points[0]
        plane_normal /= np.linalg.norm(plane_normal)

        leading_edge_vector, perpendicular_axis, tangent_axis = self.compute_orthogonal_vectors(leading_edge_vector, plane_normal) 

        line_set = self.visualize_vectors(
            origin=initial_center,
            vectors=[leading_edge_vector, perpendicular_axis, tangent_axis],
            scale=20.0  # Adjust scale for visual clarity
        )
        vis_elements.append(line_set)

        initial_center = np.mean(points, axis=0)

        # Compute dot products (projections onto perpendicular vector)
        projections = np.dot(points - initial_center, perpendicular_axis)

        # Compute the range of projections to get min and max values
        min_val = projections.min()
        max_val = projections.max()
        total_range = max_val - min_val

        # Calculate the center width and side widths based on mid_ratio
        center_width = total_range * mid_ratio
        side_width = (total_range - center_width) / 2

        # Define boundaries for left, center, and right sections
        bounds = [
            min_val,                         # Start of the left section
            min_val + side_width,            # End of the left section and start of the center
            min_val + side_width + center_width,  # End of the center section and start of the right
            max_val                          # End of the right section
        ]

        # Initialize containers for left, center, and right sections
        left_section = []
        center_section = []
        right_section = []
        sub_section = []

        # Iterate through all points to categorize them based on their projections
        for i, projection in enumerate(projections):
            point = points[i]
            if projection < bounds[1]:
                # Point belongs to the left section
                left_section.append(point)
            elif bounds[1] <= projection < bounds[2]:
                # Point belongs to the center section
                center_section.append(point)
            else:
                # Point belongs to the right section
                right_section.append(point)

        # Convert the sections (numpy arrays) to Open3D PointCloud objects
        left_pcd = o3d.geometry.PointCloud()
        left_pcd.points = o3d.utility.Vector3dVector(np.array(left_section))
        left_pcd.paint_uniform_color([1, 0, 0])  # Red for left section

        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(np.array(center_section))
        center_pcd.paint_uniform_color([0, 1, 0])  # Green for center section

        right_pcd = o3d.geometry.PointCloud()
        right_pcd.points = o3d.utility.Vector3dVector(np.array(right_section))
        right_pcd.paint_uniform_color([0, 0, 1])  # Blue for right section

        sub_sections = {left_pcd, center_pcd, right_pcd}

        vis_elements.append(left_pcd)
        vis_elements.append(center_pcd)
        vis_elements.append(right_pcd)

        # Visualize all sections and vectors
        #o3d.visualization.draw_geometries(vis_elements, window_name="Sectioned Point Cloud with Vectors", width=800, height=600)

        return sub_sections, bounds

    def section_leading_edge_on_segmentedPCL(self, segmented_point_clouds, leading_edge_points, num_sections=3, mid_ratio=0.6, use_bounds=None):
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

    def calculate_lost_volume(self, mesh_1, mesh_2, pcd_1, pcd_2):
        mesh_1.compute_vertex_normals()
        mesh_2.compute_vertex_normals()

        distances = pcd_2.compute_point_cloud_distance(pcd_1)
        distances = np.asarray(distances)

        reference_area = mesh_2.get_surface_area()
        volume_lost = np.mean(distances) * reference_area

        return volume_lost

    def random_color(self):
        """Generate a random RGB color."""
        return [random.random(), random.random(), random.random()]

    def slice_point_cloud_mid(self, point_cloud, leading_edge_points, num_sections=1, threshold=0.0001):
        """Slice the point cloud into sections using leading edge points."""
        vis_element = []
        def extract_points_on_plane(point_cloud, plane_point, plane_normal, threshold=0.0001):
            """Extract points lying near a specified plane."""
           
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            distances = np.abs(np.dot(np.asarray(point_cloud.points) - plane_point, plane_normal))

            mask = distances < threshold
            points_on_plane = np.asarray(point_cloud.points)[mask]

            points_on_plane_cloud = o3d.geometry.PointCloud()
            points_on_plane_cloud.points = o3d.utility.Vector3dVector(points_on_plane)

            return points_on_plane_cloud

        start_point = leading_edge_points[0]
        end_point = leading_edge_points[-1]
        midpoint = (start_point + end_point) / 2

        flow_axis = end_point - start_point
        flow_axis /= np.linalg.norm(flow_axis)

        
        points_on_plane = extract_points_on_plane(point_cloud, midpoint, flow_axis, threshold)
        '''
        points_on_plane.paint_uniform_color([0, 0, 0])
        vis_element.append(points_on_plane)
        o3d.visualization.draw_geometries(vis_element)
        '''
        return np.asarray(points_on_plane.points)

    def rotate_point_cloud(self, pcd, theta_x=0, theta_y=0, theta_z=0):
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
