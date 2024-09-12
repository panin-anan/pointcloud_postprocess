import open3d as o3d
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk

class MeshProcessor:
    def __init__(self):
        self.mesh1 = None
        self.mesh2 = None
        self.mesh1_pcl = None
        self.mesh2_pcl = None
        self.worn_sections = []
        self.desired_sections = []
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

    #TO DO: develop this more as it is now only 3 segments and cannot handle complex shape
    def section_leading_edge(self, input_data, num_segments=3, mid_ratio=0.4, use_bounds=None):
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

    def joggle_points(self, pcd, scale=1e-6):
        points = np.asarray(pcd.points)
        jitter = np.random.normal(scale=scale, size=points.shape)
        pcd.points = o3d.utility.Vector3dVector(points + jitter)

    def create_mesh_from_point_cloud(self, pcd):
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
