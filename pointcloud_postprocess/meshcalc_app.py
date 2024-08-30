import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree, Delaunay

# Import functions from mesh_calculations.py
from mesh_calculations import (
    calculate_lost_volume,
    filter_unchangedpointson_mesh,
    calculate_lost_thickness,
    compute_average_x,
    compute_average_y,
    compute_average_z,
    calculate_curvature,
    calculate_point_density
)


class MeshApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Visualization and Analysis Tool")
        
        # Initialize Open3D visualizers
        self.vis_mesh1 = None
        self.vis_mesh2 = None
        self.vis_unchanged = None
        self.vis_changed = None

        # Initialize meshes
        self.mesh1 = None
        self.mesh2 = None
        self.unchanged_mesh = None
        self.changed_mesh = None

        # Create UI buttons and labels
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Load Mesh 1", command=self.load_mesh1).pack(pady=5)
        tk.Button(self.root, text="Load Mesh 2", command=self.load_mesh2).pack(pady=5)
        tk.Button(self.root, text="Visualize Mesh 1", command=self.show_mesh1).pack(pady=5)
        tk.Button(self.root, text="Visualize Mesh 2", command=self.show_mesh2).pack(pady=5)
        tk.Button(self.root, text="Probe Points Mesh 1", command=self.probe_points_mesh1).pack(pady=5)
        tk.Button(self.root, text="Probe Points Mesh 2", command=self.probe_points_mesh2).pack(pady=5)
        tk.Button(self.root, text="Compute All", command=self.compute_all).pack(pady=5)
        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=5)

    def load_mesh1(self):
        path = filedialog.askopenfilename(title="Select the mesh file before grinding",
                                          filetypes=[("PLY files", "*.ply")])
        if path:
            self.mesh1 = o3d.io.read_triangle_mesh(path)
            messagebox.showinfo("Success", "Mesh Before Grinding Loaded Successfully")
        else:
            messagebox.showwarning("Warning", "No file selected")

    def load_mesh2(self):
        path = filedialog.askopenfilename(title="Select the mesh file after grinding",
                                          filetypes=[("PLY files", "*.ply")])
        if path:
            self.mesh2 = ""
            self.mesh2 = o3d.io.read_triangle_mesh(path)
            messagebox.showinfo("Success", "Mesh After Grinding Loaded Successfully")
        else:
            messagebox.showwarning("Warning", "No file selected")


    def show_mesh1(self):
        if self.mesh1:
            self.vis_mesh1= self.create_visualizer(self.mesh1, "Mesh 1", 800, 600, 50, 700)
        else:
            messagebox.showwarning("Warning", "Mesh not available")

    def show_mesh2(self):
        if self.mesh2:
            self.vis_mesh2 = self.create_visualizer(self.mesh2, "Mesh 2", 800, 600, 900, 700)
        else:
            messagebox.showwarning("Warning", "Mesh not available")

    def show_unchanged(self):
        if self.unchanged_mesh:
            self.vis_unchanged = self.create_visualizer(self.unchanged_mesh, "Unchanged Vertices", 800, 600, 50, 700)
        else:
            messagebox.showwarning("Warning", "Unchanged mesh not available")

    def show_changed(self):
        if self.changed_mesh:
            self.vis_changed = self.create_visualizer(self.changed_mesh, "Changed Vertices", 800, 600, 900, 700)
        else:
            messagebox.showwarning("Warning", "Changed mesh not available")

    def create_visualizer(self, geometry, window_name, width, height, left, top):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=width, height=height, left=left, top=top)
        vis.add_geometry(geometry)
        
        while True:
            vis.poll_events()
            vis.update_renderer()
        
        # Optionally, add a condition to break the loop, e.g., a key press or window close event
            if not vis.poll_events():
                break
        vis.destroy_window()

    def probe_points_mesh1(self):
        if self.mesh1:
            mesh_1_pcl = self.mesh1.sample_points_uniformly(number_of_points=100000)
            picked_ids = self.pick_points(mesh_1_pcl)
            if picked_ids:
                points = np.asarray(mesh_1_pcl.points)
                print("Picked point coordinates:")
                for i in picked_ids:
                    print(f"Point {i}: {points[i]}")
        else:
            messagebox.showwarning("Warning", "Please load Mesh first")

    def probe_points_mesh2(self):
        if self.mesh2:
            mesh_2_pcl = self.mesh1.sample_points_uniformly(number_of_points=100000)
            picked_ids = self.pick_points(mesh_2_pcl)
            if picked_ids:
                points = np.asarray(mesh_2_pcl.points)
                print("Picked point coordinates:")
                for i in picked_ids:
                    print(f"Point {i}: {points[i]}")
        else:
            messagebox.showwarning("Warning", "Please load Mesh first")

    def pick_points(self, pcd):
        # Create a Visualizer instance
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick Points", width=800, height=600)
        vis.add_geometry(pcd)

        print("1) Please pick points using [Shift + Left Click]")
        print("2) After picking points, press 'Q' to close the window")

        # Run the visualizer, allowing point picking
        vis.run()

        # Get the indices of the picked points
        picked_points = vis.get_picked_points()

        # Destroy the visualizer window
        vis.destroy_window()
        
        return picked_points

    def compute_all(self):
        # Postprocess data
        num_points_before, point_density_before, resolution_before = calculate_point_density(self.mesh1)
        num_points_after, point_density_after, resolution_after = calculate_point_density(self.mesh2)
        print(f"First Mesh:")
        print(f"num of points: {num_points_before}")
        print(f"point density: {point_density_before}")
        print(f"resolution: {resolution_before}")

        print(f"Second Mesh:")
        print(f"num of points: {num_points_after}")
        print(f"point density: {point_density_after}")
        print(f"resolution: {resolution_after}")
   
        # Filter Mesh
        unchanged_mesh, changed_mesh = filter_unchangedpointson_mesh(self.mesh1, self.mesh2, threshold=0.1)


        # Calculate the lost volume, thickness, and change in curvature
        lost_volume = calculate_lost_volume(self.mesh1, changed_mesh)
        lost_thickness = calculate_lost_thickness(self.mesh1, changed_mesh, lost_volume)

        avg_before = compute_average_z(self.mesh1)
        avg_after = compute_average_z(changed_mesh)
        avg_diff = np.abs(avg_before - avg_after)
        #change_in_curvature = calculate_change_in_curvature(self.mesh1, self.mesh2)

        print(f"Estimated volume of lost material: {lost_volume} mm^3")
        print(f"Estimated grinded thickness mesh method: {lost_thickness} mm")
        print(f"Estimated grinded thickness avg method: {avg_diff} mm")
        #print(f"Mean change in curvature: {change_in_curvature}")

        mean_curvature_before, overall_curvature_before = calculate_curvature(self.mesh1)
        mean_curvature_after, overall_curvature_after = calculate_curvature(self.mesh2)

        print(f"Estimated curvature before: {overall_curvature_before}")
        print(f"Estimated curvature after: {overall_curvature_after}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MeshApp(root)
    root.mainloop()
