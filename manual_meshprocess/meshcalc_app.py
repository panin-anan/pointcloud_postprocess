import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree, Delaunay
import signal
import sys


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


def signal_handler(sig, frame):
    print("Interrupt received, shutting down...")
    # Close windows
    if app.vis_mesh1:
        app.vis_mesh1.destroy_window()
    if app.vis_mesh2:
        app.vis_mesh2.destroy_window()
    if app.vis_unchanged:
        app.vis_unchanged.destroy_window()
    if app.vis_changed:
        app.vis_changed.destroy_window()
    root.destroy()
    sys.exit(0)


class MeshApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Visualization and Analysis Tool")

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
        tk.Button(self.root, text="Overlay Mesh", command=self.show_overlay).pack(pady=5)
        tk.Button(self.root, text="Probe Points Mesh 1", command=self.probe_points_mesh1).pack(pady=5)
        tk.Button(self.root, text="Probe Points Mesh 2", command=self.probe_points_mesh2).pack(pady=5)
        tk.Button(self.root, text="Compute All", command=self.compute_all).pack(pady=5)
        tk.Button(self.root, text="Show Area with Difference", command=self.show_changed).pack(pady=5)


        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=5)

        # Progress label
        self.progress_label = tk.Label(self.root, text="Progress: ")
        self.progress_label.pack(pady=5)


    def load_mesh(self, mesh_number):
        path = filedialog.askopenfilename(title=f"Select the mesh file for Mesh {mesh_number}",
                                          filetypes=[("PLY files", "*.ply"), ("All Files", "*.*")])
        if path:
            try:
                # Try loading as a triangle mesh
                mesh = o3d.io.read_triangle_mesh(path)
                if len(mesh.triangles) > 0:
                    messagebox.showinfo("Success", f"Mesh {mesh_number} Loaded Successfully (Triangle Mesh)")
                else:
                    # If no triangles, load as a point cloud
                    mesh = o3d.io.read_point_cloud(path)
                    if len(mesh.points) > 0:
                        messagebox.showinfo("Success", f"Mesh {mesh_number} Loaded Successfully (Point Cloud)")
                    else:
                        messagebox.showwarning("Warning", f"File for Mesh {mesh_number} contains no recognizable mesh or point cloud data.")

                # Assign to the corresponding mesh
                if mesh_number == 1:
                    self.mesh1 = mesh
                elif mesh_number == 2:
                    self.mesh2 = mesh
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Mesh {mesh_number}: {str(e)}")
        else:
            messagebox.showwarning("Warning", f"No file selected for Mesh {mesh_number}")


    def load_mesh1(self):
        self.load_mesh(mesh_number=1)

    def load_mesh2(self):
        self.load_mesh(mesh_number=2)

    def show_overlay(self):
        if self.mesh1 and self.mesh2:
            overlay_meshes = [self.mesh1, self.mesh2]
            vis_overlay = o3d.visualization.Visualizer()
            vis_overlay.create_window(window_name="Overlay Meshes", width=800, height=600, left=50, top=50)
            for mesh in overlay_meshes:
                vis_overlay.add_geometry(mesh)

            while True:
                vis_overlay.poll_events()
                vis_overlay.update_renderer()
        
                # Optionally, add a condition to break the loop, e.g., a key press or window close event
                if not vis_overlay.poll_events():
                    break
            vis_overlay.destroy_window()
        else:
            messagebox.showwarning("Warning", "Mesh not available")

    def show_changed(self):
        if self.changed_mesh:
            self.create_visualizer(self.changed_mesh, "Changed Vertices", 800, 600, 900, 700)
        else:
            messagebox.showwarning("Warning", "Changed mesh not available")

    def create_visualizer(self, geometry, window_name, width, height, left, top):
        vis = o3d.visualization.VisualizerWithEditing(-1, False, "")
        vis.create_window(window_name=window_name, width=width, height=height, left=left, top=top)
        vis.add_geometry(geometry)
        vis.run()
        vis.destroy_window()


    def probe_points_mesh1(self):
        if self.mesh1:
            mesh_1_pcl = self.mesh1.sample_points_uniformly(number_of_points=100000)
            picked_ids = self.pick_points(mesh_1_pcl)
            if picked_ids:
                points = np.asarray(mesh_1_pcl.points)
                print("Picked point coordinates:")
                for i in picked_ids:
                    print(f"Point {i}: [{points[i][0]:.6f}, {points[i][1]:.6f}, {points[i][2]:.6f}]")
        else:
            messagebox.showwarning("Warning", "Please load Mesh first")

    def probe_points_mesh2(self):
        if self.mesh2:
            mesh_2_pcl = self.mesh2.sample_points_uniformly(number_of_points=100000)
            picked_ids = self.pick_points(mesh_2_pcl)
            if picked_ids:
                points = np.asarray(mesh_2_pcl.points)
                print("Picked point coordinates:")
                for i in picked_ids:
                    print(f"Point {i}: [{points[i][0]:.6f}, {points[i][1]:.6f}, {points[i][2]:.6f}]")
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
        self.progress_label.config(text="Progress: Filtering points...")
        self.root.update_idletasks()
        self.unchanged_mesh, self.changed_mesh = filter_unchangedpointson_mesh(self.mesh1, self.mesh2, threshold=0.05)


        # Calculate the lost volume, thickness, and change in curvature
        lost_volume = calculate_lost_volume(self.mesh1, self.changed_mesh)
        lost_thickness = calculate_lost_thickness(self.mesh1, self.changed_mesh, lost_volume)

        avg_before = compute_average_z(self.mesh1)
        avg_after = compute_average_z(self.changed_mesh)
        avg_diff = np.abs(avg_before - avg_after)

        #change_in_curvature = calculate_change_in_curvature(self.mesh1, self.mesh2)

        print(f"Estimated volume of lost material: {lost_volume} mm^3")
        print(f"Estimated grinded thickness mesh method: {lost_thickness} mm")
        print(f"Estimated grinded thickness avg method: {avg_diff} mm")
        #print(f"Mean change in curvature: {change_in_curvature}")

        #self.progress_label.config(text="Progress: Calculating curvature...")
        #self.root.update_idletasks()
        #mean_curvature_before, overall_curvature_before = calculate_curvature(self.mesh1)
        #mean_curvature_after, overall_curvature_after = calculate_curvature(self.changed_mesh)

        #print(f"Estimated curvature before: {overall_curvature_before}")
        #print(f"Estimated curvature after: {overall_curvature_after}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    root = tk.Tk()
    app = MeshApp(root)
    root.mainloop()
