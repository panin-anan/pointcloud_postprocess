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
    calculate_lost_volume_from_changedpcl,
    filter_unchangedpointson_mesh,
    calculate_lost_thickness,
    compute_average_x,
    compute_average_y,
    compute_average_z,
    calculate_curvature,
    calculate_point_density,
    create_mesh_from_point_cloud
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
        self.changed_mesh_surf = None
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
            mesh1_colored = self.mesh1.paint_uniform_color([1, 0, 0])  # Red color for mesh1
            mesh2_colored = self.mesh2.paint_uniform_color([0, 1, 0])  # Green color for mesh2
            
            overlay_meshes = [mesh1_colored, mesh2_colored]
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
            picked_ids = self.pick_points(self.mesh1)
            if picked_ids:
                points = np.asarray(self.mesh1.points)
                print("Picked point coordinates:")
                for i in picked_ids:
                    print(f"Point {i}: [{points[i][0]:.6f}, {points[i][1]:.6f}, {points[i][2]:.6f}]")
        else:
            messagebox.showwarning("Warning", "Please load Mesh first")

    def probe_points_mesh2(self):
        if self.mesh2:
            picked_ids = self.pick_points(self.mesh2)
            if picked_ids:
                points = np.asarray(self.mesh2.points)
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
        # Filter Mesh
        self.progress_label.config(text="Progress: Filtering points...")
        self.root.update_idletasks()
        self.changed_mesh = filter_unchangedpointson_mesh(self.mesh1, self.mesh2, threshold=0.0003, neighbor_threshold=8)

        mesh1_colored = self.changed_mesh.paint_uniform_color([1, 0, 0])  # Red color
        mesh2_colored = self.mesh2.paint_uniform_color([0, 1, 0])  # Green color
            
        overlay_meshes = [mesh1_colored, mesh2_colored]
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

        self.changed_mesh_surf = create_mesh_from_point_cloud(self.changed_mesh) 

        # Setup Open3D visualizer for the surface mesh
        self.changed_mesh_surf.paint_uniform_color([0, 0, 1])  # Blue color for changed surface mesh
        o3d.visualization.draw_geometries([self.changed_mesh_surf], window_name="Changed Mesh Surface", width=800, height=600)
        fixed_thickness = 0.002 # in m

        lost_volume = calculate_lost_volume_from_changedpcl(self.changed_mesh_surf, fixed_thickness)
        
        print(f"Estimated volume of lost material: {lost_volume} mm^3")
        #print(f"Estimated grinded thickness mesh method: {lost_thickness} mm")
        #print(f"Estimated grinded thickness avg method: {avg_diff} mm")

        print(f"Done computing")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    root = tk.Tk()
    app = MeshApp(root)
    root.mainloop()
