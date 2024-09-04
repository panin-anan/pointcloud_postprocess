import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import copy
from scipy.spatial import cKDTree, Delaunay
import signal
import sys


# Preprocess Point Cloud 

def LE_radius_from_pcl(point_cloud):
    """
    Fit a circle to the leading edge points and return the radius.
    """
    points = np.asarray(point_cloud.points)
    # Helper functions for circle fitting

    def calc_R(xc, yc, points):
        """ calculate the distance of each 2D point from the center (xc, yc) """
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f(c, points):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c, points)
        return Ri - Ri.mean()

    # We assume the leading edge is in the XY plane and use only x and y for the circle fit
    xy_points = points[:, :2]  # Take only the x and y coordinates

    # Estimate the center as the mean of the points
    center_estimate = np.mean(xy_points, axis=0)
    
    # Perform least squares fitting
    center, ier = leastsq(f, center_estimate, args=(xy_points,))
    
    # Get the fitted radius
    LE_radius = calc_R(*center, xy_points).mean()

    return center, LE_radius


def find_leading_edge_center(pcl, target_radius):
    """
    Find the center of the leading edge based on the point cloud.
    Assumes that the leading edge has the largest X values.
    """
    points = np.asarray(pcl.points)

    #TO DO: make this axis independent
    max_x = np.max(points[:, 0])
    
    # Select points within radius distance from the maximum X value
    leading_edge_points = points[points[:, 0] >= max_x - target_radius]
    
    # Compute the center of the leading edge based on the XY coordinates of these points
    le_center_x = max_x - target_radius
    le_center_y = np.mean(leading_edge_points[:, 1])
    le_center_z = np.mean(leading_edge_points[:, 2])
    
    return le_center_x, le_center_y, le_center_z


def reshape_to_LE(pcl, le_center, target_radius, flow_axis = "z"):
    """
    Reshape point cloud to a leading edge profile
    This assumes that the point cloud is oriented correctly, with Z as the flow axis.
    TO DO: make this axis independent
    """
    points = np.asarray(pcl.points)

    # Axis-independent: Determine which two axes to use for reshaping
    if flow_axis == "z":
        idx_x, idx_y, idx_z = 0, 1, 2  # Use X, Y for reshaping, Z is flow
    elif flow_axis == "x":
        idx_x, idx_y, idx_z = 1, 2, 0  # Use Y, Z for reshaping, X is flow
    elif flow_axis == "y":
        idx_x, idx_y, idx_z = 0, 2, 1  # Use X, Z for reshaping, Y is flow
    else:
        raise ValueError("Invalid flow axis. Must be 'x', 'y', or 'z'.")

    transformed_points = []
    for point in points:
        coords = list(point)
        x, y, z = coords[idx_x], coords[idx_y], coords[idx_z]

        # Shift the point to the LE center in the XY-plane
        x_shifted = x - le_center[0]
        y_shifted = y - le_center[1]

        # Calculate the distance from the leading edge center in the XY plane
        r = np.sqrt(x_shifted**2 + y_shifted**2)  # Distance from the LE center
        
        # Transform only if the point lies outside the desired target radius
        if r >= target_radius and x_shifted > 0:                #TO DO: make this axis independent
            scaling_factor = target_radius / r
            x_new = le_center[0] + x_shifted * scaling_factor  # Scale and shift back
            y_new = le_center[1] + y_shifted * scaling_factor  # Scale and shift back
        #elif r >= target_radius and x_shifted > 0 
        else:
            # Keep the points within the target radius unchanged
            x_new = x
            y_new = y

        # Append the transformed point (Z stays the same)
        transformed_points.append([x_new, y_new, z])

    # Create a new point cloud from the transformed points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    return transformed_pcd

'''
# Step 2: Define CAD model alignment
def align_cad_model(cad_model, point_cloud):
    # Assume CAD model is a spline or NURBS surface
    # We use an example of aligning curves using 2D slice approximation

    # Extract relevant cross sections (simplified)
    num_sections = 10
    sections = np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), num_sections)
    
    aligned_sections = []
    for section in sections:
        # Take cross-section slice of point cloud
        slice_points = point_cloud[(point_cloud[:, 0] >= section - 0.05) & (point_cloud[:, 0] <= section + 0.05)]
        
        # Fit a spline through the points (CAD model)
        tck, u = splprep([slice_points[:, 1], slice_points[:, 2]], s=0)
        spline = splev(np.linspace(0, 1, 100), tck)
        
        aligned_sections.append(spline)
    
    return aligned_sections
'''
# Generate Robot Path
'''
def generate_toolpath(aligned_sections):
    # To Be Developed
    toolpath = []
    
    for section in aligned_sections:
        for i in range(len(section[0])):
            toolpath.append([section[0][i], section[1][i], 0])  # Add Z-depth, assuming flat
    
    return np.array(toolpath)
'''



# Main

# Load point cloud
# FOR NOW: Load mesh and sample points to get point cloud
# TO DO: change to directly load point cloud when they are available
path = filedialog.askopenfilename(title="Select the mesh file",
                                          filetypes=[("PLY files", "*.ply")])
mesh_1 = o3d.io.read_triangle_mesh(path)
mesh_1_pcl = mesh_1.sample_points_uniformly(number_of_points=100000)


LE_radius = 1.5    #Default value, comment out LE_radius_from_pcl method to use default value
#This method is meant for obtaining the standard LE radius of a part design.     
#LE_radius = LE_radius_from_pcl(mesh_1_pcl)     

print(f"Leading Edge Radius: {LE_radius}")

le_center = find_leading_edge_center(mesh_1_pcl, LE_radius)
print(f"Estimated Leading Edge Center: {le_center}")
mesh_1_LE = reshape_to_LE(mesh_1_pcl, le_center, LE_radius)

vis = o3d.visualization.Visualizer()
vis.create_window("Mesh 1 LE", 800, 600, 50, 700)
vis.add_geometry(mesh_1_LE)
    
while True:
    vis.poll_events()
    vis.update_renderer()
    
    # Optionally, add a condition to break the loop, e.g., a key press or window close event
    if not vis.poll_events():
        break
vis.destroy_window()

#aligned_sections = align_cad_model(None, preprocessed_cloud)  # Assuming CAD model loaded

