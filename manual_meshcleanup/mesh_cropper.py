#For Cropping Point Cloud
from mesh_processor import MeshProcessor

import os
import open3d as o3d
import pandas as pd
import numpy as np


mstore = MeshProcessor()

mstore.load_mesh(1)

if mstore.mesh1_pcl == None:
    mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=100000)

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
cropped_pcd = mstore.mesh1.select_by_index(cropped_indices)

# Save the cropped point cloud
o3d.io.write_point_cloud("cropped_point_cloud.ply", cropped_pcd)

