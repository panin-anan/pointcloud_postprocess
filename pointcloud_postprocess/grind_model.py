#grinding model based on input worn blade point cloud and desired LE mesh

# From recontour_LE, obtain desired LE mesh. Then this code use the scanned pcl and recontoured pcl
# to calculate grinding parameters of 3 angle of grinding to obtain LE shape based on volume difference. 
# FOR NOW: default as 3 angle of grinding: left right and middle.

import open3d as o3d
from tkinter import filedialog, messagebox
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import pymesh
'''
from mesh_calculations import (
    load_mesh, segment_leading_edge_by_y_distance, create_mesh_from_point_cloud,
    calculate_lost_volume, filter_unchangedpointson_mesh, calculate_lost_thickness,
    compute_average_x, compute_average_y, compute_average_z,
    calculate_curvature, calculate_point_density
)
'''
from mesh_processor import MeshProcessor
from grindparam_predictor import load_data, preprocess_data, train_svr, evaluate_model
from visualization import visualize_mesh, visualize_meshes_overlay, visualize_sub_section, project_worn_to_desired, visualize_lost_material


def predict_grind_params():
    file_path = 'your_dataset.csv'  # Replace with your CSV file path
    data = load_data(file_path)
    target_columns = ['RPM', 'Force']

    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_columns)
    model = train_svr(X_train, y_train)
    evaluate_model(model, X_test, y_test)


def main():

    print("Processing Meshes and Calculating Lost Volume...")

    processor = MeshProcessor()
    processor.mesh1 = processor.load_mesh(1)
    processor.mesh2 = processor.load_mesh(2)

    #because self generated tri mesh has too few vertices, sample point first and use a point cloud (optional: also regenerate mesh)
    processor.mesh1_pcl = processor.mesh1.sample_points_poisson_disk(number_of_points=25000)
    processor.mesh2_pcl = processor.mesh2.sample_points_poisson_disk(number_of_points=25000)


    processor.worn_sections, processor.y_bounds = processor.segment_leading_edge_by_y_distance(processor.mesh1_pcl, num_segments=3, mid_ratio=0.7)
    processor.desired_sections, _ = processor.segment_leading_edge_by_y_distance(processor.mesh2_pcl, num_segments=3, use_bounds=processor.y_bounds)

    processor.worn_sections = [processor.create_mesh_from_point_cloud(section) for section in processor.worn_sections]
    processor.desired_sections = [processor.create_mesh_from_point_cloud(section) for section in processor.desired_sections]

    for i, (worn, desired) in enumerate(zip(processor.worn_sections, processor.desired_sections), 1):
        lost_volume = processor.calculate_lost_volume(worn, desired)
        print(f"Lost Volume for Section {i}: {lost_volume:.2f} mm^3")
    
    visualize_meshes_overlay(processor.worn_sections, processor.desired_sections)
    visualize_lost_material(processor.worn_sections, processor.desired_sections)


if __name__ == "__main__":
    main()


'''
worn_blade_profile, y_bounds = segment_leading_edge_by_y_distance(mesh_1_pcl, num_segments=3, mid_ratio=0.7)
desired_LE_profile, _ = segment_leading_edge_by_y_distance(mesh_2_pcl, num_segments=3, use_bounds=y_bounds)

visualize_sub_section(worn_blade_profile)
visualize_sub_section(desired_LE_profile)


# Store the meshes into an array similar to worn_blade_profile
worn_blade_profile_meshes = []
desired_LE_profile_meshes = []

for pcd in worn_blade_profile:
    mesh = create_mesh_from_point_cloud(pcd)
    worn_blade_profile_meshes.append(mesh)


for pcd in desired_LE_profile:
    mesh = create_mesh_from_point_cloud(pcd)
    desired_LE_profile_meshes.append(mesh)

# Calculate the lost volume, thickness, and change in curvature
lost_volumes = []

for i in range(3):
    lost_volume = calculate_lost_volume(worn_blade_profile_meshes[i], desired_LE_profile_meshes[i])
    lost_volumes.append(lost_volume)
    print(f"Estimated material lost section {i+1}: {lost_volume} mm^3")


# Call the overlay function
visualize_meshes_overlay(worn_blade_profile_meshes, desired_LE_profile_meshes)


# Visualize the projected vertices and the lost material for each section
for i in range(3):
    print(f"Visualizing lost material for section {i+1}")
    lost_volume_visualization = project_worn_to_desired(worn_blade_profile_meshes[i], desired_LE_profile_meshes[i])
    o3d.visualization.draw_geometries([worn_blade_profile_meshes[i], desired_LE_profile_meshes[i], lost_volume_visualization])
'''


