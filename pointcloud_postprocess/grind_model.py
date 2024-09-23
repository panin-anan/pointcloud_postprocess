#grinding model based on input worn blade point cloud

from mesh_processor import MeshProcessor
from mesh_visualizer import MeshVisualizer
from grindparam_predictor import load_data, preprocess_data, train_svr, evaluate_model

import os
import open3d as o3d
import pandas as pd
import numpy as np

def save_sections(sections, prefix, directory="saved_sections"):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save each section to the designated directory
    for i, section in enumerate(sections):
        filename = os.path.join(directory, f"{prefix}_section_{i+1}.ply")
        if isinstance(section, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(filename, section)
        elif isinstance(section, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(filename, section)

def load_sections(prefix, num_sections=3, directory="saved_sections"):
    sections = []
    
    # Load each section from the designated directory
    for i in range(num_sections):
        filename = os.path.join(directory, f"{prefix}_section_{i+1}.ply")
        if os.path.exists(filename):
            mesh = o3d.io.read_triangle_mesh(filename)
            if mesh.is_empty():  # If triangle mesh fails, try loading as point cloud
                mesh = o3d.io.read_point_cloud(filename)
            sections.append(mesh)
        else:
            return None  # If any section is missing, return None to indicate processing is required
    return sections

def create_grind_model(mstore):
    """
    Train the model if it does not exist yet in the MeshProcessor object.
    :param mstore: MeshProcessor object that holds the model and data.
    """
    if mstore.model is None:
        # Load the pre-trained model dataset
        file_path = '/workspaces/BrightSkyRepoLinux/mesh_sample/grinding_material_removal.csv'
        data = load_data(file_path)
        target_columns = ['RPM', 'Force']

        # Preprocess the data (train the model using the CSV data, for example)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_columns)

        # Train the SVR model
        mstore.model = train_svr(X_train, y_train)
        mstore.scaler = scaler  # Save the scaler for scaling inputs later

        # Optionally, evaluate the model on the test set
        evaluate_model(mstore.model, X_test, y_test)
    else:
        print("Using previously trained model.")


def predict_grind_param(mstore, feed_rate):
    # Prepare inputs for the SVR model (lost volume + feed rate for each section)
    input_data = pd.DataFrame({
        'Feed_Rate': [feed_rate] * len(mstore.grind_params),
        'Lost_Volume': [param['lost_volume'] for param in mstore.grind_params]
    })

    # Scale the input_data using the stored scaler
    input_data_scaled = mstore.scaler.transform(input_data)

    # Predict the RPM and Force using the lost volume and feed rate
    predictions = mstore.model.predict(input_data_scaled)

    # Output predictions with segment and subsection indices
    predicted_rpm_force = pd.DataFrame(predictions, columns=['RPM', 'Force'])
    predicted_rpm_force['Segment'] = [param['segment_idx'] for param in mstore.grind_params]
    predicted_rpm_force['Sub_Section'] = [param['sub_section_idx'] for param in mstore.grind_params]
    
    # Print results with correct segment and subsection designations
    print(predicted_rpm_force[['Segment', 'Sub_Section', 'RPM', 'Force']])

def main():
    print("Processing Meshes and Calculating Lost Volume...")

    mstore = MeshProcessor()
    mvis = MeshVisualizer()
    # Specify the directory to save and load sections
    section_directory = "mesh_sections"

    # Check if sections exist, otherwise process, can change name, both at load and at save command to record more sections
    worn_sections = load_sections("worn", directory=section_directory)
    desired_sections = load_sections("desired", directory=section_directory)


    # TO DO: put this sectioning in a function/method

    if worn_sections and desired_sections:
        mstore.worn_sections = worn_sections
        mstore.desired_sections = desired_sections
        print("Loaded previously saved sections.")
    else:
        print("Processing and saving new sections.")
        # Load meshes
        mstore.load_mesh(1)
        mstore.load_mesh(2)

        # Sample points if loaded trimesh
        if mstore.mesh1_pcl == None:
            mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=60000)
        if mstore.mesh2_pcl == None:
            mstore.mesh2_pcl = mstore.mesh2.sample_points_poisson_disk(number_of_points=60000)
        
        # Get Leading Edge points
        mstore.mesh1_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh1_pcl, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=20, min_distance=20)
        mstore.mesh2_LE_points = mstore.detect_leading_edge_by_curvature(mstore.mesh2_pcl, curvature_threshold=(0.005, 0.04), k_neighbors=50, vicinity_radius=20, min_distance=20)

        # Segment point cloud (flow axis)
        mstore.mesh1_segments = mstore.segment_turbine_pcd(mstore.mesh1_pcl, mstore.mesh1_LE_points)
        mstore.mesh2_segments = mstore.segment_turbine_pcd(mstore.mesh2_pcl, mstore.mesh2_LE_points)

        # Section point cloud (cross section axis)
        mstore.mesh1_sections, bounds = mstore.section_leading_edge_on_segmentedPCL(mstore.mesh1_segments, mstore.mesh1_LE_points, num_sections=3, mid_ratio=0.3)
        mstore.mesh2_sections, _ = mstore.section_leading_edge_on_segmentedPCL(mstore.mesh2_segments, mstore.mesh2_LE_points, num_sections=3, mid_ratio=0.3, use_bounds=bounds)

        mstore.grind_params = []

        for segment_idx, (worn_segment, desired_segment) in enumerate(zip(mstore.mesh1_sections, mstore.mesh2_sections), 1):
            worn_sub_sections = worn_segment['sub_sections']
            desired_sub_sections = desired_segment['sub_sections']

            for sub_section_idx, (worn_sub_section, desired_sub_section) in enumerate(zip(worn_sub_sections, desired_sub_sections), 1):
                # Convert each sub-section point cloud to mesh using Ball-Pivoting Algorithm (BPA) or alpha shapes
                worn_section_mesh = mstore.create_mesh_from_pcl(worn_sub_section)
                desired_section_mesh = mstore.create_mesh_from_pcl(desired_sub_section)

                lost_volume = mstore.calculate_lost_volume(worn_section_mesh, desired_section_mesh, worn_sub_section, desired_sub_section)
                mstore.lost_volumes.append(lost_volume)
                mstore.worn_mesh_sections.append(worn_section_mesh)
                mstore.desired_mesh_sections.append(desired_section_mesh)

                '''
                # Save the converted meshes to disk (in case need repetitive testing)
                worn_mesh_filename = os.path.join(mesh_directory, f"worn_segment_{segment_idx}_subsection_{sub_section_idx}.ply")
                desired_mesh_filename = os.path.join(mesh_directory, f"desired_segment_{segment_idx}_subsection_{sub_section_idx}.ply")

                o3d.io.write_triangle_mesh(worn_mesh_filename, worn_mesh)
                o3d.io.write_triangle_mesh(desired_mesh_filename, desired_mesh)
                '''
                
                mstore.grind_params.append({
                    'segment_idx': segment_idx,
                    'sub_section_idx': sub_section_idx,
                    'lost_volume': lost_volume
                })
                
                print(f"Lost Volume for Segment {segment_idx}, Sub-section {sub_section_idx}: {lost_volume:.2f} mm^3")
        


    mvis.visualize_meshes_overlay(mstore.worn_mesh_sections, mstore.desired_mesh_sections)
    #visualize_lost_material(mstore.worn_mesh_sections, mstore.desired_mesh_sections)

        
    # Create model
    # TO DO: Store model somewhere so we dont need to train it everytime we run this code
    create_grind_model(mstore)

    # Predict RPM and Force
    # TO DO: think about how should we do this. What are the inputs (lost volume, curvature, or more?)
    #                                           What are the outputs (RPM, Force, Feed rate, or fix some variables?)

    # Fixed feed rate
    feed_rate = 20
    predict_grind_param(mstore, feed_rate)


if __name__ == "__main__":
    main()





