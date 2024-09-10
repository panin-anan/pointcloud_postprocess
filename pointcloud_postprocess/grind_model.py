#grinding model based on input worn blade point cloud and desired LE mesh

# From recontour_LE, obtain desired LE mesh. Then this code use the scanned pcl and recontoured pcl
# to calculate grinding parameters of 3 angle of grinding to obtain LE shape based on volume difference. 
# FOR NOW: default as 3 angle of grinding: left right and middle.

from mesh_processor import MeshProcessor
from grindparam_predictor import load_data, preprocess_data, train_svr, evaluate_model
from visualization import visualize_mesh, visualize_meshes_overlay, visualize_sub_section, project_worn_to_desired, visualize_lost_material

import os
import open3d as o3d
import pandas as pd


def save_sections(sections, prefix, directory="saved_sections"):
    """
    Save mesh or point cloud sections to files in a designated directory.
    :param sections: List of sections to save.
    :param prefix: Prefix for the filenames (e.g., "worn" or "desired").
    :param directory: Directory where the files will be saved.
    """
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
    """
    Load mesh or point cloud sections from files in a designated directory.
    :param prefix: Prefix for the filenames (e.g., "worn" or "desired").
    :param num_sections: Number of sections to load.
    :param directory: Directory where the files are stored.
    :return: List of loaded sections.
    """
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
        'Feed_Rate': [feed_rate] * len(mstore.lost_volumes),
        'Lost_Volume': mstore.lost_volumes
    })

    # Scale the input_data using the stored scaler
    input_data_scaled = mstore.scaler.transform(input_data)

    # Predict the RPM and Force using the lost volume and feed rate
    predictions = mstore.model.predict(input_data_scaled)

    # Output predictions
    predicted_rpm_force = pd.DataFrame(predictions, columns=['RPM', 'Force'])
    print(predicted_rpm_force)

def main():
    print("Processing Meshes and Calculating Lost Volume...")

    mstore = MeshProcessor()

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
        mstore.mesh1 = mstore.load_mesh(1)
        mstore.mesh2 = mstore.load_mesh(2)

        # Sample points
        mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=25000)
        mstore.mesh2_pcl = mstore.mesh2.sample_points_poisson_disk(number_of_points=25000)

        # Segment meshes
        mstore.worn_sections, mstore.y_bounds = mstore.segment_leading_edge_by_y_distance(mstore.mesh1_pcl, num_segments=3, mid_ratio=0.7)
        mstore.desired_sections, _ = mstore.segment_leading_edge_by_y_distance(mstore.mesh2_pcl, num_segments=3, use_bounds=mstore.y_bounds)

        # Convert segments to meshes if necessary
        mstore.worn_sections = [mstore.create_mesh_from_point_cloud(section) for section in mstore.worn_sections]
        mstore.desired_sections = [mstore.create_mesh_from_point_cloud(section) for section in mstore.desired_sections]

        # Save processed sections to files
        save_sections(mstore.worn_sections, "worn", directory=section_directory)
        save_sections(mstore.desired_sections, "desired", directory=section_directory)


    # Calculate lost volume and visualize
    for i, (worn, desired) in enumerate(zip(mstore.worn_sections, mstore.desired_sections), 1):
        lost_volume = mstore.calculate_lost_volume(worn, desired)
        mstore.lost_volumes.append(lost_volume)
        print(f"Lost Volume for Section {i}: {lost_volume:.2f} mm^3")
    
    visualize_meshes_overlay(mstore.worn_sections, mstore.desired_sections)
    visualize_lost_material(mstore.worn_sections, mstore.desired_sections)



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





