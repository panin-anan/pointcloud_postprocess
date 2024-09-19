import open3d as o3d
import numpy as np

def convert_to_pointcloud(input_data):
    """
    Convert input data to an Open3D PointCloud
    """
    if isinstance(input_data, o3d.geometry.PointCloud):
        return input_data
    elif isinstance(input_data, np.ndarray):
        # If the input is a numpy array, ensure it has the right shape
        if len(input_data.shape) == 3:  # Array of ndarrays
            if input_data.shape[-1] != 3:
                raise ValueError(f"Each ndarray must have shape Nx3, but got shape {input_data.shape}")
            # Concatenate all ndarrays along the first axis
            input_data = np.concatenate(input_data, axis=0)
        elif input_data.shape[1] != 3:
            raise ValueError(f"Input numpy array must have shape Nx3, but got shape {input_data.shape}")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(input_data)
        return point_cloud
    elif isinstance(input_data, list):
        # Handle list of ndarrays
        if all(isinstance(arr, np.ndarray) for arr in input_data):
            concatenated_points = np.concatenate(input_data, axis=0)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(concatenated_points)
            return point_cloud
        else:
            raise ValueError("All elements in the list must be numpy ndarrays.")
    else:
        raise TypeError(f"Unsupported data type for conversion to PointCloud: {type(input_data)}")


def visualize_mesh(mesh):
    o3d.visualization.draw_geometries(mesh, mesh_show_back_face=True)

def visualize_pcl_overlay(pcl_1, pcl_2):
    vis_elements = []

    # Convert to point cloud if necessary
    pcl_1 = convert_to_pointcloud(pcl_1)
    pcl_2 = convert_to_pointcloud(pcl_2)

    pcl_1.paint_uniform_color([1, 0, 0])  # Red for the first point cloud
    vis_elements.append(pcl_1)
    
    pcl_2.paint_uniform_color([0, 1, 0])  # Green for the second point cloud
    vis_elements.append(pcl_2)

    o3d.visualization.draw_geometries(vis_elements, window_name="PCL Overlay", width=800, height=600)

def visualize_meshes_overlay(worn_meshes=None, desired_meshes=None, directional_curve=None, planes=None, line_width=15.0):
    geometries = []

    # Convert single mesh input to a list for consistency
    if not isinstance(worn_meshes, list):
        worn_meshes = [worn_meshes]
    
    geometries += worn_meshes
    
    # If desired_meshes is provided, add them to geometries
    if desired_meshes is not None:
        if not isinstance(desired_meshes, list):
            desired_meshes = [desired_meshes]
        geometries += desired_meshes

    # Add directional curve if provided
    if directional_curve is not None:
        # Create a line set for the directional curve
        curve_lines = []
        num_points = len(directional_curve)
        for i in range(num_points - 1):
            curve_lines.append([i, i + 1])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(directional_curve)
        line_set.lines = o3d.utility.Vector2iVector(curve_lines)
        line_set.paint_uniform_color([0, 0, 0])  # Green color for the curve
        
        geometries.append(line_set)

    if planes is not None:
        for plane in planes:
            point_on_plane, normal = plane
            
            # Create a small plane using create_plane function
            plane_mesh = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=0.1)
            
            # Translate plane to point_on_plane
            plane_mesh.translate(point_on_plane)
            
            # Align the plane's normal with the provided normal
            # Open3D's default normal for create_plane is along the +Z axis, so we need a rotation
            default_normal = np.array([0, 0, 1])  # Z axis
            rotation_axis = np.cross(default_normal, normal)  # Rotation axis to align with the new normal
            rotation_angle = np.arccos(np.dot(default_normal, normal) / (np.linalg.norm(default_normal) * np.linalg.norm(normal)))  # Angle between normals
            
            if np.linalg.norm(rotation_axis) > 1e-6:  # If there's a significant rotation needed
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the axis
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                plane_mesh.rotate(rotation_matrix, center=point_on_plane)
            
            plane_mesh.paint_uniform_color([1, 0, 0])  # Red color for the plane
            
            geometries.append(plane_mesh)

    # Create a custom visualization window to adjust line thickness
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add all geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Ensure the visualizer has been properly initialized before getting render options
    vis.poll_events()
    vis.update_renderer()

    # Set custom rendering options for line width
    render_option = vis.get_render_option()
    render_option.line_width = line_width  # Set the line thickness

    # Visualize the meshes, directional curve, and planes
    vis.run()
    vis.destroy_window()

def visualize_sub_section(sub_section):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors for each sub-curve
    geometries = []

    for idx, section in enumerate(sub_section):
        curve_pc = o3d.geometry.PointCloud()
        curve_pc.points = o3d.utility.Vector3dVector(np.asarray(section.points))
        curve_pc.paint_uniform_color(colors[idx % len(colors)])
        geometries.append(curve_pc)

    o3d.visualization.draw_geometries(geometries)


def project_worn_to_desired(worn_mesh, desired_mesh):
    """
    Project the worn mesh points onto the desired mesh and visualize the lost material.
    """
    worn_pcd = worn_mesh.sample_points_uniformly(number_of_points=10000)
    desired_pcd = desired_mesh.sample_points_uniformly(number_of_points=10000)

    distances = np.asarray(worn_pcd.compute_point_cloud_distance(desired_pcd))
    p2p_distances = o3d.geometry.KDTreeFlann(desired_pcd)

    worn_points = np.asarray(worn_pcd.points)
    projected_points = []

    for point in worn_points:
        _, idx, _ = p2p_distances.search_knn_vector_3d(point, 1)
        projected_points.append(desired_pcd.points[idx[0]])

    projected_points = np.asarray(projected_points)
    combined_points = np.vstack((worn_points, projected_points))

    lines = [[i, i + len(worn_points)] for i in range(len(worn_points))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(combined_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    max_distance = np.max(distances)
    colors = [[d / max_distance, 0, 1 - d / max_distance] for d in distances]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def visualize_lost_material(worn_meshes, desired_meshes):
    for i in range(len(worn_meshes)):
        lost_visualization = project_worn_to_desired(worn_meshes[i], desired_meshes[i])
        o3d.visualization.draw_geometries([worn_meshes[i], desired_meshes[i], lost_visualization])

def visualize_section_pcl(sections):
    geometries = []

    # Create point cloud geometries for each section and assign different colors for visualization
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]

    for i, section in enumerate(sections):
        if len(section) > 0:
            section_pcd = o3d.geometry.PointCloud()
            section_pcd.points = o3d.utility.Vector3dVector(section)

            # Assign a unique color to each section (cycling through the colors list)
            color = colors[i % len(colors)]
            section_pcd.paint_uniform_color(color)

            geometries.append(section_pcd)

    # Visualize the sections
    o3d.visualization.draw_geometries(geometries, window_name="Sectioned Point Clouds")

