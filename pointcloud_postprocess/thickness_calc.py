import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import cKDTree, Delaunay

def calculate_lost_volume(mesh_before, mesh_after):
    # Ensure the meshes have normals computed
    mesh_before.compute_vertex_normals()
    mesh_after.compute_vertex_normals()

    # Sample points uniformly.
    pcd_before = mesh_before.sample_points_uniformly(number_of_points=100000)
    pcd_after = mesh_after.sample_points_uniformly(number_of_points=100000)

    # Compute the distance from the "before" point cloud to the "after" mesh surface
    distances = pcd_after.compute_point_cloud_distance(pcd_before)
    distances = np.asarray(distances)

    # Compute lost material volume (currently assume same area, before and after)
    # TO DO: account for change in area
    reference_area = mesh_after.get_surface_area()
    # print(f"reference area is:{reference_area}")
    volume_lost = np.mean(distances) * reference_area

    return volume_lost


def filter_unchangedpointson_mesh(mesh_before, mesh_after, threshold=0.1):
    # Convert mesh vertices to point clouds (numpy arrays)
    points_before = np.asarray(mesh_before.vertices)
    points_after = np.asarray(mesh_after.vertices)

    # Build KD-Tree for the points in mesh_before
    kdtree_before = cKDTree(points_before)

    # Find the nearest neighbor in mesh_before for each point in mesh_after
    distances, indices = kdtree_before.query(points_after)

    # Filter points based on the distance threshold
    unchanged_indices = np.where(distances < threshold)[0]
    changed_indices = np.where(distances >= threshold)[0]

    # Create new sets of points for unchanged and changed vertices
    unchanged_vertices = points_after[unchanged_indices]
    changed_vertices = points_after[changed_indices]

    # Perform Delaunay triangulation on the filtered points
    if len(unchanged_vertices) >= 3:  # Need at least 3 points to form a triangle
        delaunay_unchanged = Delaunay(unchanged_vertices[:, :2])
        unchanged_triangles = delaunay_unchanged.simplices
    else:
        unchanged_triangles = []

    if len(changed_vertices) >= 3:
        delaunay_changed = Delaunay(changed_vertices[:, :2])
        changed_triangles = delaunay_changed.simplices
    else:
        changed_triangles = []

    # Create new Open3D mesh objects for unchanged and changed parts
    mesh_unchanged = o3d.geometry.TriangleMesh()
    mesh_unchanged.vertices = o3d.utility.Vector3dVector(unchanged_vertices)
    mesh_unchanged.triangles = o3d.utility.Vector3iVector(unchanged_triangles)

    mesh_changed = o3d.geometry.TriangleMesh()
    mesh_changed.vertices = o3d.utility.Vector3dVector(changed_vertices)
    mesh_changed.triangles = o3d.utility.Vector3iVector(changed_triangles)

    return mesh_unchanged, mesh_changed




def calculate_lost_thickness(mesh_before, changed_mesh_after, lost_volume):
    reference_area = changed_mesh_after.get_surface_area()
    lost_thickness = lost_volume / reference_area
    
    return lost_thickness


def compute_average_x(mesh):
    vertices = np.asarray(mesh.vertices)
    average_x = np.mean(vertices[:, 0])
    return average_x

def compute_average_y(mesh):
    vertices = np.asarray(mesh.vertices)
    average_y = np.mean(vertices[:, 1])
    return average_y

def compute_average_z(mesh):
    vertices = np.asarray(mesh.vertices)
    average_z = np.mean(vertices[:, 2])
    return average_z


def calculate_curvature(mesh):
    # Ensure the mesh has vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Get vertices and triangles from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Initialize Laplace-Beltrami operator and area weight arrays
    L = np.zeros(vertices.shape)
    area_weight = np.zeros(vertices.shape[0])

    for tri in triangles:
        # Extract the indices of the vertices in the triangle
        i1, i2, i3 = tri

        # Get the positions of the vertices
        v1 = vertices[i1]
        v2 = vertices[i2]
        v3 = vertices[i3]

        # Compute the edges of the triangle
        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3

        # Compute the angles at each vertex
        angle1 = np.arccos(np.dot(e1, -e3) / (np.linalg.norm(e1) * np.linalg.norm(e3)))
        angle2 = np.arccos(np.dot(e2, -e1) / (np.linalg.norm(e2) * np.linalg.norm(e1)))
        angle3 = np.pi - angle1 - angle2

        # Compute the cotangents of the angles
        cot1 = 1 / np.tan(angle1)
        cot2 = 1 / np.tan(angle2)
        cot3 = 1 / np.tan(angle3)

        # Update the Laplacian at each vertex
        L[i1] += cot3 * (v3 - v2) + cot2 * (v2 - v3)
        L[i2] += cot1 * (v1 - v3) + cot3 * (v3 - v1)
        L[i3] += cot2 * (v2 - v1) + cot1 * (v1 - v2)

        # Update the area weights for each vertex
        area = np.linalg.norm(np.cross(e1, -e3)) / 2
        area_weight[i1] += area
        area_weight[i2] += area
        area_weight[i3] += area

    # Normalize the Laplace-Beltrami operator by the area weights
    mean_curvature = np.linalg.norm(L, axis=1) / (2 * area_weight)  #array of curvature of each triangle
    overall_curvature = np.mean(np.abs(mean_curvature))             #curvature of the entire surface mesh

    return mean_curvature, overall_curvature


def calculate_change_in_curvature(mesh_before, mesh_after):
    # Compute curvature
    mean_curvature_before = calculate_curvature(mesh_before)
    mean_curvature_after = calculate_curvature(mesh_after)
    mean_curvature_change = np.abs(mean_curvature_before - mean_curvature_after)


    return mean_curvature_change


def create_visualizer(mesh, window_name, width, height, x, y):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, left=x, top=y)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    return vis


def visualize_curvature(mesh, curvature):
    # Normalize curvature for color mapping
    normalized_curvature = curvature / curvature.max()

    # Map the curvature to a color range (e.g., from blue to red)
    colors = np.zeros((len(curvature), 3))
    colors[:, 0] = normalized_curvature  # Red channel
    colors[:, 2] = 1 - normalized_curvature  # Blue channel

    # Assign the colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Visualize the mesh with curvature-based coloring
    o3d.visualization.draw_geometries([mesh], window_name="Curvature Visualization", width=800, height=600)

def main():
    
    # Initialize tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Prompt user to select the mesh files
    mesh_before_path = filedialog.askopenfilename(title="Select the mesh file before grinding",
                                                 filetypes=[("PLY files", "*.ply")])
    mesh_after_path = filedialog.askopenfilename(title="Select the mesh file after grinding",
                                                 filetypes=[("PLY files", "*.ply")])
    
    #Load mesh file: Before grinding and after grinding
    mesh_before = o3d.io.read_triangle_mesh(mesh_before_path)
    mesh_after = o3d.io.read_triangle_mesh(mesh_after_path)

    # Filter Mesh
    unchanged_mesh, changed_mesh = filter_unchangedpointson_mesh(mesh_before, mesh_after, threshold=0.1)
    

    # Calculate the lost volume, thickness, and change in curvature
    lost_volume = calculate_lost_volume(mesh_before, changed_mesh)
    lost_thickness = calculate_lost_thickness(mesh_before, changed_mesh, lost_volume)

    avg_before = compute_average_z(mesh_before)
    avg_after = compute_average_z(changed_mesh)
    avg_diff = np.abs(avg_before - avg_after)
    print(f"average Z before grind: {avg_before}")
    print(f"average Z before grind: {avg_after}")
    #change_in_curvature = calculate_change_in_curvature(mesh_before, mesh_after)

    print(f"Estimated volume of lost material: {lost_volume} mm^3")
    print(f"Estimated grinded thickness mesh method: {lost_thickness} mm")
    print(f"Estimated grinded thickness avg method: {avg_diff} mm")
    #print(f"Mean change in curvature: {change_in_curvature}")


    # Open multiple visualizers simultaneously
    vis_before = create_visualizer(mesh_before, "Before Grinding", 800, 600, 50, 50)
    vis_after = create_visualizer(mesh_after, "After Grinding", 800, 600, 900, 50)
    vis_unchanged = create_visualizer(unchanged_mesh, "Unchanged Vertices", 800, 600, 50, 700)
    vis_changed = create_visualizer(changed_mesh, "Changed Vertices", 800, 600, 900, 700)

    # Window Loop
    while vis_before.poll_events() and vis_after.poll_events() and vis_unchanged.poll_events() and vis_changed.poll_events():
        vis_before.update_renderer()
        vis_after.update_renderer()
        vis_unchanged.update_renderer()
        vis_changed.update_renderer()

    # Once loop exits, destroy all windows
    vis_before.destroy_window()
    vis_after.destroy_window()
    vis_unchanged.destroy_window()
    vis_changed.destroy_window()


    mean_curvature_before, overall_curvature_before = calculate_curvature(mesh_before)
    mean_curvature_after, overall_curvature_after = calculate_curvature(mesh_after)

    print(overall_curvature_before)
    print(overall_curvature_after)

    visualize_curvature(mesh_after, mean_curvature_after)


if __name__ == '__main__':
    main()