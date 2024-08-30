import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import cKDTree, Delaunay

# Define flat plate mesh for testing

def create_flat_plate_mesh_before(length, width, thickness, divisions_length, divisions_width, filename="mesh.ply"):
    # Generate vertices in a grid
    vertices = []
    for i in range(divisions_length + 1):
        for j in range(divisions_width + 1):
            x = (length / divisions_length) * i
            y = (width / divisions_width) * j
            vertices.append([x, y, thickness])

    vertices = np.array(vertices)

    #triangles
    triangles = []
    for i in range(divisions_length):
        for j in range(divisions_width):
            v0 = i * (divisions_width + 1) + j
            v1 = v0 + 1
            v2 = v0 + (divisions_width + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles)


    # Create a TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Optionally compute normals for better visualization
    mesh.compute_vertex_normals()

    # Save the mesh to a file
    o3d.io.write_triangle_mesh(filename, mesh)
    
    print(f"Mesh saved to {filename}")

    return mesh


# Initialize tkinter
root = tk.Tk()
root.withdraw()

# Prompt user to select the mesh files
mesh_before_path = filedialog.askopenfilename(title="Select the mesh file before grinding",
                                             filetypes=[("PLY files", "*.ply")])

#Load mesh file: Before grinding and after grinding
mesh_before = o3d.io.read_triangle_mesh(mesh_before_path)

mesh_before_pcl = mesh_before.sample_points_uniformly(number_of_points=10000)
mesh_before_pcl.estimate_normals()
mesh_before_pcl.orient_normals_consistent_tangent_plane(100)

mesh_before_trimesh_poisson = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh_before_pcl, depth=8, width=0, scale=1.1, linear_fit=False)[0]

bbox = mesh_before_pcl.get_axis_aligned_bounding_box()
mesh_before_trimesh_poisson_cropped = mesh_before_trimesh_poisson.crop(bbox)

o3d.visualization.draw_geometries([mesh_before_pcl], window_name="Fine Mesh", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_poisson], window_name="Fine Mesh", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_poisson_cropped], window_name="Fine Mesh", width=800, height=600)
