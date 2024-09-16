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

#create point cloud from trimesh
mesh_before = o3d.io.read_triangle_mesh(mesh_before_path)
mesh_before_pcl = mesh_before.sample_points_poisson_disk(number_of_points=40000)

#Load point cloud directly
#mesh_before_pcl = o3d.io.read_point_cloud(mesh_before_path)


'''
# Downsample the point cloud
voxel_size = 0.0005 
mesh_before_pcl = mesh_before_pcl.voxel_down_sample(voxel_size)
'''


#mesh_before_pcl = mesh_before.sample_points_uniformly(number_of_points=100000)
mesh_before_pcl.estimate_normals()
#mesh_before_pcl.orient_normals_consistent_tangent_plane(100)

print("Estimated")




'''
#Poisson
mesh_before_trimesh_poisson = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh_before_pcl, depth=8, width=0, scale=1.1, linear_fit=False)[0]
bbox = mesh_before_pcl.get_axis_aligned_bounding_box()
mesh_before_trimesh_poisson_cropped = mesh_before_trimesh_poisson.crop(bbox)
mesh_before_trimesh_poisson_cropped.scale(0.1, center=mesh_before_trimesh_poisson_cropped.get_center())  # Example: scaling down the mesh
o3d.visualization.draw_geometries([mesh_before_pcl], window_name="Fine Mesh", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_poisson], window_name="Fine Mesh", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_poisson_cropped], window_name="Fine Mesh", width=800, height=600)
'''


'''
# Alpha Shape Reconstruction
alpha = 0.01
mesh_before_trimesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(mesh_before_pcl, alpha)

# Bounding box cropping and scaling
bbox = mesh_before_pcl.get_axis_aligned_bounding_box()
mesh_before_trimesh_alpha_cropped = mesh_before_trimesh_alpha.crop(bbox)
mesh_before_trimesh_alpha_cropped.scale(0.1, center=mesh_before_trimesh_alpha_cropped.get_center())  # Example: scaling down the mesh

# Visualization
mesh_before_trimesh_alpha.scale(0.1, center=mesh_before_trimesh_alpha.get_center())
o3d.visualization.draw_geometries([mesh_before_pcl], window_name="Point Cloud", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_alpha], window_name="Alpha Shape Mesh", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_trimesh_alpha_cropped], window_name="Cropped Alpha Shape Mesh", width=800, height=600)
'''

def joggle_points(pcd, scale=1e-6):
    points = np.asarray(pcd.points)
    jitter = np.random.normal(scale=scale, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + jitter)

def create_mesh_from_point_cloud(pcd):
    joggle_points(pcd) 
    pcd.estimate_normals()

    pcd.orient_normals_consistent_tangent_plane(30)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [0.1 * avg_dist, 0.4 * avg_dist, 0.7 * avg_dist, 1 * avg_dist, 1.5 * avg_dist, 2 * avg_dist, 3 * avg_dist] #can reduce to reduce computation
    r = o3d.utility.DoubleVector(radii)
    
    #ball pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, r)

    return mesh

#Ball pivoting
mesh_before_ball_pivoting = create_mesh_from_point_cloud(mesh_before_pcl)
o3d.visualization.draw_geometries([mesh_before_pcl], window_name="Point Cloud", width=800, height=600)
o3d.visualization.draw_geometries([mesh_before_ball_pivoting], window_name="TriMesh", width=800, height=600)

o3d.io.write_triangle_mesh("Grinded.ply", mesh_before_ball_pivoting)