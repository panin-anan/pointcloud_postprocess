import open3d as o3d
import numpy as np
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


def create_flat_plate_mesh_after(length, width, thickness, sink_depth, sink_size, divisions_length=4, divisions_width=4, filename="mesh.ply"):
    # Generate vertices in a grid
    vertices = []
    for i in range(divisions_length + 1):
        for j in range(divisions_width + 1):
            x = (length / divisions_length) * i
            y = (width / divisions_width) * j
            
            # Determine if the vertex is within the sink region
            if (length/2 - sink_size/2) <= x <= (length/2 + sink_size/2) and (width/2 - sink_size/2) <= y <= (width/2 + sink_size/2):
                z = thickness - sink_depth
            else:
                z = thickness
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate triangles based on the grid of vertices
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


def create_LE_mesh_before(length, width, height, curvature, divisions_length=4, divisions_width=4, filename="mesh.ply"):
    # Generate vertices in a grid
    vertices = []
    for i in range(divisions_length + 1):
        for j in range(divisions_width + 1):
            x = (length / divisions_length) * i
            y = (width / divisions_width) * j
            # Apply curvature to the height
            if curvature > 0:
                z = height * curvature * (np.sin(np.pi * x / length) * np.sin(np.pi * y / width))
            else:
                z = height
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate triangles based on the grid of vertices
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

# Parameters in mm
length = 50.0
width = 10.0
height = 9.8           ## for LE edge mesh
thickness = 4.0 
sink_depth = 3.0
sink_size = 20.0

curvature = 2

divisions_length = 10
divisions_width = 5

#mesh_before = create_flat_plate_mesh_before(length, width, thickness, divisions_length, divisions_width, filename="flat_plate_fine_mesh.ply")
#mesh_after = create_flat_plate_mesh_after(length, width, thickness, sink_depth, sink_size, divisions_length, divisions_width, filename="grinded_plate_fine_mesh.ply")
mesh_LE = create_LE_mesh_before(length, width, height, curvature, divisions_length, divisions_width, filename="LE_mesh_grinded.ply")

o3d.visualization.draw_geometries([mesh_LE], window_name="Fine Mesh", width=800, height=600)


