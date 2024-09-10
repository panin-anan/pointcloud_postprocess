import numpy as np
import open3d as o3d
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev


from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl


from mesh_processor import MeshProcessor
from visualization import visualize_meshes_overlay, visualize_section_pcl

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


def find_directional_curve(pcd, num_sections=12, flow_axis='z', swap_LE_axes=False):
    """
    Generate the directional curve based on local maxima in the leading edge direction for each section.
    Divides the model along the specified flow axis and finds the leading edge (LE) point by finding 
    the maximum value perpendicular to the flow axis. The coordinate perpendicular to both axes 
    is averaged over all points in each section.
    
    Parameters:
    - pcd: Input point cloud (open3d.geometry.PointCloud)
    - num_sections: Number of sections to divide along the flow axis
    - flow_axis: The axis representing the flow direction ('x', 'y', or 'z').
    
    Returns:
    - directional_curve: Numpy array of points representing the leading edge (LE) curve.
    """
    points = np.asarray(pcd.points)

    # Map the flow axis to the appropriate index in the point array
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if flow_axis not in axis_map:
        raise ValueError("Invalid flow axis. Must be 'x', 'y', or 'z'.")
    
    # Get axis indices based on the flow axis
    flow_idx = axis_map[flow_axis]
    le_idx = (flow_idx + 1) % 3  # The leading edge axis (perpendicular to the flow axis)
    avg_idx = (flow_idx + 2) % 3  # The axis to average (perpendicular to both)

    # Swap leading edge and averaging axes if the swap_axes flag is True
    if swap_LE_axes:
        le_idx, avg_idx = avg_idx, le_idx

    idx_to_axis = {0: 'x', 1: 'y', 2: 'z'}
    print(f"Flow axis is: {idx_to_axis[flow_idx]}")
    print(f"Leading edge axis is: {idx_to_axis[le_idx]}")
    print(f"Averaging axis is: {idx_to_axis[avg_idx]}")

    # Find the min and max along the flow axis
    flow_min = np.min(points[:, flow_idx])
    flow_max = np.max(points[:, flow_idx])

    # Create equally spaced values to divide the model into sections along the flow axis
    flow_sections = np.linspace(flow_min, flow_max, num_sections)

    le_points = []  # Leading edge points

    # Iterate over each section to find the LE point (local maximum in the leading edge direction)
    for i in range(num_sections - 1):
        flow_lower = flow_sections[i]
        flow_upper = flow_sections[i + 1]

        # Get the points that fall within the current flow axis section
        section_mask = (points[:, flow_idx] >= flow_lower) & (points[:, flow_idx] < flow_upper)
        section_points = points[section_mask]

        if len(section_points) > 0:
            # Find the point with the maximum value in the leading edge axis
            max_le_idx = np.argmax(section_points[:, le_idx])
            le_point = section_points[max_le_idx]

            # Calculate the average position of points in the axis perpendicular to both flow and LE
            avg_value = np.mean(section_points[:, avg_idx])

            # Replace the value in the perpendicular axis with the average
            le_point[avg_idx] = avg_value

            # Append the modified leading edge point to the list
            le_points.append(le_point)

    # Convert the list of leading edge points to a numpy array
    directional_curve = np.array(le_points)

    return directional_curve

def adjust_worn_LE_radius(section, le_center, target_radius, flow_axis="z"):
    """
    Reshape a section of points to fit a leading edge profile.
    Axis-independent reshaping, adjusts points outside the target radius.
    
    Parameters:
    - section: Input 2D array of points in the section (Nx3, where N is the number of points).
    - le_center: Leading edge center (calculated with `find_leading_edge_center_section`).
    - target_radius: The target radius for the leading edge.
    - flow_axis: The axis that defines the flow direction ('x', 'y', or 'z').
    
    Returns:
    - transformed_section: Reshaped section of points.
    """
    # Axis-independent: Determine which two axes to use for reshaping
    if flow_axis == "z":
        idx_le, idx_y, idx_z = 0, 1, 2  # X, Y are reshaped, Z is flow
    elif flow_axis == "x":
        idx_le, idx_y, idx_z = 1, 2, 0  # Y, Z are reshaped, X is flow
    elif flow_axis == "y":
        idx_le, idx_y, idx_z = 0, 2, 1  # X, Z are reshaped, Y is flow
    else:
        raise ValueError("Invalid flow axis. Must be 'x', 'y', or 'z'.")

    transformed_points = []
    for point in section:
        le_value, y_value, z_value = point[idx_le], point[idx_y], point[idx_z]

        # Shift the point to the LE center in the two reshaping axes
        shift_le = le_value - le_center[idx_le]
        shift_y = y_value - le_center[idx_y]

        # Calculate the distance from the leading edge center in the reshaped plane
        r = np.sqrt(shift_le**2 + shift_y**2)  # Distance from the LE center in the reshaped plane
        
        # Transform only if the point lies outside the desired target radius
        if r >= target_radius and shift_le > 0:  # Positive side of leading edge in LE axis
            scaling_factor = target_radius / r
            le_new = le_center[idx_le] + shift_le * scaling_factor  # Scale and shift back
            y_new = le_center[idx_y] + shift_y * scaling_factor  # Scale and shift back
        else:
            # Keep the points within the target radius unchanged
            le_new = le_value
            y_new = y_value

        # Append the transformed point (Z stays the same, flow axis stays unchanged)
        transformed_point = np.zeros(3)
        transformed_point[idx_le] = le_new
        transformed_point[idx_y] = y_new
        transformed_point[idx_z] = z_value
        transformed_points.append(transformed_point)

    transformed_section = np.array(transformed_points)

    return transformed_section


def create_planes_along_directional_curve(directional_curve, num_planes=12):
    """
    Create cutting planes equidistant along the directional curve.
    The planes' normal will be aligned with the tangent to the curve.
    """
    planes = []
    for i in range(1, len(directional_curve)):
        # Calculate the normal based on the difference between adjacent points on the curve
        normal = directional_curve[i] - directional_curve[i - 1]
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        
        # Store plane as a tuple (point on plane, normal vector)
        planes.append((directional_curve[i], normal))
        
    return planes


def cut_point_cloud_with_planes(pcd, planes):
    """
    Cut the point cloud with planes to get intersection curves.
    """
    sections = []
    for plane in planes:
        point_on_plane, normal = plane
        # Define the plane as the equation ax + by + cz + d = 0
        d = -np.dot(normal, point_on_plane)
        
        # Project the points onto the plane
        points = np.asarray(pcd.points)
        distances = np.dot(points, normal) + d
        
        # Keep points that are near the plane (within a small threshold)
        threshold = 0.4
        mask = np.abs(distances) < threshold
        section_points = points[mask]
        
        if section_points.shape[0] > 0:
            sections.append(section_points)
    
    return sections

def adjust_worn_LE_center(sections, target_radius, le_axis='Z'):
    """
    Find the center of the leading edge based on a section of the point cloud.
    Assumes that the leading edge has the largest value along the axis perpendicular to the flow.
    
    Parameters:
    - section: 2D array of points in the section (Nx3, where N is the number of points).
    - target_radius: The desired leading edge radius.
    - flow_axis: The axis that defines the flow direction ('x', 'y', or 'z').
    
    Returns:
    - le_center: The leading edge center coordinates (x, y, z).
    """
    # Axis-independent: Determine which axis to look for the maximum value
    if flow_axis == "z":
        idx_le, idx_y, idx_z = 0, 1, 2  # Max leading edge will be in X direction
    elif flow_axis == "x":
        idx_le, idx_y, idx_z = 1, 2, 0  # Max leading edge will be in Y direction
    elif flow_axis == "y":
        idx_le, idx_y, idx_z = 0, 2, 1  # Max leading edge will be in X direction
    else:
        raise ValueError("Invalid flow axis. Must be 'x', 'y', or 'z'.")

    # Find the maximum leading edge coordinate (along the LE axis)
    max_le_value = np.max(section[:, idx_le])
    
    # Select points within the target radius distance from the maximum LE value
    leading_edge_points = section[section[:, idx_le] >= max_le_value - target_radius]

    # Compute the center of the leading edge based on the other two axes (excluding flow axis)
    le_center_le = max_le_value - target_radius
    le_center_other1 = np.mean(leading_edge_points[:, idx_y])
    le_center_other2 = np.mean(leading_edge_points[:, idx_z])

    # Create the center in (x, y, z) format, ensuring correct axis alignment
    le_center = np.zeros(3)
    le_center[idx_le] = le_center_le
    le_center[idx_y] = le_center_other1
    le_center[idx_z] = le_center_other2
    
    return le_center


def filter_pcl_by_thresholds(pcd, min_x=None, min_y=None, min_z=None):
    """
    Filter points from the point cloud based on specified minimum values for the x, y, and z coordinates.

    Parameters:
    - pcd: Input point cloud (open3d.geometry.PointCloud)
    - min_x: Minimum value for the x-coordinate. If None, no filtering is done on x-axis.
    - min_y: Minimum value for the y-coordinate. If None, no filtering is done on y-axis.
    - min_z: Minimum value for the z-coordinate. If None, no filtering is done on z-axis.

    Returns:
    - filtered_pcd: A new point cloud containing only the points that meet the threshold criteria.
    """
    points = np.asarray(pcd.points)

    # Create a mask that will filter the points based on the thresholds
    mask = np.ones(len(points), dtype=bool)  # Start with a mask of all True values

    if min_x is not None:
        mask &= points[:, 0] >= min_x  # Apply filter on x-axis
    if min_y is not None:
        mask &= points[:, 1] >= min_y  # Apply filter on y-axis
    if min_z is not None:
        mask &= points[:, 2] >= min_z  # Apply filter on z-axis

    # Apply the mask to the points
    filtered_points = points[mask]

    # Create a new point cloud from the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd

def smooth_sections(sections):
    """
    Perform smoothing of the sections using spline interpolation to ensure smooth transitions
    between sections.

    Parameters:
    - sections: List of sections (each section is an array of points).

    Returns:
    - smoothed_sections: List of smoothed sections.
    """
    smoothed_sections = []

    for section in sections:
        # Perform spline interpolation on each section to smooth it
        tck, u = splprep(section.T, s=0)
        u_fine = np.linspace(0, 1, len(section))
        smoothed_section = np.array(splev(u_fine, tck)).T

        smoothed_sections.append(smoothed_section)

    return smoothed_sections







# Main
# Load mesh to mesh processor
mstore = MeshProcessor()
mstore.mesh1 = mstore.load_mesh(1)

print("Mesh Loaded")
# Sample points
mstore.mesh1_pcl = mstore.mesh1.sample_points_poisson_disk(number_of_points=40000)

mstore.mesh1_pcl = filter_pcl_by_thresholds(mstore.mesh1_pcl, min_x=20)


# Step 2: Generate directional curve based on the point cloud's max/min points
directional_curve = find_directional_curve(mstore.mesh1_pcl, num_sections=10, flow_axis = 'y', swap_LE_axes=True)

# Step 3: Create planes along the directional curve
planes = create_planes_along_directional_curve(directional_curve, num_planes=10)

# Step 4: Cut the point cloud using planes to get sectional curves
sections = cut_point_cloud_with_planes(mstore.mesh1_pcl, planes)

visualize_meshes_overlay(mstore.mesh1_pcl, directional_curve=directional_curve, planes=planes, line_width=5.0)
visualize_section_pcl(sections)


'''
# Segment meshes
mstore.worn_sections, mstore.y_bounds = mstore.segment_leading_edge_by_y_distance(mstore.mesh1_pcl, num_segments=3, mid_ratio=0.7)
# Convert segments to meshes if necessary
mstore.worn_sections = [mstore.create_mesh_from_point_cloud(section) for section in mstore.worn_sections]

visualize_meshes_overlay(mstore.worn_sections)
'''


# Step 5: Adjust leading edge radius for each section
ideal_le_radius = 1.0
flow_axis = 'y'

adjusted_sections = []
for section in sections:
    le_center = adjust_worn_LE_center(section, ideal_le_radius, flow_axis)
    adjusted_section = adjust_worn_LE_radius(section, le_center, ideal_le_radius, flow_axis)
    adjusted_sections.append(adjusted_section)


'''
# Visualize each section
for i, adjusted_section in enumerate(adjusted_sections):
    adjusted_pcd = o3d.geometry.PointCloud()
    adjusted_pcd.points = o3d.utility.Vector3dVector(adjusted_section)
    o3d.visualization.draw_geometries([adjusted_pcd])
'''

# Step 6: Smooth the adjusted sections for a more consistent shape
#smoothed_sections = smooth_sections(adjusted_sections)

# Visualize the final smoothed sections
visualize_section_pcl(adjusted_sections)
