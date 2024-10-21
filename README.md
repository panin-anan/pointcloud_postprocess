# pointcloud_postprocess
# for flat plate samples

## Overview
Python Script version of pcl_processing_ros2. See https://github.com/panin-ananwa/pcl_processing_ros2

Code to perform volume loss calculation between two point clouds: from before grinding and from after grinding, scanned by laser line scanner.

## Installation

#### Dependencies
- open3d
- numpy


```bash
pip install open3d==0.18.0
pip install numpy==1.24.0

```
#### Building
To build from source, clone the latest version from this repository into your workspace

```bash
git clone git@github.com:panin-ananwa/pointcloud_postprocess.git -b flat_plate_pclprocess
```
#### Running Script
the main code is `meshcalc_app.py` in manual_pclprocess folder.
To start the program from your workspace, run on terminal:

```bash
python3 src/pointcloud_postprocess/manual_pclprocess/meshcalc_app.py
```

### Utility scripts
- `mesh_cropper.py`: load point cloud and visualize using open3d. Enable manual cropping point cloud with user input.
- `create_meshfrompcl.py`: load point cloud and generate triangle mesh using alpha shape, poisson reconstruction, or ball pivoting.
