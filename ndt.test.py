import pcl
import numpy as np
import open3d as o3d

# Load point clouds
target_cloud = pcl.PointCloud.PointXYZ()
input_cloud = pcl.PointCloud.PointXYZ()

pcl.io.loadPCDFile("NDT_PCL/room_scan1.pcd", target_cloud)
pcl.io.loadPCDFile("NDT_PCL/room_scan2.pcd", input_cloud)

# Apply Voxel Grid filter to downsample the input cloud
voxel_filter = pcl.filters.VoxelGrid.PointXYZ()
voxel_filter.setInputCloud(input_cloud)
voxel_filter.setLeafSize(0.2, 0.2, 0.2)
filtered_cloud = pcl.PointCloud.PointXYZ()
voxel_filter.filter(filtered_cloud)

# Initialize NDT
ndt = pcl.registration.NormalDistributionsTransform.PointXYZ()

# Set NDT parameters
ndt.setTransformationEpsilon(0.01)
ndt.setStepSize(0.1)
ndt.setResolution(1.0)
ndt.setMaximumIterations(35)

# Set input clouds for NDT
ndt.setInputSource(filtered_cloud)
ndt.setInputTarget(target_cloud)

# Initial guess for the alignment
initial_guess = np.eye(4, dtype=np.float32)
initial_guess[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, 0.6931])
initial_guess[0, 3] = 1.79387
initial_guess[1, 3] = 0.720047

# Perform alignment
output_cloud = pcl.PointCloud.PointXYZ()
ndt.align(output_cloud, initial_guess)

# Transform the original input cloud using the final transformation
final_transformation = ndt.getFinalTransformation()
transformed_cloud = pcl.PointCloud.PointXYZ()
pcl.registration.transformPointCloud(input_cloud, transformed_cloud, final_transformation)

# Save the transformed cloud
pcl.io.savePCDFileASCII("room_scan2_transformed.pcd", transformed_cloud)

# Convert PCL point clouds to Open3D point clouds for visualization
target_cloud_o3d = o3d.geometry.PointCloud()
target_cloud_o3d.points = o3d.utility.Vector3dVector(np.asarray(target_cloud.xyz))

transformed_cloud_o3d = o3d.geometry.PointCloud()
transformed_cloud_o3d.points = o3d.utility.Vector3dVector(np.asarray(transformed_cloud.xyz))

# Visualize the clouds
target_cloud_o3d.paint_uniform_color([1, 0, 0])  # Red for target cloud
transformed_cloud_o3d.paint_uniform_color([0, 1, 0])  # Green for transformed cloud

o3d.visualization.draw_geometries([target_cloud_o3d, transformed_cloud_o3d], "NDT Result", width=800, height=600)
