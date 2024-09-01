import torch
import numpy as np
import argparse
import yaml
import json
import os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from utils.dataset import TUMDataset
from utils.camera_utils import Camera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 生成随机点云
def generate_random_point_cloud(n_points=100):
    return np.random.rand(n_points, 3) * 10

def build_kdtree(point_cloud):
    return cKDTree(point_cloud)


def compute_voxel_covariances(points, voxel_size):
    voxel_map = {}
    for point in points:
        voxel_idx = tuple((point // voxel_size).astype(int))
        if voxel_idx not in voxel_map:
            voxel_map[voxel_idx] = []
        voxel_map[voxel_idx].append(point)
    
    covariances = {}
    for voxel_idx, pts in voxel_map.items():
        pts = np.array(pts)
        mean = np.mean(pts, axis=0)
        cov = np.cov(pts, rowvar=False) + np.eye(3) * 1e-6  # 防止奇异矩阵
        covariances[voxel_idx] = (mean, cov)
    return covariances


def ndt_cost_function(params, source_cloud, voxel_covariances, voxel_size):
    rotation_matrix = euler_to_rotation_matrix(params[:3])
    translation = params[3:6]
    
    transformed_points = np.dot(source_cloud, rotation_matrix.T) + translation
    
    cost = 0
    for point in transformed_points:
        voxel_idx = tuple((point // voxel_size).astype(int))
        if voxel_idx in voxel_covariances:
            mean, cov = voxel_covariances[voxel_idx]
            diff = point - mean
            cost += np.dot(diff.T, np.dot(np.linalg.inv(cov), diff))
    
    return cost

def quaternion_to_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

def euler_to_rotation_matrix(euler_angles):
    roll, pitch, yaw = euler_angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return np.dot(Rz, np.dot(Ry, Rx))

def NDT_test():

    source_cloud = generate_random_point_cloud()
    target_cloud = generate_random_point_cloud()

    target_kdtree = build_kdtree(target_cloud)

    voxel_size = 1.0
    target_voxel_covariances = compute_voxel_covariances(target_cloud, voxel_size)

    initial_guess = np.zeros(6)  # 初始参数，3个旋转（欧拉角），3个平移
    result = minimize(ndt_cost_function, initial_guess, args=(source_cloud, target_voxel_covariances, voxel_size), method='BFGS')

    optimal_params = result.x

    optimal_rotation = euler_to_rotation_matrix(optimal_params[:3])
    optimal_translation = optimal_params[3:6]

    transformed_source_cloud = np.dot(source_cloud, optimal_rotation.T) + optimal_translation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_cloud[:,0], source_cloud[:,1], source_cloud[:,2], c='r', label='Source')
    ax.scatter(target_cloud[:,0], target_cloud[:,1], target_cloud[:,2], c='b', label='Target')
    ax.scatter(transformed_source_cloud[:,0], transformed_source_cloud[:,1], transformed_source_cloud[:,2], c='g', label='Transformed Source')
    ax.legend()
    plt.show()


def rgbd_pcd_test(dataset, config):
    frame1 = dataset[0]
    frame2 = dataset[1]
    viewport1 = Camera(frame1, config)
    viewport2 = Camera(frame2, config)
    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(viewport1.rgb_image, 
                                                               viewport1.depth_image,
                                                               depth_scale=config["Dataset"]["Calibration"]["depth_scale"],
                                                               depth_trunc=config["Dataset"]["Calibration"]["depth_trunc"],
                                                               convert_rgb_to_intensity=False)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(viewport2.rgb_image,
                                                                viewport2.depth_image,
                                                                depth_scale=config["Dataset"]["Calibration"]["depth_scale"],
                                                                depth_trunc=config["Dataset"]["Calibration"]["depth_trunc"],
                                                                convert_rgb_to_intensity=False)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, viewport1.intrinsics)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, viewport2.intrinsics)
    down_scaled_pcd1 = pcd1.voxel_down_sample(voxel_size=0.1)

    down_scaled_pcd2 = pcd2.voxel_down_sample(voxel_size=0.1)
    pcd_transform = np.array([[1, 0, 0, 0],
                                [0, 1, 0, -3],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    down_scaled_pcd2.transform(pcd_transform)
    o3d.visualization.draw([down_scaled_pcd1, down_scaled_pcd2])
    return down_scaled_pcd1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config parser')
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_dir = config['Dataset']['dir']
    dataset = TUMDataset(dataset_dir)
    # NDT_test()
    pcd = rgbd_pcd_test(dataset, config)
