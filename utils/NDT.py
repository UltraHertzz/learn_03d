import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import minimize

class NDTGrid:

    def __init__(self, grid_size, points) -> None:
        self.grid_size = grid_size
        self.points = points
        self.grid = {}
        self.build_grid()


    def build_grid(self):
        for point in self.points:
            grid_index = tuple((point // self.grid_size).astype(int))
            if grid_index not in self.grid:
                self.grid[grid_index] = []
            self.grid[grid_index].append(point)
        self.calculate_distributions()

    def calculate_distributions(self):
        for index in self.grid:
            points = np.array(self.grid[index])
            if len(points) > 1:
                mean = np.mean(points, axis=0)
                cov = np.cov(points.T)  # 计算协方差矩阵
                self.grid[index] = (mean, cov)
            else:
                self.grid[index] = (points[0], np.eye(2))

    def get_distribution(self, point):
        grid_index = tuple((point // self.grid_size).astype(int))
        if grid_index in self.grid:
            return self.grid[grid_index]
        else:
            return None
    
    # Step 2: 配准目标点云
    def ndt_score(transformation, target_points, ndt_grid):
        rotation = transformation[0]
        translation = transformation[1:3]
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        transformed_points = np.dot(target_points, rotation_matrix.T) + translation

        score = 0
        for point in transformed_points:
            distribution = ndt_grid.get_distribution(point)
            if distribution is not None:
                mean, cov = distribution
                diff = point - mean
                score += np.exp(-0.5 * np.dot(diff.T, np.dot(np.linalg.inv(cov), diff)))

        return -score  # 返回负分数，优化器最小化这个值

    def build_kdtree(self, point_cloud):
        return cKDTree(point_cloud)
    
    def compute_voxel_covariances(self, points, voxel_size):
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
            cov = np.cov(pts, rowvar=False) + np.eye(3) * 1e-6