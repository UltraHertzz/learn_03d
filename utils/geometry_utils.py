import numpy as np
from numba import cuda
import time
import open3d as o3d
from segment_utils import get_instance_mask, apply_mask_to_image
from load_utils import json_read
import torch
import cv2 as cv
# import inverse_projection_cuda

@cuda.jit
def inverse_projection_kernel(K: np.ndarray, 
                              depth_image: np.ndarray, 
                              rgb_image, 
                              instance_mask=None, 
                              points=None, 
                              colors=None, 
                              ids=None, 
                              width=640, 
                              height=480):
    # 获取线程在 2D 网格中的位置
    u, v = cuda.grid(2)
    
    if u < width and v < height:
        # 相机内参
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 深度值
        Z = depth_image[v, u]
        
        if Z > 0:  # 忽略无效的深度
            # 计算归一化平面坐标
            x_n = (u - cx) / fx
            y_n = (v - cy) / fy

            # 计算 3D 点 (X, Y, Z)
            X = x_n * Z
            Y = y_n * Z

            r = rgb_image[v, u, 0]
            g = rgb_image[v, u, 1]
            b = rgb_image[v, u, 2]
            if ids is not None:
                instance_id = instance_mask[v, u]


            # 将结果写入输出数组
            idx = v * width + u
            points[idx, 0] = X
            points[idx, 1] = Y
            points[idx, 2] = Z
            colors[idx, 0] = r
            colors[idx, 1] = g
            colors[idx, 2] = b
            if ids is not None:
                ids[idx] = instance_id
            else:
                ids = None

def inverse_projection_cuda_numba(K, depth_image, rgb_image, instance_mask=None):
    height, width = depth_image.shape

    # 创建输出数组，在 GPU 上存储 3D 点
    points = np.zeros((height*width, 3), dtype=np.float32)
    colors = np.zeros((height*width, 3), dtype=np.uint8)
    ids = np.zeros((height*width), dtype=np.int8)
    
    # 将输入数据移动到 GPU
    depth_image_device = cuda.to_device(depth_image)
    rgb_image_device = cuda.to_device(rgb_image)
    if instance_mask is not None:
        instance_mask_device = cuda.to_device(instance_mask)
        ids_device = cuda.to_device(ids)
    points_device = cuda.to_device(points)
    colors_device = cuda.to_device(colors)
    
    K_device = cuda.to_device(K)

    # 定义网格大小和线程块大小
    threads_per_block = (16, 
                         16)
    blocks_per_grid_x = int(np.ceil(width / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(height / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # 启动 CUDA 内核
    start_kernel_time = time.time()
    inverse_projection_kernel[blocks_per_grid, threads_per_block](
        K_device, 
        depth_image_device, 
        rgb_image_device,
        instance_mask_device,
        points_device, 
        colors_device,
        ids_device,
        width, 
        height)

    # 将结果从 GPU 拷贝回 CPU
    cuda.synchronize()
    end_kernel_time = time.time()
    print(f"Time taken to run kernel: {end_kernel_time - start_kernel_time} seconds")

    points_device.copy_to_host(points)
    colors_device.copy_to_host(colors)
    if instance_mask is not None:
        ids_device.copy_to_host(ids)
    
        return points, colors, ids
    return points, colors, None


def inverse_projection_cpu(K, depth_image, rgb_image, instance_mask=None, width=640, height=480):
    # 相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建输出数组
    points = np.zeros((height * width, 3), dtype=np.float32)
    colors = np.zeros((height * width, 3), dtype=np.uint8)
    ids = np.zeros((height * width), dtype=np.int8) if instance_mask is not None else None

    # 遍历每个像素
    for v in range(height):
        for u in range(width):
            # 获取深度值
            Z = depth_image[v, u]

            if Z > 0:  # 忽略无效的深度
                # 计算归一化平面坐标
                x_n = (u - cx) / fx
                y_n = (v - cy) / fy

                # 计算 3D 点 (X, Y, Z)
                X = x_n * Z
                Y = y_n * Z

                # 获取颜色信息
                r = rgb_image[v, u, 0]
                g = rgb_image[v, u, 1]
                b = rgb_image[v, u, 2]

                # 计算点的索引
                idx = v * width + u
                points[idx, 0] = X
                points[idx, 1] = Y
                points[idx, 2] = Z
                colors[idx, 0] = r
                colors[idx, 1] = g
                colors[idx, 2] = b

                # 如果提供了实例掩码，获取实例ID
                if ids is not None:
                    instance_id = instance_mask[v, u]
                    ids[idx] = instance_id

    return points, colors, ids


def inverse_projection_pytorch(K, depth_image, rgb_image, instance_mask=None):
    start_time = time.time()
    height, width = depth_image.shape

    # Convert inputs to PyTorch tensors and move to GPU
    depth_image_tensor = torch.tensor(depth_image, device='cuda')
    rgb_image_tensor = torch.tensor(rgb_image, device='cuda')
    K_tensor = torch.tensor(K, device='cuda')
    
    if instance_mask is not None:
        instance_mask_tensor = torch.tensor(instance_mask, device='cuda')
    
    # Prepare output tensors
    points = torch.zeros((height * width, 3), dtype=torch.float32, device='cuda')
    colors = torch.zeros((height * width, 3), dtype=torch.uint8, device='cuda')
    ids = torch.zeros((height * width), dtype=torch.int8, device='cuda') if instance_mask is not None else None

    # Camera intrinsics
    fx, fy = K_tensor[0, 0], K_tensor[1, 1]
    cx, cy = K_tensor[0, 2], K_tensor[1, 2]

    # Create a meshgrid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(width, device='cuda'), torch.arange(height, device='cuda'), indexing='xy')
    u = u.flatten()
    v = v.flatten()

    # Get corresponding depth values
    Z = depth_image_tensor[v, u]
    
    valid_mask = Z > 0  # Only consider valid depth points
    
    # Compute normalized coordinates
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # Compute 3D points
    X = x_n * Z
    Y = y_n * Z

    # Assign to output tensors
    points[:, 0] = X
    points[:, 1] = Y
    points[:, 2] = Z
    
    # Assign colors
    colors[:, 0] = rgb_image_tensor[v, u, 0]
    colors[:, 1] = rgb_image_tensor[v, u, 1]
    colors[:, 2] = rgb_image_tensor[v, u, 2]

    # Handle instance mask if provided
    if instance_mask is not None:
        ids = instance_mask_tensor[v, u]
        ids[~valid_mask] = 0  # Set invalid points to zero in the ID mask

    # Apply valid mask to filter out invalid points
    points = points[valid_mask]
    colors = colors[valid_mask]
    if instance_mask is not None:
        ids = ids[valid_mask]
    end_time = time.time()
    print(f"Time taken to run PyTorch code: {end_time - start_time} seconds")
    return points.cpu().numpy(), colors.cpu().numpy(), ids.cpu().numpy() if instance_mask is not None else None


def pcd_segment_with_instance_id(pcd, color, id_list, instance_id):
    # 创建一个掩码，其中实例 ID 与指定 ID 匹配
    instance_indices = np.where(id_list == instance_id)
    print(f"instance_indices shape is {instance_indices}")
    pcd_instance = pcd[instance_indices]
    color_instance = color[instance_indices]
    id_instance = id_list[instance_indices]
    return pcd_instance, color_instance, id_instance



# 示例使用
if __name__ == "__main__":


    start_time = time.time()
    # 假设我们有一个相机内参矩阵 K
    K = np.array([[535.4,   0.0, 320.1],
                  [  0.0, 539.2, 247.6],
                  [  0.0,   0.0,   1.0]], dtype=np.float32)

    # 生成一个假设的深度图像
    depth_image = o3d.io.read_image("1341846647.802269.png")
    # depth_image = cv.imread("1341846647.802269.png", cv.IMREAD_UNCHANGED)
    depth_image = np.asarray(depth_image, dtype=np.float32)
    # depth_image = np.random.uniform(0.5, 4.0, (480, 640)).astype(np.float32)
    # 生成一个随机布尔掩码，其中一半为 True
    # mask = np.random.rand(480, 640) < 0.5

    # 应用掩码，将一半的深度值置为零
    # depth_image[mask] = 0

    # rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # instance_mask = np.random.randint(0, 10, (480, 640), dtype=np.int8)
    rgb_image = o3d.io.read_image("1341846647.734820.png")
    # rgb_image = cv.imread("1341846647.734820.png", cv.IMREAD_UNCHANGED)
    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    sem_info = json_read("1341846647.734820_info.json")
    sem_dict = {
        0 : 100
    }
    instance_mask, sem_dict = get_instance_mask(sem_info,sem_dict)
    print(sem_dict.items())
    instance_mask = np.asarray(instance_mask, dtype=np.int8)
    print(instance_mask)

    # 使用 CUDA 加速的逆向投影生成点云
    point_cloud, rgb, id = inverse_projection_cuda_numba(K, depth_image, rgb_image, instance_mask[0])
    # point_cloud, rgb, id = inverse_projection_cpu(K, depth_image, rgb_image, instance_mask[0])
    # point_cloud, rgb, id = inverse_projection_pytorch(K, depth_image, rgb_image, instance_mask[0])
    non_zero_indices = np.nonzero(point_cloud[:, 2])
    
    point_cloud = point_cloud[non_zero_indices]
    rgb = rgb[non_zero_indices]
    id = id[non_zero_indices]
    print("Time:", time.time() - start_time)
    
    # 打印结果
    print("Generated point cloud shape:", point_cloud.shape)
    print("Generated non-zero point cloud shape:", point_cloud.shape)
    print("Generated RGB shape:", rgb.shape)
    print("Generated instance ID shape:", id.shape)


    
    pcd = o3d.geometry.PointCloud()
    point_cloud, rgb, id = pcd_segment_with_instance_id(point_cloud, rgb, id, 0)
    print(f"point_cloud shape is {point_cloud.shape}")
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb/255)
    o3d.visualization.draw([pcd])
