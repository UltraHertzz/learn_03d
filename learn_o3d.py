
import os
import time
import open3d as o3d
import json
import numpy as np
import torch
import cv2 
import argparse
from utils.load_utils import json_read, img_read, dict_check, load_config
from utils.camera_utils import Camera, quaternion_to_rotation_matrix, getWorld2View2, rotation_matrix, transformation_matrix
from utils.segment_utils import get_instance_mask, apply_mask_to_image
from utils.geometry_utils import inverse_projection_cuda
from utils.dataset import TUMDataset



def obj_bg_separation(rgb_img_array, depth_img_array, combined_mask):
        # obj rgbd
    obj_rgb_image = np.zeros_like(rgb_img_array)
    obj_rgb_image[combined_mask] = rgb_img_array[combined_mask]
    obj_depth_image = np.zeros_like(depth_img_array)
    obj_depth_image[combined_mask] = depth_img_array[combined_mask]
    print(f"obj_rgb_image type is {type(obj_rgb_image)}")
    obj_depth_image = obj_depth_image.astype(np.float32)

    # bg rgbd
    bg_rgb_image = np.zeros_like(rgb_img_array)
    bg_rgb_image[~combined_mask] = rgb_img_array[~combined_mask]
    bg_depth_image = np.zeros_like(depth_img_array)
    bg_depth_image[~combined_mask] = depth_img_array[~combined_mask]
    bg_depth_image = bg_depth_image.astype(np.float32)

    return    bg_rgb_image, \
            bg_depth_image, \
             obj_rgb_image, \
           obj_depth_image

def create_pcd_from_depth_test():
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, 
                                                          intrinsics, 
                                                          extrinsic_matrix,
                                                          depth_scale=1000.0, 
                                                          depth_trunc=20.0, 
                                                          project_valid_depth_only=True)
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    downpcd = pcd.voxel_down_sample(voxel_size=0.2)
    pcd_tmp = pcd_tmp.random_down_sample(1/64)
    print(f"original pcd has {len(pcd.points)} points, downsampled pcd has {len(downpcd.points)} points")
    o3d.visualization.draw([downpcd, coordinate_frame])

def create_pcd_from_rgbd_test(rgb_img, depth_img, extrinsic_matrix, intrinsics, depth_trunc=20.0):
    rgb_img = o3d.geometry.Image(rgb_img)
    depth_img = o3d.geometry.Image(depth_img)

    start_time = time.time()
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, 
                                                              depth_img, 
                                                              depth_scale=5000, 
                                                              depth_trunc=depth_trunc, 
                                                              convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                                         intrinsics, 
                                                         )
    pcd.transform(extrinsic_matrix)
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_tmp = pcd.random_down_sample(1/32)
    tmp_transform = np.array([[1, 0, 0, 3], 
                              [0, 1, 0, 0], 
                              [0, 0, 1, 0], 
                              [0, 0, 0, 1]])
    pcd_transform = np.array([[1, 0, 0, 0],
                                [0, 1, 0, -3],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    pcd_tmp = pcd_tmp.transform(tmp_transform)
    pcd = pcd.transform(pcd_transform)
    end_time = time.time()
    print(f"Time taken to create pcd from rgbd: {end_time - start_time} seconds")
    print(f"original pcd has {len(pcd.points)} points, downsampled pcd has {len(downpcd.points)} points, pcd_tmp has {len(pcd_tmp.points)} points")
    o3d.visualization.draw([downpcd, pcd_tmp, pcd])
    return pcd

def depth_array_metrix_printer(depth_array):

    print(f"shape of depth_image is {depth_array.shape}")
    print(f"type of depth_image is {type(depth_array)}")

    max_depth = np.max(depth_array)
    mean_depth = np.mean(depth_array)
    std_depth = np.std(depth_array)
    print(f"Maximum Depth: {max_depth} meters")
    print(f"Mean Depth: {mean_depth} meters")
    print(f"Standard Deviation of Depth: {std_depth} meters")
    trunc_depth = mean_depth + 2*std_depth
    print(f"Truncated Depth: {trunc_depth} meters")
    return trunc_depth

def self_pcd_creation_test():
    points, colors, ids = inverse_projection_cuda(K, depth_img_array, rgb_img_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Process some integers.')
    argparser.add_argument('--config', type=str, default='config.yaml')
    args = argparser.parse_args()
    config = load_config(args.config)

    dataset = TUMDataset(config)

    # rgb_img_path = '1341847980.722988.png'
    # depth_img_path = '1341847980.723020.png'

    rgb_img_path = '1341846647.734820.png'
    depth_img_path = '1341846647.802269.png'
    rgb_name, ext = os.path.splitext(rgb_img_path)
    json_path = rgb_name + '_info.json'
    
    state = torch.tensor([-0.6243, -2.7776, 1.4753, -0.7614, -0.0229, 0.0090, 0.6478])

    t = state[:3]
    q = state[3:]

    R = quaternion_to_rotation_matrix(q)
    W2C = getWorld2View2(torch.tensor(R), t).cpu().numpy()
    K = Camera.create_camera_intrinsics(535.4, 539.2, 320.1, 247.6)

    rgb_img = o3d.io.read_image(rgb_img_path)
    depth_img = o3d.io.read_image(depth_img_path)
    sem_info = json_read(json_path)
    # desired_labels = np.array([0.])
    rgb_img_array = np.asarray(rgb_img)

    depth_img_array = np.asarray(depth_img)
    print(f"count of depth_img_array is {np.count_nonzero(depth_img_array)}")
    trunc_depth = depth_array_metrix_printer(depth_img_array)
    # combined_mask = apply_mask_to_image(sem_info, desired_labels)
    # bg_rgb_img, bg_depth_img, obj_rgb_img, obj_depth_img = obj_bg_separation(rgb_img_array, depth_img_array, combined_mask)
    

    rotation_angles = {
        'x': torch.tensor(180., device='cuda'),
        'y': torch.tensor(90., device='cuda'),
        'z': torch.tensor(0., device='cuda')
    }
    rotation_angles = {
        'x': torch.tensor(0., device='cuda'),
        'y': torch.tensor(0., device='cuda'),
        'z': torch.tensor(0., device='cuda')
    }

    rotations = rotation_matrix(rotation_angles)
    translation = torch.tensor([0, 0, 0], device='cuda')
    extrinsic_matrix = transformation_matrix(rotations, translation).cpu().numpy()
    extrinsic_matrix = W2C
    print(f"extrinsic_matrix is {extrinsic_matrix}")

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width=480, height=640, fx=535.4, fy=539.2, cx=320.1, cy=247.6)
    #create_pcd_from_depth_test()
    # pcd = create_pcd_from_rgbd_test(rgb_img_array,      # bg_rgb_img, 
                                    # depth_img_array,    # bg_depth_img, 
                                    # extrinsic_matrix,    
                                    # intrinsics,   
                                    # depth_trunc=1000)   # trunc_depth/5000.0)
 