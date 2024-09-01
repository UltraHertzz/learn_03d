import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
import numpy as np
from .load_utils import json_read

class TUMDataset(Dataset):


    def __init__(
            self,  
            dataset_dir, config=None, transform=None):
        if config is not None:
            self.config = config
            self.dataset_dir = config['Dataset']['dir']
            if self.config["Dataset"]["semantic_info"]:
                self.semantic_info = json_read(os.path.join(self.dataset_dir, 'semantic_infomation.txt'))
        else:
            self.dataset_dir = dataset_dir
        
        self.rgb_file_list = self.read_file_list(
            os.path.join(dataset_dir, 'rgb.txt'))
        self.depth_file_list = self.read_file_list(
            os.path.join(dataset_dir, 'depth.txt'))
        self.gt_pose_list = self.read_pose_list(
            os.path.join(dataset_dir, 'groundtruth.txt'))
        
        self.rgb_file_list, self.depth_file_list, self.gt_pose_list = self.align_files(
            self.rgb_file_list, self.depth_file_list, self.gt_pose_list)
        
        
        self.transform = transform
    
    def __len__(self):
        return len(self.rgb_file_list)
    
    def __getitem__(self, idx):

        rgb_img_name = os.path.join(
            self.dataset_dir, self.rgb_file_list[idx])
        
        depth_img_name = os.path.join(
            self.dataset_dir, self.depth_file_list[idx])
        
        gt_pose = self.gt_pose_list[idx][0]

        rgb_image = Image.open(rgb_img_name)
        depth_image = Image.open(depth_img_name).convert('I')
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        return rgb_img_name, depth_img_name, gt_pose
    
    def read_file_list(self, filename):

        with open(filename, 'r') as file:
            lines = file.readlines()
        file_list = []
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp, filename = parts
                file_list.append((float(timestamp), filename))

        return file_list
    
    def read_pose_list(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        pose_list = []
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = float(parts[0])
                pose = [float(x) for x in parts[1:]]
                pose_list.append((timestamp, pose))
        return pose_list
    
    
    def align_files(self, rgb_list, depth_list, pose_list, threshold=0.1):

        def find_nearest_index(taget_timestamp, timestamp):
            timestamp = np.array(timestamp)
            idx = np.argmin(np.abs(timestamp - taget_timestamp))
            return idx

        aligned_rgb = []
        aligned_depth = []
        aligned_pose = []

        max_gap_rgb_depth = 0.0
        max_gap_rgb_pose = 0.0

        for rgb_time, rgb_file in rgb_list:

            depth_idx = find_nearest_index(rgb_time, [x[0] for x in depth_list])
            if abs(depth_list[depth_idx][0] - rgb_time) > max_gap_rgb_depth:
                max_gap_rgb_depth = abs(depth_list[depth_idx][0] - rgb_time)
            pose_idx = find_nearest_index(rgb_time, [x[0] for x in pose_list])
            if abs(pose_list[pose_idx][0] - rgb_time) > max_gap_rgb_pose:
                max_gap_rgb_pose = abs(pose_list[pose_idx][0] - rgb_time)

            aligned_rgb.append(rgb_file)
            aligned_depth.append(depth_list[depth_idx][1])
            aligned_pose.append(pose_list[pose_idx][1:])
        print('max_gap_rgb_depth: ', max_gap_rgb_depth)
        print('max_gap_rgb_pose: ', max_gap_rgb_pose)

        return aligned_rgb, aligned_depth, aligned_pose


    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config


