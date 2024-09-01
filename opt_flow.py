import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import keyboard
import argparse
import yaml
import open3d as o3d

# 读取两帧图像



class TUMDataset(Dataset):
    def __init__(
            self, 
            rgb_file_list, 
            depth_file_list, 
            root_dir, transform=None):
        
        self.rgb_file_list = rgb_file_list
        self.depth_file_list = depth_file_list
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.rgb_file_list)
    
    def __getitem__(self, idx):
        rgb_img_name = os.path.join(
            self.root_dir, self.rgb_file_list[idx])
        depth_img_name = os.path.join(
            self.root_dir, self.depth_file_list[idx])
        rgb_image = Image.open(rgb_img_name)
        depth_image = Image.open(depth_img_name).convert('I')
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        return rgb_image, depth_image
    
def read_file_list(filename):
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

def render_flow(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def render_magnitude_on_rgb(rgb_image, magnitude):
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_color = cv2.applyColorMap(magnitude_norm.astype(np.uint8), cv2.COLORMAP_JET)
    blended_image = cv2.addWeighted(rgb_image, 0.6, magnitude_color, 0.4, 0)
    return blended_image

def opt_flow_angle_viz(mag, ang):
    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb



def align_files(root_dir, rgb_list, depth_list):
    aligned_rgb = []
    aligned_depth = []
    depth_iter = iter(depth_list)
    depth_item = next(depth_iter, None)
    
    for rgb_time, rgb_file in rgb_list:
        while depth_item and depth_item[0] < rgb_time:
            depth_item = next(depth_iter, None)
        if depth_item and abs(depth_item[0] - rgb_time) < 0.02 and os.path.exists(os.path.join(root_dir,depth_item[1])):  # 0.02 seconds threshold
            aligned_rgb.append(rgb_file)
            aligned_depth.append(depth_item[1])
    
    return aligned_rgb, aligned_depth

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optical Flow')
    parser.add_argument('--config', type=str, help='config parser')

    args = parser.parse_args()
    config = load_config(args.config)
    dataset_dir = config['Dataset']['dir']

    rgb_dir = os.path.join(dataset_dir, "rgb.txt")
    depth_dir = os.path.join(dataset_dir, "depth.txt")

    rgb_list = read_file_list(rgb_dir)
    depth_list = read_file_list(depth_dir)

    # Align the files
    aligned_rgb_list, aligned_depth_list = align_files(dataset_dir, rgb_list, depth_list)


    root_dir = config['Dataset']['dir']
    dataset = TUMDataset(
        aligned_rgb_list, 
        aligned_depth_list, 
        root_dir, 
        transform=transforms.ToTensor())
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    step = 0

    while step < len(dataset) - 1:

        gap = config["Param"]["image_gap"]
        rgb_image1, depth_image1 = dataset[step]
        rgb_image2, depth_image2 = dataset[step+gap]

        # Convert tensors to numpy arrays
        rgb_image1_np = rgb_image1.numpy().transpose(1, 2, 0)
        depth_image1_np = depth_image1.numpy().squeeze()  # Assuming depth image has single channel
        rgb_image2_np = rgb_image2.numpy().transpose(1, 2, 0)

        # Convert to grayscale for optical flow calculation
        gray1 = cv2.cvtColor(rgb_image1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(rgb_image2.numpy().transpose(1, 2, 0), 
                             cv2.COLOR_RGB2GRAY)


        # 计算光流
        start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(f"shape of flow is {flow.shape}")
        mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Display results
            # Plotting

        # Compute the optical flow vector field
        h, w = gray1.shape

        step_size = config["Param"]["sample_step"]
        y, x = np.mgrid[step_size//2:h:step_size, step_size//2:w:step_size].astype(int)
        fx, fy = flow[y, x].T
        scalar = np.sqrt(fx**2 + fy**2)
        fx, fy = fx / scalar, fy / scalar
        print("Max flow magnitude:", np.max(mag))
        print("Min flow magnitude:", np.min(mag))
        print("Max flow angle:", np.max(angle))
        print("Min flow angle:", np.min(angle))

        num_zeros = np.count_nonzero(depth_image1_np == 0.0)
        num_nans = np.count_nonzero(np.isnan(depth_image1_np))
        if step == 0:
            num_prev_zeros = num_zeros
            num_zeros_diff = 0
        else:
            num_zeros_diff = num_zeros - num_prev_zeros
            num_prev_zeros = num_zeros

        rendered_Image = render_magnitude_on_rgb(rgb_image2_np, mag)
        # combined_image = cv2.addWeighted(cv2.cvtColor(rgb_image1_np, cv2.COLOR_BGR2RGB), 0.6, depth_image1_np, 0.4, 0)

        axs[0, 0].imshow(rgb_image1_np)
        axs[0, 0].set_title('RGB Image')

        axs[0, 1].imshow(depth_image1_np, cmap='gray')
        axs[0, 1].set_title('Depth Image')

        if config["Mode"]["mode"] == "angle":
            # Convert to HSV image: hue = angle, saturation and value are set to max (1)
            angle_normalized = angle / (2 * np.pi)
            hsv_image = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.float32)
            hsv_image[..., 0] = angle_normalized  # Hue (H) corresponds to the angle
            hsv_image[..., 1] = 1  # Saturation (S) set to 1 (max)
            hsv_image[..., 2] = 1  # Value (V) set to 1 (max)

            # Convert HSV to RGB for visualization
            rgb_angle_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            axs[1, 0].imshow(rgb_angle_image)
            axs[1, 0].set_title('Optical Flow Angle (HSV)')


        else:
            scale = 1/config["Param"]["sample_step"]
            axs[1, 0].quiver(x, y, fx, fy, color='red', angles='xy', scale_units='xy', scale=scale, linewidth=0.5)
        axs[1, 0].set_title('Optical Flow Vector Field')

        # axs[1, 0].imshow(mag, cmap='gray')
        # axs[1, 0].set_title('Optical Flow Magnitude')
        # axs[1, 1].imshow(combined_image)
        axs[1, 1].imshow(rendered_Image, cmap='hsv')
        axs[1, 1].set_title('Optical Flow Angle')

        plt.draw()
        plt.pause(0.1)  # Short pause to allow update

        # Wait for user to press enter to continue
        user_input = input("Press Enter to continue to the next frame...")
        if user_input == '0':
            step += 10
        else:
            step += 1
    
        # Clear the current axes.
        for ax in axs.flat:
            ax.clear()
        # Print depth statistics
        print(f'Idx: {step}, Depth mean: {np.mean(depth_image1_np)/5000}, median: {np.median(depth_image1_np)/5000}, max: {np.max(depth_image1_np)/5000}')
        print(f'Number of 0 values: {num_zeros/(640*480)}, 0 value diff: {num_zeros_diff/num_zeros}')

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Ensure window stays open at the end