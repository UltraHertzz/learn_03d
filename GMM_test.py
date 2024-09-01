import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils.dataset import TUMDataset
from utils.load_utils import load_config
import argparse
from ultralytics import YOLO
import pcl

def calculate_mse(imageA, imageB):
    # 计算均方误差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_psnr(imageA, imageB):
    # 计算峰值信噪比
    mse = calculate_mse(imageA, imageB)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def calculate_ssim(imageA, imageB):
    # 计算结构相似性
    s, _ = ssim(imageA, imageB, full=True)
    return s

if __name__ == '__main__':
    # 加载配置文件
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='config.yaml')
    args = args.parse_args()
    config = load_config(args.config)
    dataset = TUMDataset(config['Dataset']['dir'])

    initialized = False
    for frame in iter(dataset):
        if not initialized:
            image1 = cv2.imread(frame[0], cv2.IMREAD_GRAYSCALE)
            initialized = True
            continue
        image2 = cv2.imread(frame[0], cv2.IMREAD_GRAYSCALE)

        mse_value = calculate_mse(image1, image2)
        psnr_value = calculate_psnr(image1, image2)
        ssim_value = calculate_ssim(image1, image2)

        print(f"MSE: {mse_value}")
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
        image1 = image2
