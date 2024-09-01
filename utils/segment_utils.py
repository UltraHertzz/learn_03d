import numpy as np
import cv2
import os
from ultralytics import YOLO
from load_utils import load_config, dict_check





def get_instance_mask(sem_info, inst_dict):

    labels = np.array(sem_info['labels'])
    ids = np.array(sem_info['id'])
    masks = np.array(sem_info['masks'])
    instance_mask = np.zeros([1,480,640], dtype=int)

    for i, id in enumerate(ids):
        inst_dict = dict_check(int(id), int(labels[i]), inst_dict)

        mask = masks[i] 
        instance_mask[mask] = int(id)
    return instance_mask, inst_dict

def apply_mask_to_image(sem_info : dict, desired_labels : np.ndarray):
    labels = np.array(sem_info['labels'])
    ids = np.array(sem_info['id'])
    masks = np.array(sem_info['masks'])
    print(f"masks shape is {masks.shape}")
    if len(masks.shape) == 4:
        masks = np.squeeze(masks, axis=1)
    print(f"masks shape after squeeze is {masks.shape}")

    combined_mask = np.zeros_like(masks[0], dtype=bool)

    for disired_label in desired_labels:
        if disired_label in labels:
            indices = np.where(labels == disired_label)[0]
            for index in indices:
                combined_mask = np.logical_or(combined_mask, masks[index])

    return combined_mask


