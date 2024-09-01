import yaml
import argparse
import json
import open3d as o3d

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def json_read(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def img_read(img_path):
    img = o3d.io.read_image(img_path)
    return img

def dict_check(add_key, add_value, dict):
    if add_key not in dict:
        dict[add_key] = add_value
    return dict




    