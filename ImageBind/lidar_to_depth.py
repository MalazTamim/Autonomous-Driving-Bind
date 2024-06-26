from PIL import Image

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
# from os import listdir
from python_sdk.nuscenes import NuScenes


import numpy as np

import os
import matplotlib.pyplot as plt





val = \
    ['scene-0556','scene-0562', 'scene-0778',
 'scene-0099', 'scene-1062', 'scene-0797', 'scene-0634', 'scene-1060', 'scene-0924', 'scene-1066', 'scene-0100', 'scene-1064', 'scene-0919', 'scene-0912',
 'scene-0560', 'scene-0925', 'scene-0107', 'scene-0926', 'scene-0096', 'scene-0770', 'scene-0798', 'scene-0554', 'scene-1065', 'scene-0771', 'scene-0095', 'scene-0625',
 'scene-0559', 'scene-0015', 'scene-0971', 'scene-0968', 'scene-0275', 'scene-0273', 'scene-0003', 'scene-0632', 'scene-0555', 'scene-1069', 'scene-1073', 'scene-0564', 'scene-0274',
 'scene-0106', 'scene-0039', 'scene-0271', 'scene-0094', 'scene-0108', 'scene-0329', 'scene-1067', 'scene-0921', 'scene-0332', 'scene-0557', 'scene-1071']

cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
path_to_dataset = '../../../tmp/datasets/nuscenes'

# batch_size = 32

nusc = NuScenes(version='v1.0-trainval', dataroot=path_to_dataset, verbose=True)
scenes = nusc.scene
pathlist = []
for scene in scenes:
    if scene['name'] in val: 
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        while True:
            for camera in cameras:
                cam_front_data = nusc.get('sample_data', sample['data'][camera])
                pathlist.append(cam_front_data['filename'])
            if sample['next'] == '':
                break
            sample = nusc.get('sample', sample['next'])

with open('sample_paths.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["path"])
    for path in pathlist:
        writer.writerow([path])
# scenes = nusc.scene
# pathlist = []
# for scene in scenes:
#     if scene['name'] in val: 
#         first_sample_token = scene['first_sample_token']
#         sample = nusc.get('sample', first_sample_token)
#         while True:
#             for camera in cameras:
#                 cam_front_data = nusc.get('sample_data', sample['data'][camera])
#                 pathlist.append(cam_front_data['filename'])
#             if sample['next'] == '':
#                 break
#             sample = nusc.get('sample', sample['next'])

def ConverToDepth(camera_channel):  # Ex: camera == "CAM_FRONT"
    counter = 0
    for scene in (nusc.scene):
        if scene['name'] in val:
            # print("hh")
            first_sample_token = scene['first_sample_token']
            my_sample = nusc.get('sample', first_sample_token)

            while True:
                cam_front_data = nusc.get(
                    'sample_data', my_sample['data'][camera_channel])
                camera_file_name = os.path.basename(
                    cam_front_data['filename'])  # Extract camera file name

                

                nusc.render_pointcloud_in_image_black(
                my_sample['token'], pointsensor_channel='LIDAR_TOP', dot_size=130, out_path=os.path.join(
                    '/mnt/workfiles/ImageBind-LoRA/depth_output_whole_dataset/', camera_file_name),
                camera_channel=camera_channel
                )
                counter += 1
                if my_sample['next'] == '':
                    break
                else:
                    my_sample = nusc.get('sample', my_sample['next'])
    print(counter)

# Example
# ConverToDepth("CAM_BACK")
# ConverToDepth("CAM_BACK_RIGHT")
# ConverToDepth("CAM_BACK_LEFT")

# ConverToDepth("CAM_FRONT_RIGHT")
# ConverToDepth("CAM_FRONT_LEFT")
# ConverToDepth("CAM_FRONT")


