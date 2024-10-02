import multiprocessing
import numpy as np
import os
import shutil
import tifffile as tiff
import time
import torch

from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from functools import partial
from phenobench import PhenoBench
from PIL import Image
from scipy.ndimage import distance_transform_edt as eucl_distance
from tqdm import tqdm

def convert_to_unsigned(matrix):
    sign_bit = np.where(matrix < 0, np.uint16(1 << 15), np.uint16(0))  # Set the sign bit where necessary
    unsigned_matrix = np.abs(matrix).astype(np.uint16) + sign_bit
    return unsigned_matrix

def calculate_dist_map(posmask):
    negmask = ~posmask
    signed_dist_map = eucl_distance(negmask) * negmask - (eucl_distance(posmask) - 1) * posmask
    signed_dist_map = signed_dist_map.astype(np.int16)
    unsigned_dist_map = convert_to_unsigned(signed_dist_map)
    img = Image.fromarray(unsigned_dist_map)
    return img

def process_dataset(image, tfm_gens=None, completed=None):
    print('Checking for image:', image['image_name'])
    if image['image_name'] not in completed:
        print('Processing image:', image['image_name'])
        name = image["image_name"].split('.')[0]
        os.makedirs(f'dist_maps/{name}', exist_ok=True)
        if tfm_gens:
            _, transforms = T.apply_transform_gens(tfm_gens, np.array(image["image"]))
        for level in ["plant", "leaf"]:
            os.makedirs(f'dist_maps/{name}/{level}', exist_ok=True)
            if os.path.exists(f'dist_maps/{name}/{level}/half'):
                shutil.rmtree(f'dist_maps/{name}/{level}/half')
            os.makedirs(f'dist_maps/{name}/{level}/half', exist_ok=True)
            pan_seg_gt_before = image[f"{level}_instances"]
            sem_seg_gt_before = image["semantics"]

            sem_seg_gt = sem_seg_gt_before.astype(np.float32)
            pan_seg_gt = pan_seg_gt_before.astype(np.float32)

            if tfm_gens:
                if level == "plant":
                    sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

                sem_seg_gt = np.round(sem_seg_gt)
                pan_seg_gt = np.round(pan_seg_gt)

                if pan_seg_gt.shape[0] > 1024:
                    if level == "plant":
                        sem_seg_gt = sem_seg_gt[0:1024, 0:1024]
                    pan_seg_gt = pan_seg_gt[0:1024, 0:1024]
                elif pan_seg_gt.shape[0] < 1024:
                    if level == "plant":
                        sem_seg_gt = np.pad(sem_seg_gt, [int((1024 - sem_seg_gt.shape[0]) / 2), int((1024 - sem_seg_gt.shape[1]) / 2)], mode='constant', constant_values=0)
                    pan_seg_gt = np.pad(pan_seg_gt, [int((1024 - pan_seg_gt.shape[0]) / 2), int((1024 - pan_seg_gt.shape[1]) / 2)], mode='constant', constant_values=0)

            sem_seg_gt = sem_seg_gt.astype(np.int32)
            pan_seg_gt = pan_seg_gt.astype(np.int32)

            if not os.path.exists(f'dist_maps/{name}/{level}/half/0.png'):
                posmask = pan_seg_gt == 0
                img = calculate_dist_map(posmask)
                img.save(f'dist_maps/{name}/{level}/half/0.png')    

            if level == "plant":
                for label in [1, 2]:
                    for plant_id in np.unique(pan_seg_gt[sem_seg_gt == label]):
                        if not os.path.exists(f'dist_maps/{name}/{level}/half/{plant_id}.png'):
                            posmask = pan_seg_gt == plant_id
                            img = calculate_dist_map(posmask)
                            img.save(f'dist_maps/{name}/{level}/half/{plant_id}.png') 
            else:
                for leaf_id in np.unique(pan_seg_gt):
                    if not os.path.exists(f'dist_maps/{name}/{level}/half/{leaf_id}.png'):
                        posmask = pan_seg_gt == leaf_id
                        img = calculate_dist_map(posmask)
                        img.save(f'dist_maps/{name}/{level}/half/{leaf_id}.png')
        with open('half.txt', 'a') as file:
            file.write(image['image_name'] + '\n')
    print('Finished image:', image['image_name'])


plant_train_data = PhenoBench("../../datasets/PhenoBench", split='train', target_types=["semantics", "plant_instances", "plant_bboxes", "leaf_instances", "leaf_bboxes"], make_unique_ids=False)

with open('half.txt', 'r') as file:
        file_contents = file.readlines()

file_contents = [f.strip('\n') for f in file_contents]

image_size = 1024
factor = 0.5

print('Factor', factor)

transforms = [T.Resize((int(image_size * factor), int(image_size * factor)), interp=0)]

with multiprocessing.Pool() as pool:
    pool.map(partial(process_dataset, tfm_gens=transforms, completed=file_contents), plant_train_data)

    