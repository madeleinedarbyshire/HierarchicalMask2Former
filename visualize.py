import cv2
import json
import numpy as np
import os

import matplotlib as mpl
from panopticapi.utils import IdGenerator, id2rgb, rgb2id
from phenobench import PhenoBench
from PIL import Image
from pycocotools import mask as mask_utils

id_generator = IdGenerator({0: {"name": "soil", "isthing": 0, "color": (0, 0, 0)}, 
                            1: {"name": "crop", "isthing": 1, "color": (66, 135, 245)},
                            2: {"name": "weed", "isthing": 1, "color": (245, 66, 66)}})

colormap = mpl.colormaps['Set1'].colors
print(colormap[0] + (0.5,))

for split in ['train']:
    plant_train_data = PhenoBench("/workspace/PhenoBench", split=split, target_types=["semantics", "plant_instances", "plant_bboxes", "leaf_instances", "leaf_bboxes"])
    images = []
    annotation_id = 0
    panoptic_annotations = []
    instance_annotations = []
    for image_id, image in enumerate(plant_train_data):
        img_width, img_height = image['image'].size
        print(img_width, img_height)
        soil_annotation_id = id_generator.get_id(0)
        # image_with_unique_ids = np.full((img_height, img_width), soil_annotation_id)
        image_with_unique_ids = cv2.cvtColor(np.array(image["image"]), cv2.COLOR_RGB2BGR)
        alpha = np.zeros((img_height, img_width), dtype=np.uint8)
        segments_info = []
        annotation_id += 1 
        for label in [1, 2]:
            for plant_id in np.unique(image['plant_instances'][image['semantics'] == label]):
                ys, xs = np.where((image['plant_instances'] == plant_id) & (image['semantics'] == label))
                inst_width, inst_height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                center = (np.min(xs) + inst_width // 2, np.min(ys) + inst_height // 2)
                binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                binary_mask[ys, xs] = 1
                color = tuple(np.flip(np.array(colormap[annotation_id % 9])) * 255)
                image_with_unique_ids[ys, xs] = color
                annotation_id = annotation_id + 1

        rgb_image = cv2.cvtColor(image_with_unique_ids, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        blended_image = Image.blend(image["image"], pil_image, 0.5)
        blended_image.save(f'examples/plant/{image["image_name"]}')
        
        image_with_unique_ids = cv2.cvtColor(np.array(image["image"]), cv2.COLOR_RGB2BGR)
        for leaf_id in np.unique(image['leaf_instances']):
                ys, xs = np.where((image['leaf_instances'] == leaf_id) & (image['semantics'] != 0))
                inst_width, inst_height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                center = (np.min(xs) + inst_width // 2, np.min(ys) + inst_height // 2)
                binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                binary_mask[ys, xs] = 1
                color = tuple(np.flip(np.array(colormap[annotation_id % 9])) * 255)
                image_with_unique_ids[ys, xs] = color
                annotation_id = annotation_id + 1
        
        rgb_image = cv2.cvtColor(image_with_unique_ids, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        blended_image = Image.blend(image["image"], pil_image, 0.5)
        blended_image.save(f'examples/leaf/{image["image_name"]}')

        # cv2.imwrite(f'examples/plant/{image["image_name"]}', cv2.cvtColor(id2rgb(image_with_unique_ids), cv2.COLOR_RGB2BGR))

        if image_id == 10:
            break