import cv2
import json
import numpy as np
import os

import matplotlib as mpl
# from panopticapi.utils import IdGenerator, id2rgb, rgb2id
from phenobench import PhenoBench
from PIL import Image
from pycocotools import mask as mask_utils

# id_generator = IdGenerator({0: {"name": "soil", "isthing": 0, "color": (0, 0, 0)}, 
#                             1: {"name": "crop", "isthing": 1, "color": (66, 135, 245)},
#                             2: {"name": "weed", "isthing": 1, "color": (245, 66, 66)}})

colormap = mpl.colormaps['Set1'].colors
print(colormap[0] + (0.5,))

val_gt = PhenoBench("~/datasets/PhenoBench", split='val', target_types=["plant_instances", "leaf_instances"])
val_predictions = PhenoBench("~/workspace/HierarchicalMask2Former/results", split='val', target_types=["plant_instances", "leaf_instances", "semantics"], make_unique_ids=False)
    
images = []
annotation_id = 0
panoptic_annotations = []
instance_annotations = []
for image_id, image in enumerate(val_predictions):
    img_width, img_height = image['image'].size
    segments_info = []
    annotation_id += 1
    image_with_unique_ids = np.zeros((img_width, img_height, 3), dtype=np.uint8) 
    for label in [1, 2]:
        for plant_id in np.unique(image['plant_instances'][image['semantics'] == label]):
            if plant_id != 0:
                ys, xs = np.where((image['plant_instances'] == plant_id))
                color = tuple(np.array(colormap[annotation_id % 9]) * 255)
                image_with_unique_ids[ys, xs] = color
                annotation_id = annotation_id + 1

    # rgb_image = cv2.cvtColor(image_with_unique_ids, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_with_unique_ids)
    # blended_image = Image.blend(image["image"], pil_image, 0.5)
    pil_image.save(f'examples/plant/{image["image_name"]}')
            
    # image_with_unique_ids = cv2.cvtColor(np.array(image["image"]), cv2.COLOR_RGB2BGR)
    for leaf_id in np.unique(image['leaf_instances']):
            ys, xs = np.where((image['leaf_instances'] == leaf_id) & (image['semantics'] != 0))
            color = tuple(np.array(colormap[annotation_id % 9]) * 255)
            image_with_unique_ids[ys, xs] = color
            annotation_id = annotation_id + 1
    
    # rgb_image = cv2.cvtColor(image_with_unique_ids, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_with_unique_ids)
    # blended_image = Image.blend(image["image"], pil_image, 0.5)
    pil_image.save(f'examples/leaf/{image["image_name"]}')

    # cv2.imwrite(f'examples/plant/{image["image_name"]}', cv2.cvtColor(id2rgb(image_with_unique_ids), cv2.COLOR_RGB2BGR))

    if image_id == 1:
        break