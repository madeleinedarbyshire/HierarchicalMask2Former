# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import cv2
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["PhenoBenchDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
        T.RandomLighting(255),
        # T.RandomRotation([0, 359])
    ])

    return augmentation

class PhenoBenchDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        is_test=False,
        *,
        tfm_gens,
        image_format,
        image_size
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[PhenoBenchDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.is_test = is_test
        self.image_size = image_size

    @classmethod
    def from_config(cls, cfg, is_train=True, is_test=False):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_test": is_test,
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "image_size": cfg.INPUT.IMAGE_SIZE
        }
        return ret

    def gen_color(self, taken_colors):
        while True:
            color = tuple(np.random.randint(0, 256, size=3))
            if color not in taken_colors:
                return color

    def apply_segmentation_transforms(self, transforms, gt):
        color_map = {}
        labels = np.unique(gt)                    
        h, w = gt.shape[:2]
        rgb_gt = np.zeros((h, w, 3))
        for l in labels:
            color = self.gen_color(color_map.values())
            color_map[l] = color
            ys, xs = np.where(gt == l)
            rgb_gt[ys, xs] = color
        rgb_gt = transforms.apply_segmentation(rgb_gt)
        h,w = rgb_gt.shape[:2]
        gt = np.zeros((h, w))
        for i, color in color_map.items():
            color_mask = np.all(rgb_gt == color, axis=-1)
            gt[color_mask] = i
        return gt
                    

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        input_dict = {}
        image = dataset_dict['image']

        if self.image_size != 1024:
            resized = image.resize((self.image_size, self.image_size), resample=0)
            resized = np.array(resized)
            input_dict["resized_image"] = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)))
        else:
            resized = np.array(image)
            input_dict["resized_image"] = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)))

        # if self.is_train:
        #     image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image = np.array(image)
        image_shape = image.shape[:2]
        input_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        input_dict["image_name"] = dataset_dict["image_name"]

        if not self.is_test:
            for level in ["plant", "leaf"]:
                classes = [0]

                pan_seg_gt = dataset_dict[f"{level}_instances"]
                sem_seg_gt = dataset_dict["semantics"]

                # if self.image_size != 1024:
                #     pan_seg_gt = np.resize(pan_seg_gt, (self.image_size, self.image_size))
                #     sem_seg_gt = np.resize(sem_seg_gt, (self.image_size, self.image_size))

                if self.is_train:
                    if level == "plant":
                        sem_seg_gt = np.array(sem_seg_gt, dtype=np.uint8)
                        # sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                    pan_seg_gt = np.array(pan_seg_gt, dtype=np.uint8)
                    # pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
                    
                masks = [pan_seg_gt == 0]

                if level == "plant":
                    for label in [1, 2]:
                        for plant_id in np.unique(pan_seg_gt[sem_seg_gt == label]):
                            classes.append(label)
                            masks.append(pan_seg_gt == plant_id)
                else:
                    for leaf_id in np.unique(pan_seg_gt):
                        classes.append(1)
                        masks.append(pan_seg_gt == leaf_id)

                instances = Instances(image_shape)
                classes = np.array(classes)
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                if len(masks) == 0:
                    # Some images do not have any annotations (all ignored)
                    instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                    instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                else:
                    masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                    )
                    instances.gt_masks = masks.tensor
                    instances.gt_boxes = masks.get_bounding_boxes()

                input_dict[f"{level}_instances"] = instances

        return input_dict