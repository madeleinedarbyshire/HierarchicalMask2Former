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

    augmentation.extend([
        T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            ),
        T.RandomRotation([0, 90, 180, 270], sample_style="choice"),
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

        if is_train:
            logging.getLogger(__name__).info(
                "[PhenoBenchDatasetMapper] Full TransformGens used in training: {}".format(
                    str(self.tfm_gens)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[PhenoBenchDatasetMapper] Full TransformGens used in evaluation: {}".format(
                    str(self.tfm_gens)
                )
            )

        self.img_format = image_format
        self.is_train = is_train
        self.is_test = is_test

    @classmethod
    def from_config(cls, cfg, is_train=True, is_test=False):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_test": is_test,
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def gen_color(self, taken_colors):
        while True:
            color = tuple(np.random.randint(0, 256, size=3))
            if color not in taken_colors:
                return color

    def convert_to_signed(self, unsigned_matrix):
        sign_bit_mask = np.uint16(1 << 15)
        sign_bit_set = unsigned_matrix & sign_bit_mask != 0
        signed_matrix = np.int16(unsigned_matrix)
        signed_matrix[sign_bit_set] -= np.uint16(1 << 15)
        signed_matrix[sign_bit_set] *= -1        
        return signed_matrix

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        input_dict = {}
        image = np.array(dataset_dict['image'])
        image_shape = image.shape[:2]

        if self.is_train:
            resize_aug = [T.Resize((int(1024 * dataset_dict['scale']), int(1024 * dataset_dict['scale'])), interp=0)]
            image, resize_transforms = T.apply_transform_gens(resize_aug, image)
            if image.shape[0] > 1024:
                image = image[0:1024, 0:1024]
            elif image.shape[0] < 1024:
                p = int((1024 - image.shape[0]) / 2)
                image = np.pad(image, [(p,p), (p,p), (0,0)], mode='constant', constant_values=128)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        input_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        input_dict["image_name"] = dataset_dict["image_name"]

        if not self.is_test:
            for level in ["plant", "leaf"]:
                pan_seg_gt = dataset_dict[f"{level}_instances"]
                sem_seg_gt = dataset_dict["semantics"]                

                if self.is_train:
                    if dataset_dict["scale"] != 1:
                        sem_seg_gt = sem_seg_gt.astype(np.float32)
                        sem_seg_gt = resize_transforms.apply_segmentation(sem_seg_gt)
                        sem_seg_gt = np.round(sem_seg_gt)
                        pan_seg_gt = pan_seg_gt.astype(np.float32)    
                        pan_seg_gt = resize_transforms.apply_segmentation(pan_seg_gt)
                        pan_seg_gt = np.round(pan_seg_gt)

                    if pan_seg_gt.shape[0] > 1024:
                        sem_seg_gt = sem_seg_gt[0:1024, 0:1024]
                        pan_seg_gt = pan_seg_gt[0:1024, 0:1024]
                    elif pan_seg_gt.shape[0] < 1024:
                        sem_seg_gt = np.pad(sem_seg_gt, [int((1024 - sem_seg_gt.shape[0]) / 2), int((1024 - sem_seg_gt.shape[1]) / 2)], mode='constant', constant_values=0)
                        pan_seg_gt = np.pad(pan_seg_gt, [int((1024 - pan_seg_gt.shape[0]) / 2), int((1024 - pan_seg_gt.shape[1]) / 2)], mode='constant', constant_values=0)

                    sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                    sem_seg_gt = sem_seg_gt.astype(np.int32)

                    pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
                    pan_seg_gt = pan_seg_gt.astype(np.int32)
                    dist_maps = [transforms.apply_segmentation(self.convert_to_signed(np.array(d, dtype=np.uint16))) for d in dataset_dict[f"{level}_dist_maps"]]

                classes = []
                masks = []
                if level == "plant":
                    classes.append(0)
                    masks.append(pan_seg_gt == 0)
                    for label in [1, 2]:
                        for plant_id in sorted(np.unique(pan_seg_gt[sem_seg_gt == label])):
                            if plant_id != 0:
                                classes.append(label)
                                masks.append(pan_seg_gt == plant_id)

                else:
                    classes.append(0)
                    masks.append(pan_seg_gt == 0)
                    for leaf_id in sorted(np.unique(pan_seg_gt)):
                        if leaf_id != 0:
                            classes.append(1)
                            masks.append(pan_seg_gt == leaf_id)
                                
                instances = Instances(image_shape)
                classes = np.array(classes)
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                if len(masks) == 0:
                    # Some images do not have any annotations (all ignored)
                    instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

                    if self.is_train:
                        instances.dist_maps = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                else:
                    masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                    )
                    instances.gt_masks = masks.tensor

                    if self.is_train:
                        instances.dist_maps = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in dist_maps])

                input_dict[f"{level}_instances"] = instances

        return input_dict