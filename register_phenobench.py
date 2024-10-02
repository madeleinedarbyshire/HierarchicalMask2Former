import json
import numpy as np
import os
import time

from detectron2.data import DatasetCatalog, MetadataCatalog

import os
from typing import Dict, List

import numpy as np
from PIL import Image

class PhenoBench:
  """ `PhenoBench <https://www.phenobench.org/>`_ Dataset.

    This is a very basic dataloader with some utility function that can be easily integrated into a PyTorch, 
    PyTorch Lightning, etc. data loader. We kept the requirements low profile to ensure extensibility.
  
    Args:
      root (str): Root directory of the PhenoBench dataset, where the folders `train`, `val`, and `test` data is location.
      split (str, optional): The image split to use, `train`, `test`, or `val`.
      target_type (List[str], optional): Types of targets to use and provide with each sample, i.e., `semantics`, `plant_instances`, `leaf_instances`, `plant_bboxes`, `leaf_bboxes`, `plant_visibility`, and/or `leaf_visibility`.
      ignore_partial (bool, optional): Ignore partial annotations? if `True` all partial plants, i.e., will be mask using the `ignore_mask` in `semantics` and mapped to `0` in the instance masks. Default: `False`.
      make_unique_ids (bool, optional): Make the instance ids of weeds and crops unique. Default: True.
      ignore_mask(int, optional): ignore_index. Default: 255.
  """
  def __init__(self,
               root: str,
               split: str = 'train',
               target_types: List[str] = ['semantics'],
               ignore_partial: bool = False,
               make_unique_ids: bool = True,
               ignore_mask: int = 255,
               resize_aug: bool = True) -> None:

    # sanity checks:
    root = os.path.expanduser(root)
    assert os.path.exists(root), f"The path to the dataset does not exist: `{root}`."
    assert split in ['train', 'val', 'test']
    for target_type in target_types:
      assert target_type in [
          'semantics', 'plant_instances', 'leaf_instances', 'plant_bboxes', 'leaf_bboxes', 'plant_visibility',
          'leaf_visibility', 'plant_dist_maps', 'leaf_dist_maps'
      ]

    self.root = root
    self.split = split
    self.target_types = target_types
    self.ignore_partial = ignore_partial
    self.make_unique_ids = make_unique_ids
    self.ignore_mask = ignore_mask
    self.resize_aug = resize_aug

    self.filenames = sorted(os.listdir(os.path.join(self.root, split, "images")))

  def __getitem__(self, index: int) -> Dict[str, np.array]:
    """
    Args:
        index (int): index of the image & annotation

    Returns:
        Dict[str, Any]: returns a sample with the specified target_types.

        A data point is comprised of the following information (at least):
        ```
        {
          "image_name": "
          "image": PIL Image with size W x H.
          ...
          <target>: np.array of shape W x H or a dictionary with the bounding boxes.
        }
        ```
    """
    sample = {}

    sample["image_name"] = self.filenames[index]
    sample["image"] = Image.open(os.path.join(self.root, self.split, "images", self.filenames[index])).convert("RGB")

    if self.split in ["train", "val"]:
      
      if self.resize_aug:
        sample["scale"] = np.random.choice([0.5, 1.0, 2.0])
      else:
        sample["scale"] = np.random.choice([1.0])

      if sample["scale"] == 0.5:
        dist_map_dir = 'half'
      elif sample["scale"] == 1.0:
        dist_map_dir = 'original'
      else:
        dist_map_dir = 'double'

      for target in self.target_types:
        if target in ["semantics", "plant_instances", "leaf_instances", "plant_visibility", "leaf_visibility"]:
          sample[target] = np.array(Image.open(os.path.join(self.root, self.split, target, self.filenames[index])))

        if self.split == "train":
          image_name = sample["image_name"].split(".")[0]
          if target == "plant_dist_maps":
            sample[target] = [Image.open(os.path.join('/nvmedrive/dist_maps', image_name, 'plant', dist_map_dir, file)) for file in sorted(os.listdir(os.path.join('/nvmedrive/dist_maps', image_name, 'plant', dist_map_dir)), key=lambda x: int(os.path.splitext(x)[0]))]

          if target == "leaf_dist_maps":
            sample[target] = [Image.open(os.path.join('/nvmedrive/dist_maps', image_name, 'leaf', dist_map_dir, file)) for file in sorted(os.listdir(os.path.join('/nvmedrive/dist_maps', image_name, 'leaf', dist_map_dir)), key=lambda x: int(os.path.splitext(x)[0]))]

      if self.ignore_partial:
        semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
        partial_crops = semantics == 3
        partial_weeds = semantics == 4

        if "semantics" in self.target_types:
          sample["semantics"][partial_crops] = self.ignore_mask
          sample["semantics"][partial_weeds] = self.ignore_mask

        for target in self.target_types:
          if target in ["plant_instances", "leaf_instances"]:
            sample[target][partial_crops] = 0
            sample[target][partial_weeds] = 0
          ## FIXME: do something for the bounding boxes.
          if target in ["plant_bboxes", "leaf_bboxes"]:
            pass

      else:
        if "semantics" in self.target_types:
          # remap partial_crop to crop and partial_weed to weed:
          sample["semantics"][sample["semantics"] == 3] = 1
          sample["semantics"][sample["semantics"] == 4] = 2

        if "plant_bboxes" in self.target_types:
          semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
          plant_instances = np.array(
              Image.open(os.path.join(self.root, self.split, "plant_instances", self.filenames[index])))
          leaf_instances = np.array(
              Image.open(os.path.join(self.root, self.split, "leaf_instances", self.filenames[index])))

          sample["plant_bboxes"] = []
          for label in [1, 2]:
            for plant_id in np.unique(plant_instances[semantics == label]):
              ys, xs = np.where((plant_instances == plant_id) & (semantics == label))

              width, height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
              center = (np.min(xs) + width // 2, np.min(ys) + height // 2)
              sample["plant_bboxes"].append({
                  "label": label,
                  "corner": (np.min(xs), np.min(ys)),
                  "center": center,
                  "width": width,
                  "height": height
              })

        if "leaf_bboxes" in self.target_types:
          semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
          leaf_instances = np.array(
              Image.open(os.path.join(self.root, self.split, "leaf_instances", self.filenames[index])))

          sample["leaf_bboxes"] = []
          for leaf_id in np.unique(leaf_instances[leaf_instances > 0]):
            ys, xs = np.where((leaf_instances == leaf_id))

            width, height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
            center = (np.min(xs) + width // 2, np.min(ys) + height // 2)
            sample["leaf_bboxes"].append({
                "label": 1,
                "corner": (np.min(xs), np.min(ys)),
                "center": center,
                "width": width,
                "height": height
            })

    # remapping the instance ids to consecutive ids:
    if self.make_unique_ids:

      def replace(array: np.array, values, replacements):
        temp_array = array.copy()

        for v, r in zip(values, replacements):
          temp_array[array == v] = r

        array = temp_array

      if "plant_instances" in self.target_types:
        semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))

        crop_ids = np.unique(sample["plant_instances"][semantics == 1])
        weed_ids = np.unique(sample["plant_instances"][semantics == 2])

        N, M = len(crop_ids), len(weed_ids)

        replace(sample["plant_instances"][semantics == 1], crop_ids, np.arange(1, N + 1))
        replace(sample["plant_instances"][semantics == 2], weed_ids, np.arange(N + 1, N + M + 1))

      if "leaf_instances" in self.target_types:
        leaf_ids = np.unique(sample["leaf_instances"][sample["leaf_instances"] > 0])

        replace(sample["leaf_instances"], leaf_ids, np.arange(1, len(leaf_ids) + 1))

    return sample

  def __len__(self):
    return len(self.filenames)

def register_phenobench(name, metadata, root, split, resize_aug=True):
    """
    Args:
        name (str): dataset_name (e.g. phenobench_train)
        root (str): the phenobench dataset root
        split (dict): train, val or test
        image_root (str): directory which contains all the images
    """
    if split == "test":
        load_fn = lambda: PhenoBench(root, split=split, target_types=[])
    else:
        load_fn = lambda: PhenoBench(root, split=split, target_types=["semantics", "plant_instances", "leaf_instances", "plant_dist_maps", "leaf_dist_maps"], make_unique_ids=False, resize_aug=resize_aug)
        
    DatasetCatalog.register(
        name,
        load_fn
    )
    MetadataCatalog.get(name).set(
        root=root,
        panoptic_root=os.path.join(root, split, 'plant_instances'),
        semantic_root=os.path.join(root, split, 'semantics'),
        image_root=os.path.join(root, split, 'images'),
        evaluator_type = "phenobench",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )