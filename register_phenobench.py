import json
import numpy as np
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from phenobench import PhenoBench

def register_phenobench(name, metadata, root, split):
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
        load_fn = lambda: PhenoBench(root, split=split, target_types=["semantics", "plant_instances", "leaf_instances"], make_unique_ids=True)
        
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