import json
import numpy as np
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from phenobench import PhenoBench

def load_phenobench(root, split):
    """
    Args:
        root (str): path to the raw dataset. e.g., "~/Phenobench".
        split (str): train or val.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    data = PhenoBench(root, split=split, target_types=["semantics", "plant_instances", "leaf_instances"], make_unique_ids=True)
    ret = []

    for image_id, image in enumerate(data):
        segments_info = [{'id': 0,
                          'category_id': 0,
                          'isthing': False,
                          'iscrowd': 0}]
        for label in [1, 2]:
            for plant_id in np.unique(image['plant_instances'][image['semantics'] == label]):
                    segments_info.append({'id': plant_id,
                                          'category_id': int(label),
                                          'isthing': True,
                                          'iscrowd': 0})
        ret.append(
            {
                "file_name": f"images/{image['image_name']}",
                "image_id": image_id,
                "pan_seg_file_name": f"plant_instances/{image['image_name']}",
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret
    

def register_phenobench(name, metadata, root, split):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

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