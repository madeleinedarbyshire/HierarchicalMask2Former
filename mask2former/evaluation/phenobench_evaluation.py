import cv2
import itertools
import logging
import numpy as np
import os
from pathlib import Path
import tempfile
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator

from phenobench.evaluation.evaluate_leaf_instance_masks_panoptic import (
    evaluate_leaf_instances,
)
from phenobench.evaluation.evaluate_plant_bounding_boxes import evaluate_plant_detection
from phenobench.evaluation.evaluate_plant_instance_masks_panoptic import (
    evaluate_plant_instances,
)
from phenobench.evaluation.evaluate_semantics import evaluate_semantics

class PhenoBenchEvaluator(DatasetEvaluator):

        def __init__(self, dataset_name, prediction_dir, test=False):
            self._metadata = MetadataCatalog.get(dataset_name)
            self._prediction_dir = prediction_dir
            self.test = test
            if self.test:
                self.root_name = "results"
            else:
                self.root_name = "/tmp/out2"
            self._semantics_dir = os.path.join(self.root_name, "semantics")
            self._plant_instances_dir = os.path.join(self.root_name, "plant_instances")
            self._leaf_instances_dir = os.path.join(self.root_name, "leaf_instances")
            self._distributed = True
            self._predictions = []

            os.makedirs(self._semantics_dir, exist_ok=True)
            os.makedirs(self._plant_instances_dir, exist_ok=True)
            os.makedirs(self._leaf_instances_dir, exist_ok=True)

        def process(self, inputs, outputs):
            for input, output in zip(inputs, outputs):
                plant_panoptic_img, segments_info = output["plant_panoptic_seg"]
                # print('number of plant segments', len(segments_info))
                plant_panoptic_img = plant_panoptic_img.cpu().numpy()
                leaf_panoptic_img, segments_info = output["leaf_panoptic_seg"]
                # print('number of leaf segments', len(segments_info))
                leaf_panoptic_img = leaf_panoptic_img.cpu().numpy()
                # print(panoptic_img.shape)
                # panoptic_img = panoptic_img.cpu().numpy()
                # print('sem shape', output["sem_seg"].shape)
                semantics = output["plant_sem_seg"].argmax(dim=0).cpu().numpy()
                cv2.imwrite(os.path.join(self._semantics_dir, input["image_name"]), semantics)
                cv2.imwrite(os.path.join(self._plant_instances_dir, input["image_name"]), plant_panoptic_img)
                cv2.imwrite(os.path.join(self._leaf_instances_dir, input["image_name"]), leaf_panoptic_img)

        def _evaluate(self, rank=None, world_size=None, result_queue=None):

            semantic_results = evaluate_semantics({"phenobench_dir": Path(self._metadata.root),
                                                    "prediction_dir": Path(self.root_name),
                                                    "split": "val"})

            instance_results = evaluate_plant_instances({"phenobench_dir": Path(self._metadata.root),
                                                        "prediction_dir": Path(self.root_name),
                                                        "split": "val"})

            leaf_results = evaluate_leaf_instances({"phenobench_dir": Path(self._metadata.root),
                                                    "prediction_dir": Path(self.root_name),
                                                    "split": "val"})

            iou_soil = semantic_results['soil']
            iou_crop = semantic_results['crop']
            iou_weed = semantic_results['weed']
            pq_crop = instance_results['plants_cls'][1]['pq']
            pq_weed = instance_results['plants_cls'][2]['pq']
            pq_leaf = leaf_results["leaves_pq"]

            results = {'IoU (soil)': iou_soil,
                       'IoU (crop)': iou_crop,
                       'IoU (weed)': iou_weed,
                       'PQ (crop)': pq_crop,
                       'PQ (leaf)': pq_leaf,
                       'PQ (weed)': pq_weed,
                       'PQ': (pq_crop+pq_leaf)/2,
                       'PQ+': (iou_soil+iou_weed+pq_crop+pq_leaf)/4}
            if result_queue:
                result_queue.put(results)

            print(f"{'IoU (soil)':10}: {iou_soil:4}")
            print(f"{'IoU (crop)':10}: {iou_crop:4}")
            print(f"{'IoU (weed)':10}: {iou_weed:4}")

            print(f"{'PQ (crop)':10}: {pq_crop:4}")
            print(f"{'PQ (leaf)':10}: {pq_leaf:4}")
            print(f"{'PQ (weed)':10}: {pq_weed:4}")
            print(f"{'PQ':10}: {(pq_crop+pq_leaf)/2:4.2f}")
            print(f"{'PQ+':10}: {(iou_soil+iou_weed+pq_crop+pq_leaf)/4:4.2f}")

            return results          

        def evaluate(self):
            if not self.test:
                print('Evaluating Results...')
                logger = logging.getLogger(__name__)
                results = self._evaluate()
                logger.info(results)
            

