# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import tempfile
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    HookBase,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.data.common import ToIterableDataset, MapDataset
from detectron2.data.samplers import TrainingSampler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    PhenoBenchDatasetMapper,
    PhenoBenchEvaluator,
    add_maskformer2_config,
)

from register_phenobench import register_phenobench


class WeightUpdateHook(HookBase):
    def __init__(self, trainer, n_iterations):
        super().__init__()
        self.trainer = trainer
        self.n_iterations = n_iterations

    def after_step(self):
        if self.trainer.iter % self.n_iterations == 0:
            self.trainer.model.update_weights()

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def build_hooks(cls):
        hooks = super().build_hooks()
        hooks.insert(-1, WeightUpdateHook(cls, n_iterations=448))  # Insert WeightUpdateHook before the last hook
        return hooks


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Returns:
            PhenoBenchEvaluator
        """
        prediction_dir = tempfile.TemporaryDirectory()
        return PhenoBenchEvaluator(dataset_name, prediction_dir, test=cfg.TEST.NO_EVAL)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PhenoBenchDatasetMapper(cfg, True)
        dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        return build_detection_train_loader(dataset=dataset, mapper=mapper, aspect_ratio_grouping=False, num_workers=cfg.DATALOADER.NUM_WORKERS, total_batch_size=cfg.SOLVER.IMS_PER_BATCH)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = PhenoBenchDatasetMapper(cfg, False, cfg.TEST.NO_EVAL, tfm_gens=[])
        if cfg.TEST.NO_EVAL:
            dataset = DatasetCatalog.get(cfg.DATASETS.TEST_NO_EVAL[0])
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        return build_detection_test_loader(dataset=dataset, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    meta = {"thing_dataset_id_to_contiguous_id": {1:1, 2:2},
        "stuff_dataset_id_to_contiguous_id": {0:0},
        "thing_classes": ['crop', 'weed'],
        "thing_colors": [(66, 135, 245), (245, 66, 66)], 
        "stuff_classes": ['soil']}

    # register_phenobench("phenobench_train_extra", meta, "/nvmedrive/PhenoBenchExtra", split="train", resize_aug=cfg.RESIZE_AUG)
    # register_phenobench("phenobench_val_extra", meta, "/nvmedrive/PhenoBenchExtra", split="val", resize_aug=cfg.RESIZE_AUG)

    register_phenobench("phenobench_train", meta, "/nvmedrive/PhenoBenchExtra", split="train", resize_aug=cfg.RESIZE_AUG)
    register_phenobench("phenobench_val", meta, "/nvmedrive/PhenoBenchExtra", split="val", resize_aug=cfg.RESIZE_AUG)
    register_phenobench("phenobench_test", meta, "/nvmedrive/PhenoBench", split="test", resize_aug=cfg.RESIZE_AUG)
    set_seed(cfg.SEED)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
