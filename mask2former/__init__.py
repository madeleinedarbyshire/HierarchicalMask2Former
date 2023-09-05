# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

from .data.dataset_mappers.phenobench_dataset_mapper import (
    PhenoBenchDatasetMapper,
)

# models
# from .maskformer_model import MaskFormer
from .hierarchical_mask2former_model import MaskFormer

# evaluation
from .evaluation.phenobench_evaluation import PhenoBenchEvaluator
