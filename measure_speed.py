import argparse
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image
import time
import numpy as np
import cv2
from detectron2.modeling import build_model

import detectron2.utils.comm as comm

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from phenobench import PhenoBench

# MaskFormer
from mask2former import add_maskformer2_config


parser = argparse.ArgumentParser(description="Example argparse parser for config and weights.")

parser.add_argument("--config_file", type=str, help="Path to the configuration file")
parser.add_argument("--weights", type=str, help="Path to the weights file")

args = parser.parse_args()

def warmup():
    # Set the device to GPU (assuming you have a GPU available)
    device = torch.device("cuda")

    # Create a simple tensor and send it to the GPU
    warm_up_tensor = torch.randn(1, 1).to(device)

    # Perform some simple operations on the tensor
    for _ in range(1000):
        warm_up_tensor = torch.matmul(warm_up_tensor, warm_up_tensor)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

cfg = setup(args)
predictor = DefaultPredictor(cfg)
data = PhenoBench("/workspace/PhenoBench", split="val", target_types=[], make_unique_ids=True)
model = build_model(cfg)
model.eval()

print('Warm up...')
warmup()

print('Run inference...')


for i, image in enumerate(data):
    im = np.array(image["image"])
    im = torch.as_tensor(np.ascontiguousarray(im.transpose(2, 0, 1)))
    if i % 2 == 0:
        batch = [{'image': im}]
    else:
        batch.append({'image': im})
        # batch = torch.stack(batch)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(batch)
        end_time = time.time()

