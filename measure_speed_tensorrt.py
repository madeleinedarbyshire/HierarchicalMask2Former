import argparse
import time
import torch
import torch_tensorrt
import torch.nn as nn

from torchvision import models, transforms
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image
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

print(torch.__version__)

parser = argparse.ArgumentParser(description="Example argparse parser for config and weights.")

parser.add_argument("--config", type=str, help="Path to the configuration file")
parser.add_argument("--weights", type=str, help="Path to the weights file")
parser.add_argument("--batch-size", type=int, help="Batch size")

args = parser.parse_args()


def add_channels(model, channels):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(len(channels), 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.conv1.weight[:, i] = weight[:, weight_indices[channel]]
    return model

def compile_trt_model(model, batch_size, resolution):
    compile_spec = {'inputs': [torch_tensorrt.Input(min_shape=[batch_size, 3, resolution, resolution],
                                                    opt_shape=[batch_size, 3, resolution, resolution],
                                                    max_shape=[batch_size, 3, resolution, resolution],
                                                    dtype=torch.float32)],
                    'enabled_precisions': torch.float32,
                    'require_full_compilation': True,
                    'truncate_long_and_double': True}

    print('Compiling TensorRT module....')
    trt_ts_module = torch_tensorrt.compile(model, **compile_spec)

    return trt_ts_module

def warm_up(model, batch_size,  num_channels, resolution):
    for _ in range(5):
        inputs = torch.randn(batch_size, num_channels, resolution, resolution).cuda()
        model(inputs)

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
    cfg.merge_from_file(args.config)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

# cfg = setup(args)
# predictor = DefaultPredictor(cfg)
data = PhenoBench("/workspace/PhenoBench", split="val", target_types=[], make_unique_ids=True)
# model = build_model(cfg)
# model.eval()
for i, image in enumerate(data):
    im = np.array(image["image"])
    im = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    example_input = [{'image': im}]
    break
# example_input = [{'image': torch.randn(1, 3, 1024, 1024), 'width': 1024, 'height': 1024}]
# scripted_model = torch.jit.trace(model, example_input)
scripted_model = torch.jit.load('ts_models/old.ts', map_location=torch.device("cuda:0"))
trt_model = compile_trt_model(scripted_model, 1, 1024)

# print('Warm up...')
# warmup()

# print(f'Run inference with batch size: {args.batch_size}...')
# batch = []
# for i, image in enumerate(data):
#     im = np.array(image["image"])
#     im = im[:, :, ::-1]
#     im = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
#     batch.append({'image': im, 'width': 1024, 'height': 1024})
#     if (i + 1) % args.batch_size == 0:
#         start_time = time.time()
#         print('model')
#         outputs = model(batch)
#         end_time = time.time()
#         batch = []
        




# def main(channels, img_root, resolution, model_path, precision, test_device):

#     print('Channels: ', str(channels), '. Resolution: ', resolution, ' Precision:', precision)

#     model_ft = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
#     model_ft = add_channels(model_ft, channels)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, 2)
#     model_ft = model_ft.to(device)
#     state_dict = torch.load(model_path)
#     model_ft.load_state_dict(state_dict)

#     model_ft.eval()

#     model_name = model_path.split('/')[-1].split('.')[0]
#     filename = f'results/2511/tensorrt/{model_name}_{precision}_tensorrt.csv'
#     print('Writing results to file:', filename)
#     file = open(filename, 'w')
#     file.write(f'Batch_Size, Accuracy, Speed, Total\n')
#     file.close()

#     for bsize in [pow(2, x) for x in range(20)]:

#         dataloaders = load_data(resolution, channels, ['val', 'test'], 'resources/dataset_2511.csv', path=img_root, class_path='resources/labels.txt', batch_size=bsize, test_device=test_device)
#         trt_ts_module = compile_trt_model(model_ft, dataloaders['val'], precision, bsize, len(channels), resolution)

#         print('Warming up...')
#         warm_up(trt_ts_module, bsize, len(channels), resolution)
        
#         running_corrects = 0.0
#         running_time = 0.0
#         total = 0.0
#         for input_data, labels in dataloaders['test']:
#             input_data = input_data.to(device)
#             labels = labels.to(device)
#             t1 = time.time()
#             result = trt_ts_module(input_data)
#             running_time += time.time() - t1
#             _, preds = torch.max(result, 1)
#             running_corrects += torch.sum(preds == labels.data)
#             total += input_data.size(0)

#         acc = running_corrects / total
#         inference_speed = 1 / (running_time / total)
#         file = open(filename, 'a')
#         file.write(f'{bsize}, {acc:.4f}, {inference_speed:.4f}, {total:.0f}\n')
#         file.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--channels', nargs="+", type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
#     parser.add_argument('--img-root', type=str, default='../data')
#     parser.add_argument('--model-path', type=str)
#     parser.add_argument('--resolution', type=int)
#     parser.add_argument('--precision', type=str, default='float32')
#     parser.add_argument('--test-device', type=bool)
#     args = parser.parse_args()
#     precision = parse_precision(args.precision)
#     main(args.channels, args.img_root, args.resolution, args.model_path, precision, args.test_device)