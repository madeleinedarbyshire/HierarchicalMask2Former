# Hierarchical Mask2Former for Segmenting Crops, Weeds and Leaves
An adaptation of [Mask2Former](https://github.com/facebookresearch/Mask2Former) to perform hierarchical panoptic segmentation of crops, weeds and leaves. For implementation details read [our preprint](https://www.researchgate.net/publication/373549760_Hierarchical_Mask2Former_Panoptic_Segmentation_of_Crops_Weeds_and_Leaves).

## Dataset
The implementation was tested on the [PhenoBench Dataset](https://www.phenobench.org).

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2:
  - `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
  - Or other installation options can be found at [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Getting Started
To train:
```
python train_net.py --num-gpus 2 --config-file configs/phenobench/mask2former_R50_bs2_100ep.yaml
```

To test:
```
python train_net.py --num-gpus 2 --config-file configs/phenobench/mask2former_R50_bs2_100ep.yaml \
--eval-only MODEL.WEIGHTS path/to/weights
```

## Citation
If you find my work useful please consider the following bibtex entry:
```
@article{hierarchical2023darbyshire,
         author = {Darbyshire, Madeleine and Sklar, Elizabeth and Parsons, Simon},
         year = {2023},
         month = {08},
         pages = {},
         title = {Hierarchical Mask2Former: Panoptic Segmentation of Crops, Weeds and Leaves},
         doi = {10.13140/RG.2.2.33051.23847}
}
```

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Hierarchical Mask2Former is licensed under an MIT License.

However portions of the project are available under separate license terms: Mask2Former is licensed under the [MIT license](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE), Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).
