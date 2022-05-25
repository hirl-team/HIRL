# HIRL: A General Framework for Hierarchical Image Representation Learning

This repository provides the PyTorch implementation of the paper [HIRL: A General Framework for Hierarchical Image Representation Learning](https://arxiv.org/pdf/xxx.pdf) and the re-implementations of multiple superior image self-supervised learning (SSL) methods. 
This repository contains complete source code and model weights to reproduce the results in the paper. 

<p align="center">
  <img src="resources/framework.png" /> 
</p>

HIRL is an effective and flexible framework to learn the hierarchical semantic information underlying a large-scale image database. 
It can be flexibly combined with off-the-shelf image SSL approaches and improve them by learning multiple levels of image semantics.
We employ three representative CNN based SSL methods and three representative Vision Transformer based SSL methods as baselines. 
After adapted to the HIRL framework, the effectiveness of all six baseline methods are improved on diverse downstream tasks. 

**Note**: To minimize the dependencies required to reproduce our results on classification-related downstream tasks, 
we put the source code of two transfer learning tasks (object detection and instance segmentation on COCO) in the [det-seg](https://github.com/hirl-team/HIRL/tree/det-seg) branch.
Please move to that branch for reproducing the results on these two tasks.

## Roadmap
- [2022/05/26] The initial release! We release all source code for pre-training and downstream evaluation. We release all pre-trained model weights for (HIRL-)MoCo v2, (HIRL-)SimSiam, (HIRL-)SwAV, (HIRL-)MoCo v3, (HIRL-)DINO and (HIRL-)iBOT.

## TODO
- [ ] Incorporate more baseline image SSL methods in this codebase, e.g., CAE, MAE, BEiT and SimMIM.
- [ ] Adapt more baselines into the HIRL framework, e.g., HIRL-CAE, HIRL-MAE, HIRL-BEiT and HIRL-SimMIM.  
- [ ] Explore other ways to learn hierarchical image representations, except for semantic path discrimination.

## Benchmark and Model Zoo
| Method | Arch. | Epochs | Batch Size | KNN | Linear | Fine-tune | Url | Config |
|---------------|---------|:------:|:------:|:--------:|:--------:|:--------:|-------|---------|
| MoCo v2 | ResNet-50 | 200 | 256 | 55.74 | 67.60 | 73.14 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/mocov2_200eps_backbone.pth) | [cfg](configs/pretrain/baseline/mocov2_resnet50_200eps.yaml) |
| HIRL-MoCo v2 | ResNet-50 | 200 | 256 | 57.56 | 68.40 | 73.86 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_mocov2_200eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_mocov2_resnet50_200eps.yaml) |
| SimSiam | ResNet-50 | 200 | 512 | 60.17 | 69.74 | 72.25 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/simsiam_200eps_backbone.pth) | [cfg](configs/pretrain/baseline/simsiam_resnet50_200eps.yaml) |
| HIRL-SimSiam | ResNet-50 | 200 | 512 | 62.68 | 69.81 | 72.88 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_simsiam_200eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_simsiam_resnet50_200eps.yaml) |
| SwAV | ResNet-50 | 200 | 4096 | 63.45 | 72.68 | 76.82 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/swav_200eps_backbone.pth) | [cfg](configs/pretrain/baseline/swav_resnet50_200eps.yaml) |
| HIRL-SwAV | ResNet-50 | 200 | 4096 | 63.99 | 73.43 | 77.18 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_swav_200eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_swav_resnet50_200eps.yaml) |
| SwAV | ResNet-50 | 800 | 4096 | 64.84 | 73.36 | 77.77 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/swav_800eps_backbone.pth) | [cfg](configs/pretrain/baseline/swav_resnet50_800eps.yaml) |
| HIRL-SwAV | ResNet-50 | 800 | 4096 | 65.43 | 74.80 | 78.05 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_swav_800eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_swav_resnet50_800eps.yaml) |
| MoCo v3 | ViT-B/16 | 400 | 4096 | 71.29 | 76.44 | 81.94 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/mocov3_400eps_backbone.pth) | [cfg](configs/pretrain/baseline/mocov3_vit_base_400eps.yaml) |
| HIRL-MoCo v3 | ViT-B/16 | 400 | 4096 | 71.68 | 75.12 | 82.19 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_mocov3_400eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_mocov3_vit_base_400eps.yaml) |
| DINO | ViT-B/16 | 400 | 1024 | 76.01 | 78.07 | 82.09 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/dino_400eps_backbone.pth) | [cfg](configs/pretrain/baseline/dino_vit_base_400eps.yaml) |
| HIRL-DINO | ViT-B/16 | 400 | 1024 | 76.84 | 78.32 | 83.24 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_dino_400eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_dino_vit_base_400eps.yaml) |
| iBOT | ViT-B/16 | 400 | 1024 | 76.64 | 79.00 | 82.47 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/ibot_400eps_backbone.pth) | [cfg](configs/pretrain/baseline/ibot_vit_base_400eps.yaml) |
| HIRL-iBOT | ViT-B/16 | 400 | 1024 | 77.49 | 79.36 | 83.37 | [model](https://hirlmodels.s3.us-east-2.amazonaws.com/hirl_ibot_400eps_backbone.pth) | [cfg](configs/pretrain/hirl/hirl_ibot_vit_base_400eps.yaml) |

## Installation
This repository is officially tested with the following environments:
- Linux
- Python 3.6+
- PyTorch 1.10.0
- CUDA 11.3

The environment could be prepared in the following steps:
1. Create a virtual environment with conda:
```
conda create -n hirl python=3.7.3 -y
conda activate hirl
```
2. Install PyTorch with the [official instructions](https://pytorch.org/). For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3. Install dependencies:
```
## install apex for LARC
pip install git+https://github.com/NVIDIA/apex \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
## install other dependencies
pip install -r requirements.txt
```

## Usage

### Prepare Dataset
#### ImageNet
We support ImageNet [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/) for pre-training, KNN evaluation, linear classification, fine-tuning, semi-supervised evaluation and unsupervised clustering evaluation. 

We recommend symlink the dataset folder to `./datasets/ImageNet1K`. The folder structure would be:
```
datasets/
  ImageNet1K/
    ILSVRC/
      Annotations/
      Data/
        train/
        val/
      ImageSets/
      meta/
```
After downloading and unzip the dataset, go to path `./datasets/ImageNet1K/ILSVRC/Data/val/` and move images to labeled sub-folders with [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

#### Places205
We also support [Places205](http://places.csail.mit.edu/user/index.php) (resized 256x256 version) for transfer classification experiment.

We recommend symlink the dataset folder to `./datasets/places205`. The folder structure would be:
```
datasets/
  places205/
    data/vision/torralba/deeplearning/images256/
    trainvalsplit_places205/
      train_places205.csv
      val_places205.csv
```

## Launch Experiments
We provide an easy yaml based configuration file system. The config could be modified by 
command line arguments.

To run an experiment:
```
python3 launch.py --launch ./tools/train.py -c [config file] [config options]
```
The config options are in "key=value" format. For example, `ouput_dir=your_path batch_size=64`. 
Sub module is seperated by `.`. For example, `optimizer.name=AdamW` modifies the sub key `name` in 
`optimizer` with value `AdamW`.

A full example:
```
python3 launch.py --launch ./tools/train.py -c configs/pretrain/hirl/hirl_mocov2_resnet50_200eps.yaml \
pipeline.num_mlp_layer=3 output_dir=./experiments/pretrain/hirl/mocov2_3layers/
```

All the pre-training configuration files are in `./configs/pretrain/`. To reproduce the pre-training, please follow the corresponding config file. It is also straight-forward to use customized config files. Suppose the customized config file is stored in `./customized_configs/custom_mocov2.yaml`, the experiment could be launched by :
```
python3 launch.py --launch ./tools/train.py -c ./customized_configs/custom_mocov2.yaml
```

## Multinode training
`launch.py` would automatically find a free port to launch single node experiments. However, some pre-training methods are trained across multiple nodes. In this case, the number of nodes `--nn`, node rank `--nr`, master port `--port` and master address `-ma` should be set. 

Take two node iBOT pre-training as example, use the following commands at node1 and node2, respectively.
```
# use this command at node 1
python3 launch.py --nn 2 --nr 0 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/pretrain/baseline/ibot_vit_base_400eps.yaml

# use this command at node 2
python3 launch.py --nn 2 --nr 1 --port [port] -ma [address of node 0] --launch ./tools/train.py \
-c configs/pretrain/baseline/ibot_vit_base_400eps.yaml
```


## Evaluation
We provide two independent scripts for KNN and unsupervised clustering evaluation on ImageNet. 
For downstream evaluation with training process, you can use `./tools/train.py` with specific configs. 
In most case, the only required argument is `--pretrained`.

### KNN evaluation 
Perform KNN evaluation on a pretrained model:
```
python3 launch.py --launch ./eval_common/eval_knn.py --backbone_prefix backbone --pretrained [pretrained model file in .pth]
```
*Note*: set `--backbone_prefix model.backbone` for HIRL based models. Set `--arch vit_base` for MoCo v3, DINO and iBOT.

### Linear classification
Perform ImageNet linear classification based on a pretrained model (e.g., MoCo v2):
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/lincls/mocov2/mocov2_resnet50_200eps_lincls.yaml \
pretrained=[pretrained model file in .pth]
```
The corresponding HIRL-MoCo v2:
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/lincls/mocov2/hirl_mocov2_resnet50_200eps_lincls.yaml \
pretrained=[pretrained model file in .pth]
```

### Fine-tuning
Perform ImageNet fine-tuning based on a pretrained model (e.g., MoCo v2):
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/finetune/mocov2/mocov2_resnet50_200eps_finetune.yaml \
pretrained=[pretrained model file in .pth]
```
The corresponding HIRL-MoCo v2:
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/finetune/mocov2/hirl_mocov2_resnet50_200eps_finetune.yaml \
pretrained=[pretrained model file in .pth]
```
### Semi-supervised learning
Perform ImageNet semi-supervised learning based on a pretrained model (e.g., MoCo v2):
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/semisup/mocov2/mocov2_resnet50_200eps_semisup_1percent.yaml \
pretrained=[pretrained model file in .pth]
```
The corresponding HIRL-MoCo v2:
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/imagenet/semisup/mocov2/hirl_mocov2_resnet50_200eps_semisup_1percent.yaml \
pretrained=[pretrained model file in .pth]
```

### Transfer learning
Perform Places205 fine-tuning based on a pretrained model (e.g., MoCo v2):
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/places205/finetune/mocov2/mocov2_resnet50_200eps_finetune_places205.yaml \
pretrained=[pretrained model file in .pth]
```
The corresponding HIRL-MoCo v2:
```
python3 launch.py --launch ./tools/train.py \
-c configs/downstream/places205/finetune/mocov2/hirl_mocov2_resnet50_finetune_places205.yaml \
pretrained=[pretrained model file in .pth]
```
### Clustering evaluation
Perform clustering evaluation on a baseline model:
```
python3 launch.py --launch ./eval_common/eval_clustering.py \
--backbone_prefix backbone --pretrained [pretrained model file in .pth]
```
*Note*: set `--backbone_prefix model.backbone` for HIRL based models. Set `--arch vit_base` for MoCo v3, DINO and iBOT.

### Object Detection & Instance Segmentation
See [det-seg](https://github.com/hirl-team/HIRL/tree/det-seg) branch.

## License
This repository is released under the MIT license as in the [LICENSE](LICENSE) file.

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@article{xu2022hirl,
  title={HIRL: A General Framework for Hierarchical Image Representation Learning},
  author={Xu, Minghao and Guo, Yuanfan and Zhu, Xuanyu and Li, Jiawen and Sun, Zhenbang and Tang, Jian and Xu, Yi and Ni, Bingbing},
  journal={arXiv preprint arXiv:2205.xxx},
  year={2022}
}
```

## Acknowledgements
The baseline methods in this codebase are based on the following open-resource projects. We would like to thank the authors for releasing the source code.
- [MoCo](https://github.com/facebookresearch/moco)
- [SimSiam](https://github.com/facebookresearch/simsiam)
- [SwAV](https://github.com/facebookresearch/swav)
- [MoCo V3](https://github.com/facebookresearch/moco-v3)
- [DINO](https://github.com/facebookresearch/dino)
- [iBOT](https://github.com/bytedance/ibot)