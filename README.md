# GaussianLSS
A 3D detection extension for CVPR'25 paper:

[[Paper](https://arxiv.org/abs/2504.01957)] [[Project page](https://hcis-lab.github.io/GaussianLSS/)] [[BEV segmentation](https://github.com/HCIS-Lab/GaussianLSS)]

> [**Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting**](https://arxiv.org/abs/2504.01957)<br>
> [Shu-Wei Lu](https://nargoo0328.github.io/shu_wei_lu/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Yi-Ting Chen](https://sites.google.com/site/yitingchen0524/home).<br> 
> [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025)

This implementation is modified from ICCV'23 paper - SparseBEV:

https://github.com/MCG-NJU/SparseBEV
## Environment

Install PyTorch 2.0 + CUDA 11.8:

```
conda create -n GaussianLSS_det python=3.8
conda activate GaussianLSS_det
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other dependencies:

```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

Install turbojpeg and pillow-simd to speed up data loading (optional but important):

```
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow
pip install pillow-simd==9.0.0.post1
pip install -r requirements.txt
```

Compile CUDA extensions:

```
cd models/diff-gaussian-rasterization
pip install -e .
```

## Prepare Dataset

1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. Download the generated info file from [Google Drive](https://drive.google.com/file/d/1uyoUuSRIVScrm_CUpge6V_UzwDT61ODO/view?usp=sharing) and unzip it.
3. Folder structure:

```
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_train_mini_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── nuscenes_infos_val_mini_sweep.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

These `*.pkl` files can also be generated with our script: `gen_sweep_info.py`.

## Training

Download pretrained [weights](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) and put it in directory `pretrain/`:

```
pretrain
├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

Train GaussianLSS with 2 GPUs (i.e the last four GPUs):

```
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 train.py --config configs/r50_nuimg_704x256.py
```

The batch size for each GPU will be scaled automatically. So there is no need to modify the `batch_size` in config files.

## Evaluation

Single-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/r50_nuimg_704x256.py --weights {PATH_TO_WEGIHTS.pth}
```

## Timing

FPS is measured with a single GPU:

```
export CUDA_VISIBLE_DEVICES=0
python timing.py --config configs/r50_nuimg_704x256.py --weights {PATH_TO_WEGIHTS.pth}
```

## Visualization

Visualize the predicted bbox:

```
python viz_bbox_predictions.py --config configs/r50_nuimg_704x256.py --weights {PATH_TO_WEGIHTS.pth}
```
