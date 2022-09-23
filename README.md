
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2dpass-2d-priors-assisted-semantic/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=2dpass-2d-priors-assisted-semantic)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2dpass-2d-priors-assisted-semantic/lidar-semantic-segmentation-on-nuscenes)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-nuscenes?p=2dpass-2d-priors-assisted-semantic)

# 2DPASS

[![arXiv](https://img.shields.io/badge/arXiv-2203.09065-b31b1b.svg)](https://arxiv.org/pdf/2207.04397v2.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/yanx27/2DPASS?style=social)](https://github.com/yanx27/2DPASS)
![visitors](https://visitor-badge.glitch.me/badge?page_id=https://github.com/yanx27/2DPASS)



This repository is for **2DPASS** introduced in the following paper

[Xu Yan*](https://yanx27.github.io/), [Jiantao Gao*](https://github.com/Gao-JT), [Chaoda Zheng*](https://github.com/Ghostish), Chao Zheng, Ruimao Zhang, Shuguang Cui, [Zhen Li*](https://mypage.cuhk.edu.cn/academics/lizhen/), "*2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds*", ECCV 2022 [[arxiv]](https://arxiv.org/pdf/2207.04397v2.pdf).
 ![image](figures/2DPASS.gif)

If you find our work useful in your research, please consider citing:
```latex
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      booktitle={ECCV}
}
```
## News
* **2022-09-20** We release codes for SemanticKITTI single-scan and NuScenes :rocket:!
* **2022-07-03** 2DPASS is accepted at **ECCV 2022** :fire:!
* **2022-03-08** We achieve **1st** place in both single and multi-scans of [SemanticKITTI](http://semantic-kitti.org/index.html) and **3rd** place on [NuScenes-lidarseg](https://www.nuscenes.org/) :fire:! 
<p align="center">
   <img src="figures/singlescan.jpg" width="80%"> 
</p>
<p align="center">
   <img src="figures/multiscan.jpg" width="80%"> 
</p>
<p align="center">
   <img src="figures/nuscene.png" width="80%"> 
</p>

## Installation

### Requirements
- pytorch >= 1.8 
- yaml
- easydict
- [lightning](https://github.com/Lightning-AI/lightning) (tested with pytorch_lightning==1.3.8 and torchmetrics==0.5)
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) (pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==2.1.16 and cuda==11.1, pip install spconv-cu111==2.1.16)

## Data Preparation

### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder.
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
        |   └── image_2/ 
        |   |   ├── 000000.png
        |   |   ├── 000001.png
        |   |   └── ...
        |   calib.txt
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org/) and extract it.
```
./dataset/
├── 
├── ...
└── nuscenes/
    ├──v1.0-trainval
    ├──v1.0-test
    ├──samples
    ├──sweeps
    ├──maps

```

## Training
### SemanticKITTI
You can run the training with
```shell script
cd <root dir of this repo>
python main.py --log_dir 2DPASS_semkitti --config config/2DPASS-semantickitti.yaml --gpu 0
```
The output will be written to `logs/SemanticKITTI/2DPASS_semkitti` by default. 
### NuScenes
```shell script
cd <root dir of this repo>
python main.py --log_dir 2DPASS_nusc --config config/2DPASS-nuscenese.yaml --gpu 0 1 2
```

### Vanilla Training without 2DPASS
We take SemanticKITTI as an example.
```shell script
cd <root dir of this repo>
python main.py --log_dir baseline_semkitti --config config/2DPASS-semantickitti.yaml --gpu 0 --baseline_only
```

## Testing
You can run the testing with
```shell script
cd <root dir of this repo>
python main.py --config config/2DPASS-semantickitti.yaml --gpu 0 --test --num_vote 12 --checkpoint <dir for the pytorch checkpoint>
```
Here, `num_vote` is the number of views for the test-time-augmentation (TTA). We set this value to 12 as default (on a Tesla-V100 GPU), and if you use other GPUs with smaller memory, you can choose a smaller value. `num_vote=1` denotes there is no TTA used, and will cause about ~2\% performance drop.

## Model Zoo
You can download the models with the scores below from [this Google drive folder](https://drive.google.com/drive/folders/1Xy6p_h827lv8J-2iZU8T6SLFkxfoXPBE?usp=sharing).
|Model (validation)|Reported (mIoU)|Reproduced (mIoU)|Parameters|
|:---:|:---:|:---:|:---:|
|2DPASS-SemanticKITTI|69.3%|70.0%|1.9M|
|2DPASS-NuScenes|79.4%|79.5%|45.6M|

Here, we train the model for SemanticKITTI with more epochs and thus gain the higher mIoU. Note that the results on benchmarks are gained by training with additional validation set and using instance-level augmentation.

## Acknowledgements
Code is built based on [SPVNAS](https://github.com/mit-han-lab/spvnas), [Cylinder3D](https://github.com/xinge008/Cylinder3D), [xMUDA](https://github.com/valeoai/xmuda) and [SPCONV](https://github.com/traveller59/spconv).

## License
This repository is released under MIT License (see LICENSE file for details).



