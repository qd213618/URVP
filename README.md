# Unstructured Road Vanishing Point Detection Using the Convolutional Neural Networks and Heatmap Regression Method (URVP)
Full implementation of URVP in PyTorch.

## Overview
#### Unstructured road vanishing point (VP) detection is a challenging problem, especially in the field of autonomous driving. In this paper, we proposed a novel solution combining the convolutional neural network (CNN) and heatmap regression to detect unstructured road VP. The proposed algorithm first adopted a lightweight backbone, i.e., depthwise convolution modified HRNet, to extract hierarchical features of the unstructured road image. Then, three advanced strategies, i.e., multi-scale supervised learning, heatmap super-resolution, and coordinate regression techniques were utilized to carry out fast and high-precision unstructured road VP detection. The empirical results on Kong's dataset showed that our proposed approach had the highest detection accuracy in real-time compared with the state-of-the-art methods under various conditions, and achieved the highest speed of 33 fps.



## Installation
##### Environment
* pytorch >= 1.2.0
* python >= 3.5.0
##### Get code
```
git clone https://github.com/qd213618/URVP.git
cd URVP
pip3 install -r requirements.txt --user
```
##### Download PLVP dataset
```
cd data/
bash get_URVP_dataset.sh
```

## Training
##### Download pretrained weights
1. See [weights readme](weights/README.md) for detail.   
2. Download pretrained backbone wegiths from [Google Drive](to be add) or [Baidu Drive](to be add)   
3. Move downloaded file ```URVP.pth``` to ```wegihts``` folder in this project.   
##### Modify training parameters
1. Review config file ```training/params.py```   
2. Replace ```YOUR_WORKING_DIR``` to your working directory. Use for save model and tmp file.
3. Adjust your GPU device. see parallels.   
4. Adjust other parameters.   
##### Start training
```
cd training
python training.py params.py
```
##### Option: Visualizing training
```
#  please install tensorboard in first
python -m tensorboard.main --logdir=YOUR_WORKING_DIR   
```
<p><img src="common/demo/loss_curve.png"\></p>


## Evaluate
##### Download pretrained weights
1. See [weights readme](weights/README.md) for detail.   
2. Download pretrained yolo3 full wegiths from [Google Drive]() or [Baidu Drive]()   
3. Move downloaded file ```URVP.pth``` to ```wegihts``` folder in this project.   
##### Start evaluate
```
cd evaluate
python eval_coco.py params.py
```

## Quick test
##### pretrained weights
Please download pretrained weights ```URVP.pth``` or use yourself checkpoint.   
##### Start test
```
cd test
python test_images.py params.py
```
You can got result images in output folder.   
<p align="center"><img src="common/demo/demo0.jpg"\></p>
<p align="center"><img src="common/demo/demo1.jpg"\></p>

## Measure FPS
##### pretrained weights
Please download pretrained weights ```URVP.pth``` or use yourself checkpoint.   
##### Start test
```
cd test
python test.py params.py
```
##### Results
* Test in 2080Ti GPU with different input size and batch size.   
CPU Results

|    Methods   | Mean error | CPU Running Time (s) |
|:------------:|:----------:|:--------------------:|
| Kong (Gabor) |  0.040639  |        20.1021       |
|  Kong (gLoG) |  0.051556  |        21.213        |
|   Moghadam   |  0.063407  |        0.2423        |
|     Yang     |  0.045931  |         0.752        |
|   Proposed   |  0.034875  |        0.2024        |

GPU Results
|  Backbones | Number of images with NormDist error <= 0.01  | Number of images with NormDist error > 0.1 | GPU-speed | CPU-speed |
|:----------:|:---------------------------------------------:|:------------------------------------------:|:---------:|:---------:|
|     Hg4    |                      192                      |                     113                    | 23.04 fps |  2.02 fps |
|  HRNet-48  |                      205                      |                     115                    | 29.15 fps |  2.90 fps |
| HRNet-48-M |                      207                      |                     106                    | 33.05 fps |  4.94 fps |
## Credit
```
@article{liu2020unstructured,
  title={Unstructured Road Vanishing Point Detection Using the Convolutional Neural Network and Heatmap Regression},
  author={Liu, Yin-Bo and Zeng, Ming and Meng, Qing-Hao},
  journal={arXiv preprint arXiv:2006.04691},
  year={2020}
}
```

## Reference
* [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3): Thanks for YOLO loss code
# PL4VP
