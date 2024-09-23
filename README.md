## GelRoller
### GelRoller: A Rolling Vision-based Tactile Sensor for Large Surface Reconstruction Using Self-Supervised Photometric Stereo Method
This is the PyTorch implementation of [GelRoller](https://ieeexplore.ieee.org/document/10610417/authors#authors) (ICRA2024).

GelRoller introduces two key features:

1. **Cylindrical Vision-based Tactile Sensing**: GelRoller is designed as a cylindrical sensor capable of swiftly and continuously sensing large surface conditions by rolling over them. It enables efficient 3D reconstruction of large surfaces and facilitates defect detection.

2. **Self-Supervised Photometric Stereo (PS)**: This novel method overcomes the limitations of traditional calibration-based PS techniques, such as:

   (1) Sensitivity to varying lighting conditions;

   (2) Challenges in handling curved surfaces;

   (3) The time-consuming process of repeatedly pressing different areas on large surfaces.

By leveraging GelRoller, users can quickly sense large surfaces and reconstruct the 3D shape of the contact region using just a single input image.


## Demo
Here are some sample results (~10MB gif for each) of our GelRoller sensor equipped with the self-supervised PS method.
### Local Contact Region Reconstruction
![local_region](demo/local_region.gif)
### Large Surface Reconstrction
![large_surface](demo/large_surface.gif)
### Swiftly Fabric Sensing
![fabric_sensing](demo/fabric_sensing.gif)
### Fabric Defect Detection
![defect_detection](demo/defect_detection.gif)

## Installation

For installation, please run

```sh
cd GelRoller
conda create --name GelRoller python=3.8
pip3 install -r requirements.txt
```

## Notes
1. The self-supervised PS network takes a single input image and its corresponding MASK, training over many epochs to minimize the loss function. Upon completion of the training process, the output is the surface normals for the input image. The MASK image is generated by comparing the input image to the original image (which is free from any objects).

2. The key aspect of the self-supervised PS method is its utilization of the background surface normals. Although the surface normals in the contact region may vary when interacting with different objects, the surface normals of the background portion remain constant.

   By leveraging this key feature, the self-supervised PS method constrains the background portion of the estimated surface normals to match the original background surface normals, as outlined in Eq. (6) in the paper.

   To implement this method, it is essential to first obtain the ground truth background surface normals. Thanks to the geometric features of GelRoller, these can be easily obtained, as shown in Fig. (5) in the paper.

If you wish to apply this method to other types of vision-based tactile sensors, you will need to calculate the ground truth background surface normals specific to that sensor. For instance, when using GelSight (which has a flat surface), you can just set the background surface normals to zero.

## Training and Evaluation

You can simply use the demo in GelRoller/data and configure the settings in configs/roller.yml, specifying which data you want to use (dataset.data_path) and the path where you want to save the results (experiment.log_path).

Then, run the following command in your GelRoller environment:

```sh
python train.py
```

The reconstructed surface normals will be saved in your specified experiment.log_path.

Once you have obtained the surface normals, you can utilize the fast Poisson solver to get the desired surface depth information.


## BibTex
If you find this repo useful in your research, please consider citing:

```
@inproceedings{inproceedings,
author = {Zhang, Zhiyuan and Ma, Huan and Zhou, Yulin and Ji, Jingjing and Yang, Hua},
year = {2024},
month = {05},
pages = {7961-7967},
title = {GelRoller: A Rolling Vision-based Tactile Sensor for Large Surface Reconstruction Using Self-Supervised Photometric Stereo Method},
doi = {10.1109/ICRA57147.2024.10610417}
}
```


## Acknowledgement
This repository, during construction, referenced the code of [SCPS-NIR](https://github.com/junxuan-li/SCPS-NIR). We sincerely thank the authors for open-sourcing the codebase!