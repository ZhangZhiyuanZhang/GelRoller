import cv2 as cv
import os
import numpy as np
import torch
import glob
import scipy.io as sio

"""
Normal coordinates for DiLiGenT dataset: (same for light direction)
    y    
    |   
    |  
    | 
     --------->   x
   /
  /
 z 
x direction is looking right
y direction is looking up
z direction is looking outside the image

we convert it to :
Normal coordinates for DiLiGenT dataset: (same for light direction)
     --------->   x
    |   
    |  
    | 
    y    
x direction is looking right
y direction is looking down
z direction is looking into the image

"""


def load_roller(path, cfg=None):
    images = []
    for img_file in sorted(glob.glob(os.path.join(path, "[0-9]*.png"))):
        img = cv.imread(img_file)[:, :, ::-1].astype(np.float32) / 255.
        img = cv.GaussianBlur(img, (5, 5), 0)
        images.append(img)
    images = np.stack(images, axis=0)

    valid_files = os.path.join(path, "valid.png")  # 1 for pixel to be reconstructed
    valid_region = cv.imread(valid_files, 0).astype(np.float32) / 255.

    mask_files = os.path.join(path, "mask.png")  # 1 for contact region, 0 for background
    mask = cv.imread(mask_files, 0).astype(np.float32) / 255.
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    # Obtained from geometry of GelRoller, calculated by Eq. (5, 6, 7), see getD.m
    # We crop the original image to a well-illuminated region of 449*373 pixels for reconstruction:
    # rect = [92, 108, 448, 373];
    # image_crop = imcrop(image, rect); MATLAB
    gt_normal = sio.loadmat(os.path.join(path, "bg_Normal.mat"))
    gt_normal = gt_normal['Normal_crop']

    # Obtained from geometry of GelRoller, unit: mm
    gt_z = sio.loadmat(os.path.join(path, "D.mat"))
    gt_z = -gt_z['depth_crop']  # As the positive direction of the z-axis aligns with the camera optical axis,
                                # hence, gt_z is located in the negative half-axis of the z-axis.
    f = 0.0658  # mm/pixel, To obtain this ratio, we have to put the ruler in GelRoller surface's different regions
                # multiply times and capture images to calculate the average mm/pixel
    gt_z = gt_z / f  # unit: pixel

    if hasattr(cfg.dataset, 'sparse_input_random_seed') and hasattr(cfg.dataset, 'sparse_input'):
        if cfg.dataset.sparse_input_random_seed is not None and cfg.dataset.sparse_input is not None:
            np.random.seed(cfg.dataset.sparse_input_random_seed)
            select_idx = np.random.permutation(len(images))[:cfg.dataset.sparse_input]
            print('Random seed: %d .   Selected random index: ' % cfg.dataset.sparse_input_random_seed, select_idx)
            images = images[select_idx]

    out_dict = {'images': images, 'valid_region': valid_region, 'mask': mask, 'gt_normal': gt_normal, 'gt_z': gt_z}
    return out_dict
