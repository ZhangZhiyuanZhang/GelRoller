import torch
import torch.nn.functional as F
import torchvision
import os
import cv2 as cv
import math
import torch.nn as nn


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def cal_dirs_acc(gt_l, pred_l):
    dot_product = (gt_l * pred_l).sum(-1).clamp(-1, 1)
    angular_err = torch.acos(dot_product) * 180.0 / math.pi
    l_err_mean = angular_err.mean()
    return l_err_mean.item(), angular_err


def cal_ints_acc(gt_i, pred_i):
    # Red channel:
    gt_i_c = gt_i[:, :1]
    pred_i_c = pred_i[:, :1]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio1 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Green channel:
    gt_i_c = gt_i[:, 1:2]
    pred_i_c = pred_i[:, 1:2]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio2 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Blue channel:
    gt_i_c = gt_i[:, 2:3]
    pred_i_c = pred_i[:, 2:3]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio3 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)

    ints_ratio = (ints_ratio1 + ints_ratio2 + ints_ratio3) / 3
    return ints_ratio.mean().item(), ints_ratio.mean(dim=-1)


def add_noise_light_init(ld, li, ld_noise=10, li_noise=0.1):
    if ld_noise < 0:
        new_ld = ld
    elif ld_noise == 0:
        new_ld = torch.zeros_like(ld)
        new_ld[:, -1] = -1
    else:
        new_ld = add_noise_light_direction(ld, max_degree=ld_noise)
        new_ld = F.normalize(new_ld, p=2, dim=-1)

    if li_noise < 0:
        new_li = li
    elif li_noise == 0:
        new_li = torch.ones_like(li)
    else:
        new_li = add_noise_light_intensity(li, max_diff=li_noise)
    return [new_ld, new_li]


def add_noise_light_direction(ld, max_degree):
    num_light, c = ld.shape
    new_ld = torch.zeros_like(ld)
    for i in range(num_light):
        input_ld = ld[i]
        flag = True
        while flag:
            random_ld = torch.rand(3) * 2 - 1
            random_ld[2] = -torch.abs(random_ld[2])
            random_ld = F.normalize(random_ld, p=2, dim=0)
            degree_diff = torch.arccos((random_ld * input_ld).sum().clamp(-1,1)) / math.pi * 180
            if random_ld[2] < -0.1 and degree_diff < max_degree:
                flag = False
                new_ld[i] = random_ld
    return new_ld


def add_noise_light_intensity(li, max_diff):
    num_light, c = li.shape
    new_li = li * ((torch.rand(num_light, c) * 2 - 1) * max_diff + 1)
    return new_li


def writer_add_image(file_name, epoch, writer):
    if writer is None:
        return
    img = torch.tensor(cv.imread(file_name)[:, :, ::-1] / 255.0)
    img_grid = torchvision.utils.make_grid(img.permute(2, 0, 1)[None, ...])
    basename = os.path.basename(file_name)[:-4]
    if basename == 'est_light_map':
        basename = 'est_lighting'
    writer.add_image(basename, img_grid, epoch)
    return
