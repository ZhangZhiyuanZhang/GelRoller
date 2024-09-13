import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv


class Data_Loader(Dataset):
    def __init__(self, data_dict, data_len=1, mode='training', shadow_threshold=0.0):
        self.images = torch.tensor(data_dict['images'], dtype=torch.float32)  # (num_images, height, width, channel)

        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)  # 1 for contact region, 0 for background
        self.valid_region = torch.tensor(data_dict['valid_region'], dtype=torch.float32)
        self.gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)

        self.num_images = self.images.size(0)
        self.height = self.images.size(1)
        self.width = self.images.size(2)
        valids = self.valid_region[None, ...].repeat((self.num_images, 1, 1))  # (num_images, height, width)

        self.valid_idx = torch.where(valids > 0.5)
        temp_idx = torch.where(self.valid_region > 0.5)
        self.valid_rgb = self.images[self.valid_idx]
        self.gt_mask = self.mask[temp_idx]
        self.valid_normal = self.gt_normal[temp_idx]

        maxn = max(self.width, self.height)

        # normalize pixels' x and y
        self.input_xy = torch.stack([temp_idx[1] / maxn, temp_idx[0] / maxn], dim=-1)
        self.mean_xy = self.input_xy.mean(dim=0, keepdim=True)
        self.input_xy = self.input_xy - self.mean_xy
        # Pixels' z-values are calculated based on the geometry of the GelRoller and then divided by maxn to achieve
        # the same scale as their x and y-values.
        # For a sensor with a flat contact surface, the value of pixel z can be set to zero
        self.input_z = torch.tensor(data_dict['gt_z'], dtype=torch.float32) / maxn

        self.num_valid_rays = int(self.input_xy.size(0))

        self.data_len = min(data_len, self.num_images)
        self.mode = mode

        self.valid_rgb = self.valid_rgb.view(self.num_images, -1, 3)

        self.input_iwih = torch.stack([temp_idx[1], temp_idx[0]], dim=-1)

    def __len__(self):
        if self.mode == 'testing':
            return self.data_len
        else:
            raise NotImplementedError('Dataloader mode unknown')

    def __getitem__(self, idx):
        if self.mode == 'training':
            return self.get_all_rays()
        if self.mode == 'testing':
            return self.get_testing_rays(idx)

    def get_all_rays(self):
        idx = torch.randperm(self.num_images)
        rgb = self.valid_rgb[idx]

        sample = {'input_xy': self.input_xy, 'rgb': rgb}

        return sample

    def get_testing_rays(self, ith):
        rgb = self.valid_rgb[ith]

        sample = {'input_xy': self.input_xy, 'input_z': self.input_z, 'rgb': rgb, 'item_idx': ith,
                  'gt_mask': self.gt_mask, 'gt_normal': self.valid_normal}

        return sample

    def get_valid(self):
        return self.valid_region

    def get_mean_xy(self):
        return self.mean_xy

    def get_all_valid_images(self):
        idx = torch.where(self.valid_region > 0.5)
        x_max, x_min = max(idx[0]), min(idx[0])
        y_max, y_min = max(idx[1]), min(idx[1])

        x_max, x_min = min(x_max+15, self.images.shape[1]), max(x_min-15, 0)
        y_max, y_min = min(y_max+15, self.images.shape[2]), max(y_min-15, 0)

        out_images = self.images[:, x_min:x_max, y_min:y_max, :].permute([0, 3, 1, 2])
        # out_valids = self.valid_region[x_min:x_max, y_min:y_max][None, None, ...].repeat(out_images.size(0), 1, 1, 1)
        out_masks = self.mask[x_min:x_max, y_min:y_max][None, None, ...].repeat(out_images.size(0), 1, 1, 1)
        out = torch.cat([out_images, out_masks], dim=1)
        return out  # (num_image, 4, height, width)
