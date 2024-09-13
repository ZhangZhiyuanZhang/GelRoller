import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Model
class Light_Model(nn.Module):
    def __init__(self, num_rays, light_init, requires_grad=True):
        super(Light_Model, self).__init__()
        light1_pos_xy = light_init[0][:, 0:2].clone().detach()
        light1_pos_z = light_init[0][:, 2:3].clone().detach()
        light1_intensity = light_init[1][:, 0:3].clone().detach()

        light2_pos_xy = light_init[0][:, 3:5].clone().detach()
        light2_pos_z = light_init[0][:, 5:6].clone().detach()
        light2_intensity = light_init[1][:, 3:6].clone().detach()

        light3_pos_xy = light_init[0][:, 6:8].clone().detach()
        light3_pos_z = light_init[0][:, 8:9].clone().detach()
        light3_intensity = light_init[1][:, 6:9].clone().detach()

        beta = light_init[2].clone().detach()

        self.light1_pos_xy = nn.Parameter(light1_pos_xy.float(), requires_grad=requires_grad)
        self.light1_pos_z = nn.Parameter(light1_pos_z.float(), requires_grad=requires_grad)
        self.light1_intensity = nn.Parameter(light1_intensity.float(), requires_grad=requires_grad)

        self.light2_pos_xy = nn.Parameter(light2_pos_xy.float(), requires_grad=requires_grad)
        self.light2_pos_z = nn.Parameter(light2_pos_z.float(), requires_grad=requires_grad)
        self.light2_intensity = nn.Parameter(light2_intensity.float(), requires_grad=requires_grad)

        self.light3_pos_xy = nn.Parameter(light3_pos_xy.float(), requires_grad=requires_grad)
        self.light3_pos_z = nn.Parameter(light3_pos_z.float(), requires_grad=requires_grad)
        self.light3_intensity = nn.Parameter(light3_intensity.float(), requires_grad=requires_grad)

        self.beta = nn.Parameter(beta.float(), requires_grad=requires_grad)

        self.num_rays = num_rays

    def forward(self, idx):
        num_rays = self.num_rays
        out1_lp = torch.cat([self.light1_pos_xy[idx], -torch.abs(self.light1_pos_z[idx])], dim=-1)
        out1_lp = out1_lp.repeat(1, num_rays, 1)
        out1_lp = out1_lp.view(-1, 3)  # (1*num_rays, 3)
        out1_li = torch.abs(self.light1_intensity[idx])[:, None, :]  # (1, 1, 3)
        out1_li = out1_li.repeat(1, num_rays, 1)
        out1_li = out1_li.view(-1, 3)  # (1*num_rays, 3)

        out2_lp = torch.cat([self.light2_pos_xy[idx], -torch.abs(self.light2_pos_z[idx])], dim=-1)
        out2_lp = out2_lp.repeat(1, num_rays, 1)
        out2_lp = out2_lp.view(-1, 3)  # (1*num_rays, 3)
        out2_li = torch.abs(self.light2_intensity[idx])[:, None, :]  # (1, 1, 3)
        out2_li = out2_li.repeat(1, num_rays, 1)
        out2_li = out2_li.view(-1, 3)  # (1*num_rays, 3)

        out3_lp = torch.cat([self.light3_pos_xy[idx], -torch.abs(self.light3_pos_z[idx])], dim=-1)
        out3_lp = out3_lp.repeat(1, num_rays, 1)
        out3_lp = out3_lp.view(-1, 3)  # (1*num_rays, 3)
        out3_li = torch.abs(self.light3_intensity[idx])[:, None, :]  # (1, 1, 3)
        out3_li = out3_li.repeat(1, num_rays, 1)
        out3_li = out3_li.view(-1, 3)  # (1*num_rays, 3)

        out_beta = torch.abs(self.beta[idx])[:, None, :]
        out_beta = out_beta.repeat(1, num_rays, 1)
        out_beta = out_beta.view(-1, 3)

        out_lp = torch.cat([out1_lp, out2_lp, out3_lp], dim=-1)  # [num_rays, 9]
        out_li = torch.cat([out1_li, out2_li, out3_li], dim=-1)  # [num_rays, 9]

        return out_lp, out_li, out_beta

    def get_light_from_idx(self, idx):
        out_lp_r, out_li_r, out_beta_r = self.forward(idx)
        return out_lp_r, out_li_r, out_beta_r

    def get_all_lights(self):
        with torch.no_grad():
            light1_pos_xy = self.light1_pos_xy
            light1_pos_z = -torch.abs(self.light1_pos_z)
            light1_intensity = torch.abs(self.light1_intensity)
            out1_lp = torch.cat([light1_pos_xy, light1_pos_z], dim=-1)

            light2_pos_xy = self.light2_pos_xy
            light2_pos_z = -torch.abs(self.light2_pos_z)
            light2_intensity = torch.abs(self.light2_intensity)
            out2_lp = torch.cat([light2_pos_xy, light2_pos_z], dim=-1)

            light3_pos_xy = self.light3_pos_xy
            light3_pos_z = -torch.abs(self.light3_pos_z)
            light3_intensity = torch.abs(self.light3_intensity)
            out3_lp = torch.cat([light3_pos_xy, light3_pos_z], dim=-1)

            out_lp = torch.cat([out1_lp, out2_lp, out3_lp], dim=-1)  # [1, 9]
            light_intensity = torch.cat([light1_intensity, light2_intensity, light3_intensity], dim=-1)  # [1, 9]
            beta = self.beta

            return out_lp, light_intensity, beta


class Light_Model_CNN(nn.Module):
    def __init__(
            self,
            num_layers=3,
            hidden_size=64,
            batchNorm=False
    ):
        super(Light_Model_CNN, self).__init__()

        self.conv1 = conv_layer(batchNorm, 4, 64, k=3, stride=2, pad=1, afunc='LReLU')
        self.conv2 = conv_layer(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = conv_layer(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = conv_layer(batchNorm, 128, 128, k=3, stride=2, pad=1)
        self.conv5 = conv_layer(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv6 = conv_layer(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv7 = conv_layer(batchNorm, 256, 256, k=3, stride=1, pad=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(256, hidden_size)] + [nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 1)])
        self.intens_linear = nn.Linear(hidden_size, 12)
        self.pos_linear = nn.Linear(hidden_size, 9)

    def forward(self, inputs):
        x = inputs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        for i, l in enumerate(self.dir_linears):
            out = self.dir_linears[i](out)
            out = self.relu(out)
        intens = self.intens_linear(out)
        pos = self.pos_linear(out)

        light1 = torch.cat([pos[:, 0:3], intens[:, 0:3]], dim=1)
        light2 = torch.cat([pos[:, 3:6], intens[:, 3:6]], dim=1)
        light3 = torch.cat([pos[:, 6:9], intens[:, 6:9]], dim=1)
        beta = intens[:, 9:]

        light1_pos_xy = light1[:, :2]
        light1_pos_z = -torch.abs(light1[:, 2:3])
        light1_intensity = torch.abs(light1[:, 3:])

        light2_pos_xy = light2[:, :2]
        light2_pos_z = -torch.abs(light2[:, 2:3])
        light2_intensity = torch.abs(light2[:, 3:])

        light3_pos_xy = light3[:, :2]
        light3_pos_z = -torch.abs(light3[:, 2:3])
        light3_intensity = torch.abs(light3[:, 3:])

        out1_lp = torch.cat([light1_pos_xy, light1_pos_z], dim=-1)
        out1_li = self.relu(light1_intensity)  # (1, 3)

        out2_lp = torch.cat([light2_pos_xy, light2_pos_z], dim=-1)
        out2_li = self.relu(light2_intensity)  # (1, 3)

        out3_lp = torch.cat([light3_pos_xy, light3_pos_z], dim=-1)
        out3_li = self.relu(light3_intensity)  # (1, 3)

        out_lp = torch.cat([out1_lp, out2_lp, out3_lp], dim=-1)  # [1, 9]
        out_li = torch.cat([out1_li, out2_li, out3_li], dim=-1)  # [1, 9]
        out_beta = torch.sigmoid(beta)  # [1, 3]

        outputs = {'pos': out_lp, 'ints': out_li, 'beta': out_beta}
        return outputs

    def set_images(self, num_rays, images, device):
        self.num_rays = num_rays
        self.images = images
        self.device = device
        return

    def get_light_from_idx(self, idx):
        if hasattr(self, 'explicit_model'):
            out_lp_r, out_li_r, out_beta_r = self.explicit_model(idx)
        else:
            x = self.images[idx].to(self.device)
            outputs = self.forward(x)
            out_lp, out_li, out_beta = outputs['pos'], outputs['ints'], outputs['beta']

            num_rays = self.num_rays
            out_lp_r = out_lp[:, None, :].repeat(1, num_rays, 1)  # (1, num_rays, 9)
            out_lp_r = out_lp_r.view(-1, 9)  # (1*num_rays, 9)

            out_li_r = out_li[:, None, :].repeat(1, num_rays, 1)
            out_li_r = out_li_r.view(-1, 9)  # (1*num_rays, 9)

            out_beta_r = out_beta[:, None, :].repeat(1, num_rays, 1)
            out_beta_r = out_beta_r.view(-1, 3)  # (1*num_rays, 9)
        return out_lp_r, out_li_r, out_beta_r

    def get_all_lights(self):
        if hasattr(self, 'explicit_model'):
            out_lp, out_li, out_beta = self.explicit_model.get_all_lights()
        else:
            inputs = self.images.to(self.device)
            outputs = self.forward(inputs)
            out_lp, out_li, out_beta = outputs['pos'], outputs['ints'], outputs['beta']
        return out_lp, out_li, out_beta

    def init_explicit_lights(self, explicit_position=False, explicit_intensity=False):
        if explicit_position or explicit_intensity:
            light_init = self.get_all_lights()
            self.explicit_intensity = explicit_intensity
            self.explicit_position = explicit_position
            self.explicit_model = Light_Model(self.num_rays, light_init, requires_grad=True)
        else:
            return


def activation(afunc='LReLU'):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=True)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown activation function')


def conv_layer(batchNorm, cin, cout, k=3, stride=1, pad=-1, afunc='LReLU'):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    mList = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True)]
    if batchNorm:
        print('=> convolutional layer with batchnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)


class NeRFModel_Separate(torch.nn.Module):
    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=3,
            num_encoding_fn_input1=10,
            num_encoding_fn_input2=0,
            include_input_input1=2,  # denote images coordinates (ix, iy)
            include_input_input2=0,  # denote lighting direcions (lx, ly, lz)
            valid_region=None,
    ):
        super(NeRFModel_Separate, self).__init__()
        self.dim_ldir = include_input_input2 * (1 + 2 * num_encoding_fn_input2)
        self.dim_ixiy = include_input_input1 * (1 + 2 * num_encoding_fn_input1)
        self.skip_connect_every = skip_connect_every + 1

        # Layers for Material Map
        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        # Layers for Normal Map
        self.layers_xyz_normal = torch.nn.ModuleList()
        self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy + 3, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy + 3 + hidden_size, hidden_size))
            else:
                self.layers_xyz_normal.append(torch.nn.Linear(hidden_size, hidden_size))

        self.relu = torch.nn.functional.leaky_relu
        # self.relu = torch.nn.functional.relu
        self.valid_region = valid_region
        self.idxp = torch.where(self.valid_region > 0.5)

        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_ldir, hidden_size // 2))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))

        self.fc_diff = torch.nn.Linear(hidden_size // 2, 1)

        self.fc_normal_xy = torch.nn.Linear(hidden_size, 2)
        self.fc_normal_z = torch.nn.Linear(hidden_size, 1)

    def forward(self, input):
        xyz = input[..., : self.dim_ixiy]
        xyz_rgb = input[..., : self.dim_ixiy + 3]

        # Compute Normal Map
        x = xyz_rgb
        for i in range(len(self.layers_xyz_normal)):
            if i == self.skip_connect_every:
                x = self.layers_xyz_normal[i](torch.cat((xyz_rgb, x), -1))
            else:
                x = self.layers_xyz_normal[i](x)
            x = self.relu(x)
        normal_xy = self.fc_normal_xy(x)
        normal_z = -torch.abs(self.fc_normal_z(x))  # n_z is always facing camera
        normal = torch.cat([normal_xy, normal_z], dim=-1)
        normal = F.normalize(normal, p=2, dim=-1)

        # Compute Material Map
        x = xyz
        for i in range(len(self.layers_xyz)):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        if self.dim_ldir > 0:
            light_xyz = input[..., -self.dim_ldir:]
            feat = torch.cat([feat, light_xyz], dim=-1)
        x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_dir)):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        diff = torch.abs(self.fc_diff(x))

        return normal, diff


def Fresnel_Factor(light, half, view, normal):
    c = torch.abs((light * half).sum(dim=-1))
    g = torch.sqrt(1.33 ** 2 + c ** 2 - 1)
    temp = (c * (g + c) - 1) ** 2 / (c * (g - c) + 1) ** 2
    f = (g - c) ** 2 / (2 * (g + c) ** 2) * (1 + temp)
    return f


def totalVariation(image, valid_region, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * valid_region[1:, :] * valid_region[:-1, :]
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * valid_region[:, 1:] * valid_region[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var


def totalVariation_L2(image, valid_region, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * valid_region[1:, :] * valid_region[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * valid_region[:, 1:] * valid_region[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var