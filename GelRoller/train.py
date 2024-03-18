import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
import cv2 as cv
import yaml
import argparse

from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

from model.models import totalVariation, totalVariation_L2, Light_Model_CNN, NeRFModel_Separate

from utils.cfgnode import CfgNode
from datasets.dataloader import Data_Loader
from datasets.load_roller import load_roller
from model.position_encoder import get_embedding_function

from utils.utils import writer_add_image
from utils.utils import SSIM
from utils.draw_utils import plot_lighting


def train(input_data, testing):
    input_xy = input_data['input_xy'].view(-1, 2).to(device)
    input_z = input_data['input_z'].view(-1, 1).to(device)

    gt_rgb = input_data['rgb'].view(-1, 3).to(device)
    gt_mask = input_data['gt_mask'].view(-1, 1).to(device)
    gt_normal = input_data['gt_normal'].view(-1, 3).to(device)

    embed_input = encode_fn_input1(input_xy)

    embed_input = torch.cat([embed_input, gt_rgb, gt_mask], dim=-1)

    est_normal, est_diff = model(embed_input)

    est_light_pos, est_light_intens, beta = light_model.get_light_from_idx(idx=input_data['item_idx'].to(device))

    light1_pos = est_light_pos[:, 0:3]
    light2_pos = est_light_pos[:, 3:6]
    light3_pos = est_light_pos[:, 6:9]
    light1_intensity = est_light_intens[:, 0:3]
    light2_intensity = est_light_intens[:, 3:6]
    light3_intensity = est_light_intens[:, 6:9]

    # use near point light source model
    input_xyz = torch.cat([input_xy, input_z], dim=1)  # [N, 3]
    # calculate direction between R,G,B light sources and pixels
    light1_rel_direction = light1_pos - input_xyz
    light2_rel_direction = light2_pos - input_xyz
    light3_rel_direction = light3_pos - input_xyz

    # calculate distance between R,G,B light sources and pixels
    direction_1_len = ((light1_rel_direction * light1_rel_direction).sum(dim=-1, keepdims=True)) ** 1.5
    direction_2_len = ((light2_rel_direction * light2_rel_direction).sum(dim=-1, keepdims=True)) ** 1.5
    direction_3_len = ((light3_rel_direction * light3_rel_direction).sum(dim=-1, keepdims=True)) ** 1.5

    # Rendering Equation
    render1_shading = F.relu((est_normal * light1_rel_direction).sum(dim=-1, keepdims=True))
    render1_shading = render1_shading / (direction_1_len + 1e-12)
    render1_rgb = est_diff * render1_shading
    render1_rgb = render1_rgb * light1_intensity

    render2_shading = F.relu((est_normal * light2_rel_direction).sum(dim=-1, keepdims=True))
    render2_shading = render2_shading / (direction_2_len + 1e-12)
    render2_rgb = est_diff * render2_shading
    render2_rgb = render2_rgb * light2_intensity

    render3_shading = F.relu((est_normal * light3_rel_direction).sum(dim=-1, keepdims=True))
    render3_shading = render3_shading / (direction_3_len + 1e-12)
    render3_rgb = est_diff * render3_shading
    render3_rgb = render3_rgb * light3_intensity

    # The final rendered image can be obtained as a linear combination of each component
    render_rgb = beta[0] * render1_rgb + beta[1] * render2_rgb + beta[2] * render3_rgb

    if not testing:
        # self supervised loss function
        # background photometric loss
        ph_loss = (render_rgb - gt_rgb).abs() * (1 - gt_mask)
        ph_loss = ph_loss.mean() / (1 - gt_mask).mean()
        ph_loss_val = ph_loss.item()

        # background normal loss
        norm_loss = (est_normal - gt_normal).abs() * (1 - gt_mask)
        norm_loss = norm_loss.mean() / (1 - gt_mask).mean()
        norm_loss_val = norm_loss.item()

        # contact region loss
        rgb_render = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        rgb_render[idxp] = render_rgb
        rgb_render = rgb_render.unsqueeze(0).permute(0, 3, 2, 1)
        rgb_gt = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        rgb_gt[idxp] = gt_rgb
        rgb_gt = rgb_gt.unsqueeze(0).permute(0, 3, 2, 1)
        mask_gt = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        mask_gt[idxp] = gt_mask
        mask_gt = mask_gt.unsqueeze(0).permute(0, 3, 2, 1)
        contact_loss = []
        contact_loss += [0.15 * (rgb_render - rgb_gt).abs() * mask_gt]
        contact_loss += [0.85 * SSIM(rgb_render * mask_gt, rgb_gt * mask_gt)]
        contact_loss = sum([l.mean() for l in contact_loss]) / mask_gt.mean()
        contact_loss_val = contact_loss.item()

        loss = cfg.loss.ph_factor * ph_loss + \
               cfg.loss.contact_factor * contact_loss + \
               cfg.loss.normal_factor * norm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the running loss
        cost_t = time.time() - start_t
        print(
            'epoch: %d,  ph: %.4f, contact: %.4f, norm: %.4f, '
            'cost_time: %d m %2d s' %
            (epoch, ph_loss_val, contact_loss_val, norm_loss_val, cost_t // 60, cost_t % 60))
        writer.add_scalar('Training loss', ph_loss_val, (epoch - 1) * iters_per_epoch + iter_num)
    else:
        rgb_loss = F.l1_loss(render_rgb.view(-1), gt_rgb.view(-1))
        print("Testing RGB L1: %.4f" % (rgb_loss.item() * 255.))

        if eval_idx == 1:
            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            temp_nor = est_normal.clone()
            temp_nor[..., 1:] = -temp_nor[..., 1:]
            normal_map[idxp] = (temp_nor + 1) / 2
            normal_map = normal_map.cpu().numpy()
            normal_map = (np.clip(normal_map * 255., 0, 255)).astype(np.uint8)[:, :, ::-1]
            cv.imwrite(os.path.join(log_path, 'est_normal.png'), normal_map)
            # create grid of images
            writer_add_image(os.path.join(log_path, 'est_normal.png'), epoch, writer)

            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            normal_map[idxp] = temp_nor
            normal_map = normal_map.cpu().numpy()
            np.save(os.path.join(log_path, 'est_normal.npy'), normal_map)

        diff_map = torch.ones((h, w, 1), dtype=torch.float32, device=device)
        diff_map[idxp] = est_diff / est_diff.max()
        diff_map = diff_map.cpu().numpy()
        diff_map = np.clip(diff_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path, 'est_diff.png'), diff_map)
        # create grid of images
        writer_add_image(os.path.join(log_path, 'est_diff.png'), epoch, writer)

        if eval_idx >= 2:
            log_path_rgb = os.path.join(log_path, 'rgb')
            os.makedirs(log_path_rgb, exist_ok=True)
            log_path_shading = os.path.join(log_path, 'shading')
            os.makedirs(log_path_shading, exist_ok=True)

        else:
            log_path_rgb = log_path

        ########################################################
        rgb_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        rgb_map[idxp] = render_rgb[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_rgb, 'est_rgb.png'), rgb_map)
        # create grid of images
        writer_add_image(os.path.join(log_path_rgb, 'est_rgb.png'), epoch, writer)

        # get prediction of light sources' information, direction, intensity, combination coefficient
        pred_lp, pred_li, beta = light_model.get_all_lights()
        pred_lp = pred_lp.reshape(3, 3)
        pred_li = pred_li.reshape(3, 3)

        plot_lighting(pred_lp.cpu().numpy(), pred_li.cpu().numpy(), log_path)

        # create grid of images
        writer_add_image(os.path.join(log_path_rgb, 'est_light_map.png'), epoch, writer)

        np.savetxt(os.path.join(log_path, 'est_light_position.txt'), pred_lp.cpu().numpy())
        np.savetxt(os.path.join(log_path, 'est_light_intensity.txt'), pred_li.cpu().numpy())
        np.savetxt(os.path.join(log_path, 'est_beta.txt'), beta.cpu().numpy())

    return est_normal


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="configs/roller.yml",
        help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--testing", type=str2bool,
        default=False,
        help="Enable testing mode."
    )
    parser.add_argument(
        "--cuda", type=str,
        help="Cuda ID."
    )
    parser.add_argument(
        "--quick_testing", type=str2bool,
        default=False,
        help="Enable quick_testing mode."
    )
    configargs = parser.parse_args()

    if configargs.quick_testing:
        configargs.testing = True

    # Read config file.
    configargs.config = os.path.expanduser(configargs.config)
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if cfg.experiment.randomseed is not None:
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)
    if configargs.cuda is not None:
        cfg.experiment.cuda = "cuda:" + configargs.cuda
    device = torch.device(cfg.experiment.cuda)

    log_path = os.path.expanduser(cfg.experiment.log_path)
    train_path = os.path.expanduser(cfg.dataset.data_path)
    test_path = os.path.expanduser(cfg.dataset.data_path)

    if configargs.testing:
        writer = None
    else:
        writer = SummaryWriter(log_path)  # tensorboard --logdir=runs
        copyfile(__file__, os.path.join(log_path, 'train.py'))
        copyfile(configargs.config, os.path.join(log_path, 'config.yml'))

    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))

    # Build data loader
    train_data_dict = load_roller(train_path, cfg)
    test_data_dict = load_roller(test_path, cfg)

    training_data_loader = Data_Loader(
        train_data_dict,
        data_len=300,
        mode='testing'
    )
    training_dataloader = torch.utils.data.DataLoader(training_data_loader, batch_size=batch_size,
                                                      shuffle=not configargs.testing, num_workers=0)
    valid_region = training_data_loader.get_valid().to(device)
    eval_data_len = len(training_data_loader) if configargs.testing else 1
    if configargs.quick_testing:
        eval_data_len = 1
        configargs.testing = True
    if cfg.experiment.eval_every_iter <= (end_epoch - start_epoch + 1):
        eval_data_loader = Data_Loader(
            test_data_dict,
            data_len=eval_data_len,
            mode='testing'
        )
        eval_dataloader = torch.utils.data.DataLoader(eval_data_loader, batch_size=1, shuffle=False, num_workers=0)

    # Build model
    model = NeRFModel_Separate(
        num_layers=cfg.models.nerf.num_layers,
        hidden_size=cfg.models.nerf.hidden_size,
        skip_connect_every=cfg.models.nerf.skip_connect_every,
        num_encoding_fn_input1=cfg.models.nerf.num_encoding_fn_input1,
        num_encoding_fn_input2=cfg.models.nerf.num_encoding_fn_input2,
        include_input_input1=cfg.models.nerf.include_input_input1,  # denote images coordinates (ix, iy)
        include_input_input2=cfg.models.nerf.include_input_input2,  # denote rgb latent code (lx, ly, lz)
        valid_region=valid_region,
    )
    encode_fn_input1 = get_embedding_function(num_encoding_functions=cfg.models.nerf.num_encoding_fn_input1)
    model.train()
    model.to(device)

    if cfg.models.light_model.type == 'Light_Model_CNN':
        light_model = Light_Model_CNN(
            num_layers=cfg.models.light_model.num_layers,
            hidden_size=cfg.models.light_model.hidden_size,
            batchNorm=False
        )
        light_model.train()
        light_model.to(device)
    else:
        raise NotImplementedError('Unknown light model')

    light_model.set_images(
        num_rays=np.count_nonzero(train_data_dict['valid_region']),
        images=training_data_loader.get_all_valid_images(),
        device=device,
    )
    light_model.init_explicit_lights(
        explicit_position=cfg.models.light_model.explicit_position,
        explicit_intensity=cfg.models.light_model.explicit_intensity,
    )

    params_list = [{'params': model.parameters()}]

    if hasattr(cfg.optimizer, 'light_lr') and cfg.optimizer.light_lr is not None:
        params_list.append({'params': light_model.parameters(), 'lr': cfg.optimizer.light_lr})
    else:
        params_list.append({'params': light_model.parameters()})

    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)

    # Load checkpoints
    if configargs.testing:
        cfg.models.load_checkpoint = True
        cfg.models.checkpoint_path = log_path
    if cfg.models.load_checkpoint:
        model_checkpoint_pth = os.path.expanduser(cfg.models.checkpoint_path)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found checkpoints', model_checkpoint_pth)
        ckpt = torch.load(model_checkpoint_pth, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        light_model.load_state_dict(ckpt['light_model_state_dict'])
        start_epoch = ckpt['global_step'] + 1
    if configargs.testing:
        start_epoch = 1
        end_epoch = 1
        cfg.experiment.eval_every_iter = 1
        cfg.experiment.save_every_iter = 100
    if configargs.quick_testing:
        cfg.experiment.eval_every_iter = 100000000

    if cfg.loss.rgb_loss == 'l1':
        rgb_loss_function = F.l1_loss
    elif cfg.loss.rgb_loss == 'l2':
        rgb_loss_function = F.mse_loss
    else:
        raise AttributeError('Undefined rgb loss function.')

    start_t = time.time()
    h, w = valid_region.size(0), valid_region.size(1)
    idxp = torch.where(valid_region > 0.5)
    num_rays = len(idxp[0])
    iters_per_epoch = len(training_dataloader)

    epoch = 0
    model.eval()
    with torch.no_grad():
        print('================ evaluation results===============')
        for eval_idx, eval_datain in enumerate(eval_dataloader, start=1):
            batch_size = 1
            train(input_data=eval_datain, testing=True)
        print('==================================================')
    model.train()

    for epoch in range(start_epoch, end_epoch + 1):
        for iter_num, input_data in enumerate(training_dataloader):
            if not configargs.testing:
                batch_size = int(eval(cfg.experiment.batch_size))
                output_normal_0 = train(input_data=input_data, testing=False)

        scheduler.step()

        if epoch % cfg.experiment.save_every_epoch == 0:
            savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'light_model_state_dict': light_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, savepath)
            print('Saved checkpoints at', savepath)

        if epoch % cfg.experiment.eval_every_iter == 0:
            model.eval()
            with torch.no_grad():
                print('================ evaluation results===============')
                for eval_idx, eval_datain in enumerate(eval_dataloader, start=1):
                    batch_size = 1
                    train(input_data=eval_datain, testing=True)
                print('==================================================')
            model.train()
