import argparse
import os
import cv2
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
import numpy as np

def on_load_checkpoint(checkpoint):
    keys_list = list(checkpoint['network_state_dict'].keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint['network_state_dict'][deal_key] = checkpoint['network_state_dict'][key]
            del checkpoint['network_state_dict'][key]
    return checkpoint

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_uncolorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

class StandardizeData(torch.nn.Module):
    def __init__(self, mode='mean_robust'):
        super(StandardizeData, self).__init__()
        if mode == 'mar':
            self.fn = self.__masked_mean_robust_standardization__
        elif mode == 'md':
            self.fn = self.__masked_median_standardization__
        elif mode == 'ma':
            self.fn = self.__masked_mean_standardization__

    @staticmethod
    def __masked_mean_robust_standardization__(depth, mask, eps=1e-6):
        mask_num = torch.sum(mask, dim=(1, 2, 3))
        mask_num[mask_num == 0] = eps
        depth_mean = (torch.sum(depth * mask, dim=(1, 2, 3)) / mask_num).view(depth.shape[0], 1, 1, 1)
        depth_std = torch.sum(torch.abs((depth - depth_mean) * mask), dim=(1, 2, 3)) / mask_num
        return depth_mean, depth_std.view(depth.shape[0], 1, 1, 1) + eps

    @staticmethod
    def __masked_mean_standardization__(depth, mask, eps=1e-6):
        mask_num = torch.sum(mask, dim=(1, 2, 3))
        mask_num[mask_num == 0] = eps
        depth_mean = (torch.sum(depth * mask, dim=(1, 2, 3)) / mask_num).view(depth.shape[0], 1, 1, 1)
        depth_std = torch.sqrt(torch.sum(((depth - depth_mean) * mask) ** 2, dim=(1, 2, 3)) / mask_num + eps)
        return depth_mean, depth_std.view(depth.shape[0], 1, 1, 1) + eps

    @staticmethod
    def __masked_median_standardization__(depth, mask, eps=1e-6):
        mask_num = torch.sum(mask, dim=(1, 2, 3))
        depth_median = torch.zeros_like(mask_num)
        depth_std = torch.zeros_like(mask_num)
        for i in range(depth.shape[0]):
            if mask_num[i] != 0.0:
                temp_depth = depth[i]
                depth_median[i] = torch.median(temp_depth[torch.nonzero(temp_depth, as_tuple=True)])
                depth_std[i] = torch.sum(torch.abs((temp_depth - depth_median[i]) * mask[i])) / mask_num[i]
        return depth_median.view(depth.shape[0], 1, 1, 1), depth_std.view(depth.shape[0], 1, 1, 1) + eps

    def forward(self, depth, gt, mask_hole):
        t_d, s_d = self.fn(depth, mask_hole)
        t_g, s_g = self.fn(gt, mask_hole)
        sta_depth = (depth - t_d) / s_d
        sta_gt = (gt - t_g) / s_g
        return sta_depth, sta_gt


class DataGenerator(object):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def generate_data(self, all_data):
        rgb = all_data[0][0].cuda(non_blocking=True)
        gt = all_data[0][1].cuda(non_blocking=True)
        point_map = all_data[0][2].cuda(non_blocking=True)
        hole_gt = all_data[0][3].cuda(non_blocking=True)
        hole_point = all_data[1].cuda(non_blocking=True)

        hole_point[point_map == 0] = 0
        point_map *= hole_point

        return rgb, gt, point_map, (hole_gt, hole_point)


def min_max_norm(depth):
    max_value = torch.max(depth)
    min_value = torch.min(depth)
    norm_depth = (depth - min_value) / (max_value - min_value)
    return norm_depth


def save_img(
        data: Tensor,
        file_name: Path,
) -> None:
    data = data.squeeze()

    if data.ndim == 3:
        ndarr = data.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    else:
        ndarr = data.squeeze(0).mul(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        ndarr = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
    cv2.imwrite(str(file_name), ndarr)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DDPutils(object):
    @staticmethod
    def setup(rank, world_size, port):
        # rank: the serial number  of GPU
        # world_size: the number of GPUs

        # environment setting: localhost is the ip of local，6005 is interface number
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

        # initializing
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    @staticmethod
    def run_demo(demo_fn, world_size):
        # demo_fn: name of your main function, like "train"
        # world_size: the number of GPUs
        mp.spawn(demo_fn,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()
