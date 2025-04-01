import sys

import torch
from PIL import Image
import numpy as np
import random
import os
from torchvision.transforms import ToPILImage
from pathlib import Path
from torch import Tensor
sys.path.append('../..')


def depth_read(filename: Path) -> Tensor:
    # Read Depth and converts to 0-1
    str_file = str(filename)
    if '.png' in str_file:
        data = Image.open(filename)
        depth = torch.from_numpy(depth).unsqueeze(0)
        data.close()
    else:
        print(f'Depth reading error! Unseen Type, We want png file.{str_file}')
    return depth


def make_dir(File_Path):
    if not os.path.exists(File_Path):
        
        # 目录不存在，进行创建操作
        os.makedirs(File_Path)  # 使用os.makedirs()方法创建多层目录


class RGBPReader(object):
    def __init__(self):
        self.rel = False

    @staticmethod
    def __read_rgbp__(rgb_path, point_path):
        rgb_img = torch.from_numpy(
            (np.array(Image.open(rgb_path), dtype=float) / 255.).transpose((2, 0, 1))
        )
        if '/point_0.0/' in point_path:
            point_img = torch.zeros((1, 1, rgb_img.shape[1], rgb_img.shape[2]))
        else:
            point_img = depth_read(point_path).unsqueeze(0)
            point_img = torch.nn.functional.interpolate(
                point_img, (rgb_img.shape[1], rgb_img.shape[2]))
        return rgb_img.unsqueeze(0).to(torch.float32), point_img.to(torch.float32)

    def read_rgbp(self, rgb_path, point_path):
        self.rel = False
        rgb_img, point_img = self.__read_rgbp__(rgb_path, point_path)
        hole_point = torch.ones_like(point_img)
        hole_point[point_img == 0] = 0
        if torch.nonzero(point_img).size(0) == 0:
            self.rel = True

        return rgb_img, point_img, hole_point

    def get_hole(self, rgb_path, point_path, save_dir):
        depth = torch.from_numpy(np.array(Image.open(point_path), dtype=float))
        if torch.count_nonzero(depth) / depth.numel() >= 0.6 and min(depth.shape) >= 320:
            make_dir(str(Path(save_dir).parent))
            hole_point = torch.ones_like(depth)
            hole_point[depth == 0] = 0
            unloader = ToPILImage()
            image = hole_point.cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save(save_dir)
    @staticmethod
    def __min_max_norm__(depth):
        max_value = np.max(depth)
        min_value = np.min(depth)
        norm_depth = (depth - min_value) / (max_value - min_value + 1e-6)
        return norm_depth

    def adjust_domain(self, gen_depth):
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        if self.rel:
            gen_depth = self.__min_max_norm__(gen_depth)
        depth = np.clip(gen_depth * 65535., 0, 65535).astype(np.int32)
        return depth


class DepthEvaluation(object):
    @staticmethod
    
    def irmse(depth, ground_truth, min_depth=1e-3):
        # 定义一个最小深度值，避免除以0或处理极小的深度值导致的数值不稳定
        inverse_depth = np.where(depth > min_depth, 1.0 / depth, 0)
        inverse_ground_truth = np.where(ground_truth > min_depth, 1.0 / ground_truth, 0)
        
        # 计算残差
        residual = ((inverse_depth - inverse_ground_truth)) ** 2
        
        # 对于真实深度值小于等于最小深度值的情况，将残差设为0
        residual[ground_truth <= min_depth] = 0.
        
        # 计算iRMSE
        value = np.sqrt(np.sum(residual) / np.count_nonzero(ground_truth > min_depth))
        
        return value
    @staticmethod
    def rmse(depth, ground_truth):
        residual = (depth - ground_truth) ** 2
        residual[ground_truth == 0.] = 0.
        value = np.sqrt(np.sum(residual) / np.count_nonzero(ground_truth))
        return value


    @staticmethod
    def imae(depth, ground_truth, min_depth=1e-6):
        # 计算逆深度
        inverse_depth = np.where(depth > min_depth, 1.0 / depth, 0)
        inverse_ground_truth = np.where(ground_truth > min_depth, 1.0 / ground_truth, 0)
        
        # 计算残差的绝对值
        residual = np.abs(inverse_depth - inverse_ground_truth)
        
        # 对于真实深度值小于等于最小深度值的情况，将残差设为0
        residual[ground_truth <= min_depth] = 0.
        
        # 计算IMAE
        value = np.sum(residual) / np.count_nonzero(ground_truth > min_depth)
        
        return value

    
    @staticmethod
    def mae(depth, ground_truth):
        residual = np.abs(depth - ground_truth)
        residual[ground_truth == 0.] = 0.
        value = np.sum(residual) / np.count_nonzero(ground_truth)
        return value

    @staticmethod
    def absRel(depth, ground_truth):
        diff = depth * 1.0 - ground_truth * 1.0
        diff[ground_truth == 0] = 0.
        rel = np.sum(abs(diff) / (ground_truth + 1e-6)) / np.count_nonzero(ground_truth)
        return rel

    @staticmethod
    def threshold_accuracy(depth, ground_truth):
        ratio1 = depth * 1.0 / (ground_truth + 1e-6)
        ratio2 = ground_truth * 1.0 / (depth + 1e-6)
        count125 = 0
        count1252 = 0
        count1253 = 0
        num = np.count_nonzero(ground_truth)

        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if ground_truth[i, j] != 0:
                    max_ratio = max(ratio1[i, j], ratio2[i, j])
                    if max_ratio < 1.25:
                        count125 += 1
                        count1252 += 1
                        count1253 += 1
                    elif max_ratio < 1.25 ** 2:
                        count1252 += 1
                        count1253 += 1
                    elif max_ratio < 1.25 ** 3:
                        count1253 += 1

        thres_125 = count125 * 1.0 / num
        thres_1252 = count1252 * 1.0 / num
        thres_1253 = count1253 * 1.0 / num

        return thres_125, thres_1252, thres_1253

    @staticmethod
    def silog(depth, ground_truth):
        temp_depth = depth.copy()
        temp_gt = ground_truth.copy()
        mask = np.ones_like(temp_depth)
        mask[ground_truth == 0.0] = 0.0
        temp_depth[mask == 0.0] = 1.0
        temp_gt[mask == 0.0] = 1.0
        log_res = np.log(temp_depth) - np.log(temp_gt)
        value = np.sum(log_res ** 2) / np.count_nonzero(mask) + (np.sum(log_res) ** 2) / (
            np.count_nonzero(mask) ** 2)
        return value

    @staticmethod
    def starmse(depth, ground_truth):
        mask = np.ones_like(depth)
        # print(f'depth.shape =  {depth.shape}')
        # print(f'mask.shape = {mask.shape}')
        # print(f'ground_truth.shape = {ground_truth.shape}')
        mask[ground_truth == 0] = 0
        number_valid = np.sum(mask) + 1e-6
        # sta depth
        mean_d = np.sum(depth * mask) / number_valid
        std_d = np.sum(np.abs(depth - mean_d) * mask) / number_valid
        sta_dep = (depth - mean_d) / (std_d + 1e-6)
        # sta gt
        mean_gt = np.sum(ground_truth * mask) / number_valid
        std_gt = np.sum(np.abs(ground_truth - mean_gt) * mask) / number_valid
        sta_gt = (ground_truth - mean_gt) / (std_gt + 1e-6)

        sta_rmse = np.sqrt(np.sum(mask * (sta_dep - sta_gt) ** 2) / number_valid)

        return sta_rmse

    @staticmethod
    def oe(depth, ground_truth):
        oe_comp = OrdinalError()
        value = oe_comp.error(depth, ground_truth)
        return value


class OrdinalError(object):
    def __init__(self, t=0.01):
        super(OrdinalError, self).__init__()
        self.t = t
        self.eps = 1e-8

    def __ordinal_label__(self, point_a, point_b):
        ratio = point_a * 1.0 / (point_b + self.eps)
        if ratio >= (1 + self.t):
            l = 1
        elif ratio <= (1 / (1 + self.t)):
            l = -1
        else:
            l = 0
        return l

    def error(self, pre_d, gt):
        nonzero_pixels_index = np.nonzero(gt)
        pixel_num = nonzero_pixels_index[0].shape[0]
        select_num = int(0.5 * pixel_num)

        select_pixels_index_a = [i for i in range(pixel_num)]
        select_pixels_index_b = [i for i in range(pixel_num)]
        random.seed(1)
        random.shuffle(select_pixels_index_a)
        point_a_index_list = select_pixels_index_a[:select_num]
        random.seed(2)
        random.shuffle(select_pixels_index_b)
        point_b_index_list = select_pixels_index_b[:select_num]

        loss = 0
        for i in range(select_num):
            point_a_index = point_a_index_list[i]
            point_b_index = point_b_index_list[i]

            point_a_x = nonzero_pixels_index[0][point_a_index]
            point_a_y = nonzero_pixels_index[1][point_a_index]

            point_b_x = nonzero_pixels_index[0][point_b_index]
            point_b_y = nonzero_pixels_index[1][point_b_index]
            ordinal_pre = self.__ordinal_label__(
                pre_d[point_a_x, point_a_y], pre_d[point_b_x, point_b_y])
            ordinal_gt = self.__ordinal_label__(
                gt[point_a_x, point_a_y], gt[point_b_x, point_b_y])
            if ordinal_gt != ordinal_pre:
                loss += 1

        loss = loss * 1.0 / select_num

        return loss
