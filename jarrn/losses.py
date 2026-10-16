"""Joint absolute and relative losses from the original training objective."""

import torch
from torch import nn
from torch.nn import functional as F


def masked_l1(output, target, mask):
    return ((output - target).abs() * mask).sum() / (mask.sum() + 1e-6)


def standardize_pair(output, target, mask, mode="mar"):
    def standardize(tensor):
        count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        if mode == "md":
            center = torch.stack([
                values[valid].median() if valid.any() else values.new_tensor(0.0)
                for values, valid in zip(tensor, mask.bool())
            ]).reshape(-1, 1, 1, 1)
        else:
            center = (tensor * mask).sum(dim=(1, 2, 3), keepdim=True) / count
        residual = (tensor - center) * mask
        if mode == "ma":
            scale = ((residual.square().sum(dim=(1, 2, 3), keepdim=True) / count) + 1e-6).sqrt()
        else:
            scale = residual.abs().sum(dim=(1, 2, 3), keepdim=True) / count + 1e-6
        return (tensor - center) / scale

    return standardize(output), standardize(target)


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()
        self.scales = scales
        self.register_buffer("sobel_x", torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def forward(self, output, target, mask):
        residual = (output - target) * mask
        total = residual.new_tensor(0.0)
        for i in range(self.scales):
            current = residual if i == 0 else F.interpolate(residual, scale_factor=2 ** -i, mode="nearest")
            if min(current.shape[-2:]) < 3:
                break
            gx = F.conv2d(current, self.sobel_x.to(current.dtype))
            gy = F.conv2d(current, self.sobel_y.to(current.dtype))
            total = total + gx.abs().sum() + gy.abs().sum()
        return total / (mask.sum() + 1e-6)


class JointLoss(nn.Module):
    def __init__(self, mode="mar"):
        super().__init__()
        self.mode = mode
        self.gradient = MultiScaleGradientLoss()

    def forward(self, prediction, target, valid_mask, sparse_mask):
        absolute = masked_l1(prediction, target, sparse_mask)
        relative_prediction, relative_target = standardize_pair(prediction, target, valid_mask, self.mode)
        relative = masked_l1(relative_prediction, relative_target, valid_mask)
        gradient = self.gradient(relative_prediction, relative_target, valid_mask)
        total = absolute + relative + 0.5 * gradient
        return total, {"absolute": absolute.detach(), "relative": relative.detach(), "gradient": gradient.detach()}
