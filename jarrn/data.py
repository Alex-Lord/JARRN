"""Paired RGB/depth loading and sparse-depth synthesis for JARRN training."""

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def find_pairs(roots):
    """Find DIODE_rgb/DIODE_gt and HRWSI_rgb/HRWSI_gt pairs."""
    pairs = set()
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {root}")
        for rgb_dir in root.rglob("*_rgb"):
            if not rgb_dir.is_dir():
                continue
            gt_dir = rgb_dir.with_name(rgb_dir.name[:-4] + "_gt")
            for rgb in sorted(rgb_dir.rglob("*")):
                if rgb.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                gt = gt_dir / rgb.relative_to(rgb_dir).with_suffix(".png")
                if gt.is_file():
                    pairs.add((rgb, gt))
    if not pairs:
        raise ValueError("No paired RGB/depth images found under the given --data-root paths")
    return sorted(pairs)


def _read_depth(path):
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    name = str(path).lower()
    if "diode" in name and "_indoors_" in name:
        divisor = 20.0
    elif "diode" in name and "_outdoor_" in name:
        divisor = 100.0
    else:
        divisor = 65535.0
    return np.clip(depth / divisor, 0, 1)


def _random_crop(rgb, depth, size):
    height, width = depth.shape
    if height < size or width < size:
        raise ValueError(f"image {width}x{height} is smaller than crop size {size}")
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return rgb[top:top + size, left:left + size], depth[top:top + size, left:left + size]


class DepthTrainingDataset(Dataset):
    def __init__(self, roots, crop_size=320, hole_root=None):
        self.pairs = find_pairs(roots)
        self.crop_size = crop_size
        self.holes = sorted(Path(hole_root).rglob("*.png")) if hole_root else []
        if hole_root and not self.holes:
            raise ValueError(f"No hole-mask PNGs found under {hole_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rgb_path, gt_path = self.pairs[index]
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        depth = _read_depth(gt_path)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth dimensions differ: {rgb_path}, {gt_path}")
        rgb, depth = _random_crop(rgb, depth, self.crop_size)
        if random.random() < 0.5:
            rgb, depth = np.flip(rgb, axis=1), np.flip(depth, axis=1)
        low, high = float(depth.min()), float(depth.max())
        if high > low:
            depth = (depth - low) / (high - low)
        rgb_image = Image.fromarray(np.ascontiguousarray(rgb))
        rgb_image = ImageEnhance.Color(rgb_image).enhance(random.uniform(0.8, 1.2))
        rgb_image = ImageEnhance.Brightness(rgb_image).enhance(random.uniform(0.8, 1.2))
        rgb_image = ImageEnhance.Contrast(rgb_image).enhance(random.uniform(0.8, 1.2))
        rgb = torch.from_numpy(np.asarray(rgb_image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0)
        gt = torch.from_numpy(np.ascontiguousarray(depth[None])).float()
        valid = (gt > 0).float()

        choice = random.random()
        if choice < 0.2:
            keep = 1.0
        elif choice < 0.4:
            keep = 1.0 - random.uniform(0.0, 0.9)
        elif choice < 0.6:
            keep = 1.0 - random.uniform(0.9, 1.0)
        else:
            keep = 0.0
        distorted = gt.clone()
        if keep > 0 and random.random() < (0.0 if keep == 1.0 else 0.3):
            noise_mask = (torch.rand_like(gt) < random.random()).float() * valid
            distorted += torch.randn_like(gt) * random.uniform(0.01, 0.1) * noise_mask
        if keep > 0 and random.random() < (1.0 if keep == 1.0 else 0.3):
            factor = 2 ** random.randint(1, 4)
            smaller = F.interpolate(distorted.unsqueeze(0), size=(max(1, self.crop_size // factor),) * 2,
                                    mode="nearest")
            distorted = F.interpolate(smaller, size=(self.crop_size, self.crop_size), mode="nearest").squeeze(0)
        sparse_mask = (torch.rand_like(gt) < keep).float() * valid
        sparse_depth = distorted.clamp(0, 1) * sparse_mask

        if self.holes and random.random() < 0.5:
            with Image.open(random.choice(self.holes)) as image:
                hole = image.convert("L").resize((self.crop_size, self.crop_size), Image.Resampling.NEAREST)
                hole = torch.from_numpy(np.asarray(hole, dtype=np.uint8).copy())
            hole = (hole > 0).float().unsqueeze(0)
            if random.random() < 0.5:
                hole = torch.flip(hole, dims=(-1,))
            sparse_mask *= hole
            sparse_depth *= hole
        return rgb, gt, sparse_depth, valid, sparse_mask
