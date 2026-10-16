"""Train JARRN from RGB/depth pairs. Supports CPU, one GPU, or torchrun."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from jarrn.data import DepthTrainingDataset
from jarrn.losses import JointLoss
from jarrn.model import JARRN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, action="append", required=True,
                        help="Repeat for each DIODE/train or HRWSI/train directory")
    parser.add_argument("--hole-root", type=Path, help="Optional PNG hole-mask directory")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/jarrn"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process batch size")
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--mode", choices=("mar", "ma", "md"), default="mar")
    parser.add_argument("--rezero", action="store_true")
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 16 or args.save_every < 1 or args.log_every < 1:
        parser.error("epoch/batch/log/save counts must be positive and crop size at least 16")
    return args


def distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, device


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, args):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                   if key != "data_root"} | {"data_root": [str(p) for p in args.data_root]},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    rank, world_size, device = distributed_setup()
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        dataset = DepthTrainingDataset(args.data_root, args.crop_size, args.hole_root)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            shuffle=sampler is None, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", drop_last=False)
        model = JARRN(args.levels, args.base_channels, args.rezero).to(device)
        optimizer_type = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
        optimizer = optimizer_type(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = JointLoss(args.mode).to(device)
        start_epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"]
            if start_epoch >= args.epochs:
                raise ValueError("--epochs must exceed the checkpoint epoch")
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        raw_model = model.module if world_size > 1 else model
        if rank == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training {len(dataset)} pairs on {world_size} process(es); device={device}", flush=True)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            for step, batch in enumerate(loader, start=1):
                rgb, gt, sparse, valid, sparse_mask = [tensor.to(device, non_blocking=True) for tensor in batch]
                optimizer.zero_grad(set_to_none=True)
                amp_context = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
                with amp_context:
                    prediction = model(rgb, sparse, sparse_mask)[0]
                    loss, terms = criterion(prediction, gt, valid, sparse_mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach())
                if rank == 0 and (step == 1 or step % args.log_every == 0):
                    print(f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} "
                          f"loss={loss.item():.4f} abs={terms['absolute'].item():.4f} "
                          f"rel={terms['relative'].item():.4f} grad={terms['gradient'].item():.4f}", flush=True)
            scheduler.step()
            totals = torch.tensor([loss_sum, len(loader)], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if rank == 0:
                summary = {"epoch": epoch, "mean_loss": (totals[0] / totals[1]).item(),
                           "learning_rate": optimizer.param_groups[0]["lr"]}
                with (args.save_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(summary) + "\n")
                if epoch % args.save_every == 0 or epoch == args.epochs:
                    save_checkpoint(args.save_dir / f"epoch_{epoch:03d}.pt", raw_model, optimizer,
                                    scheduler, scaler, epoch, args)
            if world_size > 1:
                dist.barrier()
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
