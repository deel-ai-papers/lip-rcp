import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import random
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from tqdm import tqdm
from torch.amp import GradScaler
import wandb
import argparse
import itertools

this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))
sys.path.append(parent_directory)

import schedulefree
from utils.datasets import load_imagenet1k
from utils.utils_transforms import UniformResize, LIPCONSTANT_PREPROCESS_DATASET
from deel.lip_rcp.loss import SoftHKRMulticlassLoss, TauCCE
from orthogonium.losses import VRA
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import (
    ToTensor,
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomCrop,
    CenterCrop,
    RandomResizedCrop,
    RandomApply,
    RandomRotation,
    AutoAugment,
    RandAugment,
    ColorJitter,
    Normalize,
    ToDtype,
    ToImage,
    MixUp,
)

torch.backends.cudnn.benchmark = True
scaler = GradScaler("cuda")

# ------------------------------------------------------------------------------------
# Constants & Hyperparams
# ------------------------------------------------------------------------------------
BASE_MODEL_PATH = "model_weights/CNN_imagenet.pth"


# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ImageNet_model")
    parser.add_argument("--loss_fn", type=str, help="Loss function to use")
    parser.add_argument("--optimizer", type=str, default="adamw_sf")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--alpha", type=float, default=0.985)
    parser.add_argument("--min_margin", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--p_dropout", type=float, default=0.0)
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--wandb_log", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------------------------
# DDP-related: get local rank
# ------------------------------------------------------------------------------------
def get_local_rank():
    """
    Returns the local rank for the current process.
    This is set automatically by `torchrun`.
    """
    # If running via torchrun, LOCAL_RANK is guaranteed to be set.
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def is_main_process():
    # Rank 0 will be the "main" process
    return get_rank() == 0


# ------------------------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------------------------
def get_transforms(train: bool = True):
    if train:
        return Compose(
            [
                # ToImage(),
                # UniformResize(224, 288),
                # Resize((224, 224)),
                # AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
                Resize(256),
                RandomResizedCrop(
                    224,
                    scale=(0.08, 1.0),
                    ratio=(3.0/4.0, 4.0/3.0),
                ),
                RandomHorizontalFlip(
                    0.3,
                ),
                RandAugment(
                    magnitude=7,
                    num_ops=2,
                ),
                ToTensor(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return Compose(
            [
                ToImage(),
                Resize((256, 256)),
                CenterCrop(224),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


# ------------------------------------------------------------------------------------
# Datasets & Dataloaders
# ------------------------------------------------------------------------------------
def create_datasets():
    """
    Returns: train_dataset, val_dataset
    """
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset, val_dataset = load_imagenet1k(
        False,
        train_transform,
        test_transform,
    )
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset, val_dataset, batch_size, num_workers, world_size, rank
):
    """
    Creates DataLoaders with DistributedSampler.
    """
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    val_sampler = DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        prefetch_factor=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler


# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
def create_model(num_classes=1000, args=None):
    from models import imagenet_model

    model = imagenet_model(num_classes=1000, p_dropout=args.p_dropout)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    if args.load_weights and os.path.exists(BASE_MODEL_PATH):
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location="cpu"))
        if is_main_process():
            print(f"Loaded weights from {BASE_MODEL_PATH}")
    return model


# ------------------------------------------------------------------------------------
# Training / Validation
# ------------------------------------------------------------------------------------
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    train_acc_metric,
    train_vra_metric,
    device,
    epoch,
    p_mixup,
    num_classes,
    mixed_precision,
    epsilon,
):
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    train_loss_sum = 0.0
    train_acc_metric.reset()
    train_vra_metric.reset()

    train_loader.sampler.set_epoch(epoch)

    pbar = None
    if is_main_process():
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} [Train]",
        )
    assert p_mixup != 1, "MixUp cannot be present on all batches."

    for batch_idx, (images, labels) in pbar if pbar else enumerate(train_loader):
        acc_metrics = True
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels = F.one_hot(labels, num_classes=1000) if isinstance(criterion, SoftHKRMulticlassLoss) else labels
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=mixed_precision
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        
        # Clipping to avoid NaN:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        labels = labels.argmax(1) if isinstance(criterion, SoftHKRMulticlassLoss) else labels

        # Update metrics
        train_loss_sum += loss.item()

        if acc_metrics:
            train_acc_metric.update(outputs, labels)
            acc = torch.argmax(outputs, dim=1).eq(labels).sum() / labels.size(0)
            vra_val = VRA(
                outputs,
                labels,
                L=LIPCONSTANT_PREPROCESS_DATASET['imagenet'],
                eps=epsilon,
                last_layer_type="global",
            )
            train_vra_metric.update(vra_val)

        topk = torch.topk(outputs, k=2, dim=1).values
        mm = (topk[:, 0] - topk[:, 1]).mean()
        mean_top1 = torch.max(outputs, dim=1).values.mean()

        if is_main_process():
            if acc_metrics:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{acc.item():.4f}",
                        "margin": f"{mm.item():.2E}",
                        "mean_top1": f"{mean_top1.item():.2E}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "margin": f"{mm.item():.4f}",
                        "mean_top1": f"{mean_top1.item():.2E}",
                    }
                )

    train_loss_avg = train_loss_sum / len(train_loader)
    train_acc = train_acc_metric.compute().item()
    train_vra = train_vra_metric.compute().item()

    return train_loss_avg, train_acc, train_vra


@torch.no_grad()
def validate_one_epoch(
    model, val_loader, criterion, val_acc_metric, val_vra_metric, device, epoch, epsilon
):
    val_loss_sum = 0.0
    val_acc_metric.reset()
    val_vra_metric.reset()

    if is_main_process():
        pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch + 1} [Val]",
        )
    else:
        pbar = enumerate(val_loader)

    for batch_idx, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        labels = F.one_hot(labels, num_classes=1000) if isinstance(criterion, SoftHKRMulticlassLoss) else labels
        loss = criterion(outputs, labels)
        labels = labels.argmax(1) if isinstance(criterion, SoftHKRMulticlassLoss) else labels
        val_loss_sum += loss.item()
        val_acc_metric.update(outputs, labels)
        vra_val = VRA(
            outputs, labels, L=LIPCONSTANT_PREPROCESS_DATASET["imagenet"], eps=epsilon, last_layer_type="global"
        )
        val_vra_metric.update(vra_val)
        if is_main_process():
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss_avg = val_loss_sum / len(val_loader)
    val_acc = val_acc_metric.compute().item()
    val_vra = val_vra_metric.compute().item()

    return val_loss_avg, val_acc, val_vra


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    # 0. Parse args
    args = parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = get_local_rank()
    world_size = get_world_size()
    rank = get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if is_main_process() and args.wandb_log:
        wandb.init(project="rcp-pr", name=args.name)
        wandb.config.max_epochs = args.epochs
        wandb.config.update(args)

    train_dataset, val_dataset = create_datasets()
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
    )

    model = create_model(num_classes=1000, args=args).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    from functools import partial

    criterion = {
        "shkr": partial(SoftHKRMulticlassLoss, alpha=args.alpha, min_margin=args.min_margin, temperature=args.temperature),
        "tau_cce":partial(TauCCE, tau=args.temperature),
    }[args.loss_fn]()

    if args.optimizer == "adamw_sf":
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
        )
        scheduler = None
    elif args.optimizer == "custom":
        from utils.lookahead import Lookahead
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.optim import NAdam
        base_optimizer = NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(base_optimizer, k=3, alpha=0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max =args.epochs, eta_min=5e-6)
    else:
        raise ValueError("Unrecognized optimizer")

    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(
        device
    )
    val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(
        device
    )
    train_vra_metric = torchmetrics.MeanMetric().to(device)
    val_vra_metric = torchmetrics.MeanMetric().to(device)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_acc, train_vra = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            train_acc_metric=train_acc_metric,
            train_vra_metric=train_vra_metric,
            device=device,
            epoch=epoch,
            p_mixup=0,
            num_classes=1000,
            mixed_precision=args.mixed_precision,
            epsilon=args.epsilon,
        )
        if scheduler is not None:
            scheduler.step()

        if hasattr(optimizer, "eval"):
            optimizer.eval()
            with torch.no_grad():
                for batch in itertools.islice(train_loader, 50):
                    outputs = model(batch[0])
        model.eval()

        val_loss, val_acc, val_vra = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            val_acc_metric=val_acc_metric,
            val_vra_metric=val_vra_metric,
            device=device,
            epoch=epoch,
            epsilon=args.epsilon,
        )

        dist.barrier()

        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train VRA: {train_vra:.4f}"
            )
            print(
                f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   VRA: {val_vra:.4f}"
            )

            if args.wandb_log:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_vra": train_vra,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_vra": val_vra,
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("model_weights", exist_ok=True)
                torch.save(model.module.state_dict(), "model_weights/best_imagenet.pth")
                print(">> Saved best checkpoint.")

    if is_main_process():
        torch.save(model.module.state_dict(), "model_weights/latest_imagenet.pth")
        print(">> Final model saved at model_weights/latest_imagenet.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
