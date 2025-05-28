import math
import signal
import os
import sys
import torch
import numpy as np
import wandb
import time
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from torchinfo import summary
from schedulefree import AdamWScheduleFree
from utils.utils_transforms import LIPCONSTANT_PREPROCESS_DATASET
from deel.lip_rcp.loss import (
    SoftHKRMulticlassLoss,
    SelfNormalizingCrossEntropy,
    LossXent,
    HSigLoss,
    TauCCE,
)
from torch.amp import GradScaler
from functools import partial
import argparse

torch.backends.cudnn.benchmark = True
scaler = GradScaler("cuda")

parser = argparse.ArgumentParser(description="Train LipNet")
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="Dataset to train on"
)
parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--loss_fn", type=str, default="shkr", help="Loss to use")
parser.add_argument("--alpha", type=float, default=0.975, help="Alpha")
parser.add_argument("--min_margin", type=float, default=0.6, help="Min margin")
parser.add_argument("--temperature", type=float, default=10.0, help="Temperature")
parser.add_argument("--offset", type=float, default=1.41, help="Offset for cce")
parser.add_argument("--p_dropout", type=float, default=0.0, help="Dropout")
parser.add_argument("--on_gpu", type=int, default=0, help="GPU number")
parser.add_argument("--wandb_log", action="store_true", help="Wandb log")
parser.add_argument(
    "--continue_training", action="store_true", help="Continue training a model"
)
parser.add_argument("--mixed_precision", action="store_true", help="Wandb log")
parser.add_argument("--p_aug", type=float, default=0.0, help="Augmentation probability")
parser.add_argument(
    "--p_data", type=float, default=0.0, help="Data augmentation probability"
)
args = parser.parse_args()


@torch.no_grad()
def test(loader, model, num_classes, criterion, device):
    correct, total, loss = 0, 0, 0
    for i, (data, target) in enumerate(loader):
        if len(target.shape) == 2:
            pass
        elif len(target.shape) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        else:
            raise ValueError("Wrong target shape")
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        gt = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(gt).sum().item()
        total += len(target)
    return loss / len(loader), 100 * (correct / total)


def train_one_epoch(
    train_loader,
    val_loader,
    model,
    num_classes,
    criterion,
    optimizer,
    epochs,
    device,
    data_trfm,
    log,
    mixed_precision,
    args,
    epoch,
):
    model.to(device)
    model.train()
    start = time.time()
    correct, total, loss_val = 0, 0, 0
    batch_size = len(train_loader)

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} [Train]",
    )
    if hasattr(optimizer, "train"):
        optimizer.train()

    for batch_idx, data in pbar if pbar else enumerate(train_loader):
        # Mixup, Cutmix like data augmentations
        if data_trfm:
            data = data_trfm(data)
        imgs, target = data
        imgs, target = imgs.to(device), target.to(device)
        if target.dim() == 2:
            pass
        elif target.dim() == 1:
            target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        else:
            raise ValueError("Wrong target shape")
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=mixed_precision
        ):
            output = model(imgs)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # Logging
        correct += (output.argmax(dim=1) == target.argmax(dim=1)).sum().item()
        total += target.size(0)
        topk = torch.topk(output, k=2, dim=1).values
        mm = (topk[:, 0] - topk[:, 1]).mean() / LIPCONSTANT_PREPROCESS_DATASET[
            args.dataset
        ]
        loss_val += loss.item() / batch_size
        acc = 100 * correct / total
        logits = output[0, :].cpu().detach().numpy()
        logits = sorted(logits, reverse=True)[:3]
        logits = [np.round(l, 2) for l in logits]

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
                "margin": f"{mm.item():.2E}",
                "outs": f"{logits}",
            }
        )

    stop = time.time()
    train_loss, train_acc = loss_val, 100 * (correct / total)

    if hasattr(optimizer, "eval"):
        optimizer.eval()

    val_loss, val_acc = test(val_loader, model, num_classes, criterion, device)
    print(
        "Epoch: {} Train Loss: {:.6f} Train Acc: {:.2f}% Val Loss: {:.6f} Val Acc: {:.2f}%\n".format(
            epoch + 1, train_loss, train_acc, val_loss, val_acc
        )
    )
    if log:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_epoch": stop - start,
            }
        )
    return model


stop_training = False


if __name__ == "__main__":
    from utils.utils_transforms import get_data

    train_dataset, val_dataset, cal_dataset, test_dataset, data_trfm = get_data(
        args.dataset,
        augmentations_prob=["crop", "autoaugment"],
        normalize=True,
        p_augmentations_imgs=args.p_aug,
        p_augmentations_mix=args.p_data,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Multiple GPUs
    if torch.cuda.device_count() > 1:
        device = f"cuda:{args.on_gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Get model
    from models import cifar10_model, cifar100_model

    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
    }[args.dataset]
    model = {
        "cifar10": partial(
            cifar10_model, num_classes=num_classes, p_dropout=args.p_dropout
        ),
        "cifar100": partial(
            cifar100_model, num_classes=num_classes, p_dropout=args.p_dropout
        ),
    }[args.dataset]()

    if args.continue_training:
        print(f"Loading model weights from model_weights/CNN_{args.dataset}.pth")
        model.load_state_dict(
            torch.load(f"model_weights/CNN_{args.dataset}.pth", map_location=device)
        )
        print("Model weights loaded")

    if args.wandb_log:
        name_run = "CNN_" + args.dataset
        run = wandb.init(project="rcp-pr", name=name_run)

    summary(model, input_size=(1, 3, 32, 32))
    criterion = {
        "std_cce": partial(SelfNormalizingCrossEntropy, t=1 / args.temperature),
        "tau_cce": partial(TauCCE, tau=args.temperature),
        "xent": partial(
            LossXent,
            n_classes=num_classes,
            offset=args.offset,
            t=1 / args.temperature,
        ),
        "shkr": partial(
            SoftHKRMulticlassLoss,
            min_margin=args.min_margin,
            alpha=args.alpha,
            temperature=args.temperature,
        ),
        "hsig": partial(
            HSigLoss,
            alpha=args.alpha,
            min_margin=args.min_margin,
            temperature=args.temperature,
        ),
    }[args.loss_fn]()
    optimizer = AdamWScheduleFree(model.parameters(), lr=args.learning_rate)

    def on_sigint(signum, frame):
        global stop_training
        print("\n Received SIGINT — will finish this epoch, then stop.")
        stop_training = True

    signal.signal(signal.SIGINT, on_sigint)

    # Train model
    for epoch in range(args.epochs):
        model = train_one_epoch(
            train_loader,
            test_loader,
            model,
            num_classes,
            criterion,
            optimizer,
            args.epochs,
            device,
            data_trfm,
            log=args.wandb_log,
            mixed_precision=args.mixed_precision,
            args=args,
            epoch=epoch,
        )
        if stop_training:
            print(f"✅  Stopping after epoch {epoch}")
            break

    def _timeout_handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(10)

    try:
        answer = input("Save model (y/n): ").strip().lower()
        # Cancel the alarm if input arrives in time
        signal.alarm(0)
        if answer in ("y", "yes"):
            torch.save(model.state_dict(), f"model_weights/CNN_{args.dataset}.pth")
            print(f"Saved model to model_weights/CNN_{args.dataset}.pth")
        else:
            print("Will not save model")
    except TimeoutError:
        torch.save(model.state_dict(), f"model_weights/CNN_{args.dataset}.pth")
        print(f"Saved model to model_weights/CNN_{args.dataset}.pth")
