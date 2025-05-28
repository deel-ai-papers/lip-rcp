import os
import sys
import torch
from tqdm import tqdm
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.utils_transforms import LIPCONSTANT_PREPROCESS_DATASET
from utils.utils_transforms import get_loaders
from models import cifar10_model, cifar100_model, imagenet_model
from functools import partial
import argparse

parser = argparse.ArgumentParser(description="Fast RCP")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument(
    "--num_batches",
    type=int,
    default=-1,
    help="Number of batches of cal/test to use",
)
parser.add_argument(
    "--on_gpu",
    type=int,
    default=0,
    help="GPU ID to use",
)
parser.add_argument(
    "--score_fn",
    type=str,
    default="lac_sigmoid",
    help="Score function to use",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    help="Confidence level for calibration",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.03,
    help="Epsilon for robust calibration",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training",
)
parser.add_argument(
    "--temp",
    type=float,
    default=1.0,
    help="The temperature for the score",
)
parser.add_argument(
    "--bias",
    type=float,
    default=0.0,
    help="The bias for the score",
)
parser.add_argument(
    "--num_iters",
    type=int,
    default=1,
)
parser.add_argument(
    "--large",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--model_path",
    type=str,
    default="",
    help="Path to model weights",
)
parser.add_argument(
    "--propagate",
    action="store_false",
)
args = parser.parse_args()
print("Running with config: ", args)

lipconstant = LIPCONSTANT_PREPROCESS_DATASET[args.dataset]

print("Model init...")
num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}[args.dataset]
model = {
    "cifar10": partial(cifar10_model, num_classes=num_classes, p_dropout=0),
    "cifar100": partial(cifar100_model, num_classes=num_classes, p_dropout=0),
    "imagenet": partial(imagenet_model, num_classes=num_classes, p_dropout=0),
}[args.dataset]()
device = torch.device(f"cuda:{args.on_gpu}")
model = model.to(device)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
model.load_state_dict(
    torch.load(f"model_weights/CNN_{args.dataset}.pth", map_location=device)
)
print(f"Loaded weights from model_weights/CNN_{args.dataset}.pth")

from deel.lip_rcp.robust_ps import get_fcts

attack_radius = args.epsilon * lipconstant
calibrate, evaluate = get_fcts(model, device, attack_radius, args)

## Evaluate
with torch.no_grad():
    cov, pss = [], []
    for iteration in range(args.num_iters):
        cal_loader, test_loader = get_loaders(args)
        ## Sanity check
        if iteration == 0:
            print("Sanity check...")
            dict_k = {"cifar10": 3, "cifar100": 10, "imagenet": 50}
            k = dict_k[args.dataset]
            i, correct, topk, total = 0, 0, 0, 0
            model.eval()
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, preds = torch.topk(logits, k, dim=1)
                correct += (logits.argmax(1) == labels).sum().item()
                topk += (preds == labels.unsqueeze(1)).sum().item()
                total += labels.size(0)
                i += 1
                if i + 1 > 10:
                    break
            print(f"Top-1 accuracy: {100 * correct / total}%")
            print(f"Top-{k} accuracy: {100 * topk / total}%")

        print("Calibration step...")
        q_hat = calibrate[args.score_fn](calibration_loader=cal_loader)
        print("Eval step...")
        coverage, ss = evaluate[args.score_fn](threshold=q_hat, test_loader=test_loader)
        print(f"Coverage: {coverage} / Average Set size: {ss}")
        cov.append(coverage)
        pss.append(ss)

# Logging
import pandas as pd

row = {
    "score_fn": args.score_fn,
    "temp": args.temp,
    "bias": args.bias,
    "alpha": args.alpha,
    "epsilon": args.epsilon,
    "coverage": np.mean(cov),
    "set_size": np.mean(pss),
}
df = pd.DataFrame([row])
csv_path = f"results/{args.dataset}.csv"

if not os.path.exists(csv_path):
    df.to_csv(csv_path, index=False)
else:
    df.to_csv(csv_path, mode="a", header=False, index=False)
