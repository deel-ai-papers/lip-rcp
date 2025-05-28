import math
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch.utils.data
import torch.nn.functional as F

import argparse

args = argparse.ArgumentParser()
args.add_argument("--alpha", type=float, default=0.1)
args.add_argument("--batch_size", type=int, default=128)
args.add_argument("--bias", type=float, default=0.01)
args.add_argument("--delta", type=float, default=0.001)
args.add_argument("--epsilon", type=float, default=0.03)
args.add_argument("--temp", type=float, default=5.0)
args.add_argument("--n_iters", type=int, default=1)
args = args.parse_args()

if __name__ == "__main__":
    from utils.utils_transforms import get_data
    from utils.utils_transforms import LIPCONSTANT_PREPROCESS_DATASET

    train_dataset, val_dataset, cal_dataset, test_dataset, data_trfm = get_data(
        "cifar10",
        augmentations_prob=[],
        normalize=True,
        p_augmentations_imgs=0,
        p_augmentations_mix=0,
    )
    normalization_factor = LIPCONSTANT_PREPROCESS_DATASET["cifar10"]

    # Multiple GPUs
    if torch.cuda.device_count() > 1:
        device = "cuda:0"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from models import cifar10_model

    model = cifar10_model(num_classes=10, p_dropout=0)
    model.to(device)

    model.load_state_dict(torch.load("model_weights/CNN_cifar10.pth"))

    min_bounds, max_bounds = [], []
    alphas_val, ss_val = [], []
    for iteration in range(args.n_iters):
        superset = torch.utils.data.ConcatDataset(
            [val_dataset, cal_dataset, test_dataset]
        )
        val_dataset, cal_dataset, test_dataset = torch.utils.data.random_split(
            superset, lengths=[0.1, 0.45, 0.45]
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

        from deel.lip_rcp.robust_ps import calibrate_lac_sigmoid

        lbd = calibrate_lac_sigmoid(
            model=model,
            calibration_loader=cal_loader,
            device=device,
            alpha=args.alpha,
            temp=args.temp,
            bias=args.bias,
            num_batches=1000,
        )

        from deel.lip_rcp import gamma_minmax

        alpha, correct, total, set_size = 0, 0, 0, 0
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            scores = F.sigmoid((outputs - args.bias) / args.temp)
            pred_sets = scores >= lbd
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            set_size += pred_sets.sum().cpu().numpy()
            alpha += pred_sets[np.arange(len(pred_sets)), labels].sum().item()
            total += len(labels)

        alphas_val.append(alpha / total)
        ss_val.append(set_size / total)

        gamma_min, gamma_max = gamma_minmax(
            model,
            nc_method="lac_sigmoid",
            lbd=lbd,
            holdout_loader=test_loader,
            device=device,
            attack_radius=args.epsilon * normalization_factor,
            delta=args.delta,
            temp=args.temp,
            bias=args.bias,
        )
        max_bounds.append(gamma_max)
        min_bounds.append(gamma_min)

    alpha = np.mean(alphas_val)
    ss = np.mean(ss_val)
    gamma_min = np.mean(gamma_min)
    gamma_max = np.mean(gamma_max)

    print(f"\nVANILLA CP:")
    print(f"Set size On clean validation split - Coverage: {alpha}, Set size: {ss}\n")

    print(f"\nVANILLA CP COVERAGE BOUNDS FOR RADIUS {args.epsilon}:")
    print(
        f"Coverage in [{gamma_min}, {gamma_max}] with probability at least {1 - args.delta} \n"
    )
