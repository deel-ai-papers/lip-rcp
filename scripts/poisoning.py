import math
import os
import numpy as np
import sys

import torch.utils.data
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse

args = argparse.ArgumentParser()
args.add_argument("--alpha", type=float, default=0.1)
args.add_argument("--bias", type=float, default=0.01)
args.add_argument("--temp", type=float, default=5.0)
args.add_argument("--epsilon", type=float, default=0.2)
args.add_argument("--n_samples", type=int, default=300)
args.add_argument("--batch_size", type=int, default=128)
args = args.parse_args()

ALPHA = args.alpha
BIAS = args.bias
TEMP = args.temp
EPSILON = args.epsilon
N_SAMPLES = args.n_samples
BATCH_SIZE = args.batch_size


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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
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

    scores_cal = np.array([])
    for i, data in enumerate(cal_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        labels = labels.detach().cpu().numpy()

        scores = F.sigmoid((outputs - BIAS) / TEMP)
        scores = scores.detach().cpu().numpy()
        scores = scores[np.arange(len(scores)), labels]

        scores_cal = np.concatenate((scores_cal, scores))

    scores_cal = np.sort(scores_cal)
    idx_q = np.floor(ALPHA * len(scores_cal)).astype(int)

    from deel.lip_rcp.poisoning import compute_robust_threshold

    lbd = scores_cal[idx_q]
    lbd_rob = compute_robust_threshold(
        scores_cal, N_SAMPLES, EPSILON * normalization_factor * 1 / (4 * TEMP), ALPHA
    )

    sz_ps, sz_ps_rob = 0, 0
    alpha, alpha_rob = 0, 0
    total = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        scores = F.sigmoid((outputs - BIAS) / TEMP)
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        ps = scores > lbd
        ps_rob = scores > lbd_rob

        alpha += ps[np.arange(len(ps)), labels].sum()
        alpha_rob += ps_rob[np.arange(len(ps_rob)), labels].sum()
        total += len(labels)

        sz_ps += np.mean(np.sum(ps, axis=1)) / len(test_loader)
        sz_ps_rob += np.mean(np.sum(ps_rob, axis=1)) / len(test_loader)

    print(f"\nUnder k = {args.n_samples}, epsilon = {args.epsilon}: \n")
    print(f"Vanilla CP: coverage - {alpha / total}, size - {sz_ps}")
    print(f"Robust CP: coverage - {alpha_rob / total}, size - {sz_ps_rob}")
