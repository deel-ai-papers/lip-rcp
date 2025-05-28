import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial


def get_fcts(model, device, attack_radius, args):
    calibrate = {
        "lac_softmax": partial(
            calibrate_lac_softmax,
            model=model,
            alpha=args.alpha,
            temp=args.temp,
            device=device,
            num_batches=args.num_batches,
        ),
        "lac_sigmoid": partial(
            calibrate_lac_sigmoid,
            model=model,
            alpha=args.alpha,
            temp=args.temp,
            bias=args.bias,
            device=device,
            num_batches=args.num_batches,
        ),
        "max_ratio": partial(
            calibrate_max_ratio,
            model=model,
            alpha=args.alpha,
            device=device,
            num_batches=args.num_batches,
        ),
    }
    evaluate = {
        "lac_softmax": partial(
            evaluate_lac_softmax,
            model=model,
            temp=args.temp,
            attack_radius=attack_radius,
            device=device,
            num_batches=args.num_batches,
            propagate=args.propagate,
        ),
        "lac_sigmoid": partial(
            evaluate_lac_sigmoid,
            model=model,
            temp=args.temp,
            bias=args.bias,
            attack_radius=attack_radius,
            device=device,
            num_batches=args.num_batches,
            propagate=args.propagate,
        ),
        "max_ratio": partial(
            evaluate_max_ratio,
            model=model,
            attack_radius=attack_radius,
            device=device,
            num_batches=args.num_batches,
        ),
    }
    return calibrate, evaluate


### LAC softmax


# Compute conservative prediction scores
def compute_robust_ps(logits, threshold, temp, attack_radius, propagate=True):
    # Compute worst-case variation through the softmax function locally
    if propagate:
        num_classes = logits.size(1)
        opt_outs = logits.unsqueeze(1).expand(-1, num_classes, -1) - attack_radius
        idx = torch.arange(num_classes, device=logits.device)
        opt_outs[:, idx, idx] += 2.0 * attack_radius
        opt_scores = F.softmax(opt_outs / temp, dim=-1)
        opt_ps = opt_scores >= threshold
        opt_ps = torch.any(opt_ps, dim=1).float()
    # Compute worst-case variations by using the softmax's Lipschitz constant
    else:
        opt_scores = F.softmax(logits / temp, dim=1)
        opt_scores += attack_radius / (2 * temp) 
        opt_ps = opt_scores >= threshold
    return opt_ps


# Calibrate using vanilla scores
def calibrate_lac_softmax(model, calibration_loader, temp, alpha, device, num_batches):
    model.eval()
    scores = None
    with torch.no_grad():
        for i, data in tqdm(enumerate(calibration_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits / temp, dim=1)
            curr_s = probs[torch.arange(y.size(0)), y].cpu().numpy()
            if scores is None:
                scores = curr_s
            else:
                scores = np.concatenate([scores, curr_s])
            if i + 1 > num_batches:
                break
    n = scores.shape[0]
    q_hat = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, 1 - q_hat, method="lower")
    return threshold


# Evaluate performance metrics
def evaluate_lac_softmax(
    model, test_loader, temp, threshold, attack_radius, device, num_batches, propagate=True,
):
    model.eval()
    ss, alpha, total = 0, 0, 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            rps = compute_robust_ps(
                logits,
                threshold,
                temp=temp,
                attack_radius=attack_radius,
                propagate=propagate,
            )
            ss += rps.sum().item()
            alpha += rps[torch.arange(y.size(0)), y].sum().item()
            total += y.size(0)
            if i + 1 > num_batches:
                break
    coverage = alpha / total
    avg_set_size = ss / total
    return coverage, avg_set_size


### LAC sigmoid


# Calibrate using vanilla scores
def calibrate_lac_sigmoid(
    model, calibration_loader, temp, bias, alpha, device, num_batches
):
    model.eval()
    scores = None
    with torch.no_grad():
        for i, data in tqdm(enumerate(calibration_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            scores_c = F.sigmoid((logits - bias) / temp)
            curr_s = scores_c[torch.arange(y.size(0)), y].cpu().numpy()
            if scores is None:
                scores = curr_s
            else:
                scores = np.concatenate([scores, curr_s])
            if i + 1 > num_batches:
                break
    n = scores.shape[0]
    q_hat = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, 1 - q_hat, method="lower")
    return threshold


# Evaluate performance metrics
def evaluate_lac_sigmoid(
    model, test_loader, temp, bias, threshold, attack_radius, device, num_batches, propagate
):
    model.eval()
    ss, alpha, total = 0, 0, 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if propagate:
                # Use local Lipschitz computations
                rps = F.sigmoid((logits + attack_radius - bias) / temp) >= threshold
            else:
                rps = F.sigmoid((logits - bias) / temp) + attack_radius / (4 * temp) >= threshold
            ss += rps.sum().item()
            alpha += rps[torch.arange(y.size(0)), y].sum().item()
            total += y.size(0)
            if i + 1 > num_batches:
                break
    coverage = alpha / total
    avg_set_size = ss / total
    return coverage, avg_set_size


### Max ratio score


# Calibrate using vanilla scores
def calibrate_max_ratio(model, calibration_loader, alpha, device, num_batches):
    model.eval()
    scores = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(calibration_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            curr_s = torch.exp(logits - logits.max(1, keepdim=True).values)
            curr_s = curr_s[torch.arange(logits.size(0)), y].view(-1)
            curr_s = curr_s.cpu().numpy()
            scores.append(curr_s)
            if i + 1 > num_batches:
                break
    scores_np = np.array(scores, dtype=float)
    n = scores_np.shape[0]
    q_hat = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores_np, 1 - q_hat, method="lower")
    return float(threshold)


# Evaluate performance metrics
def evaluate_max_ratio(
    model,
    test_loader,
    threshold,
    attack_radius,
    device,
    num_batches,
):
    model.eval()
    n, covered, total_size = 0, 0, 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # Compute pessimistic scores
            opt_scores = torch.exp(
                logits - logits.max(1, keepdim=True).values + 2 * attack_radius
            )
            pred_sets = opt_scores >= threshold
            # Update metrics
            covered += pred_sets[torch.arange(logits.size(0)), y].sum().item()
            total_size += pred_sets.sum().item()
            n += logits.size(0)
            if i + 1 > num_batches:
                break
    coverage = covered / n
    avg_set_size = total_size / n
    return coverage, avg_set_size
