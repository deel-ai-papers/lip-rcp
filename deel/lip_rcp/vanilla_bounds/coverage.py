import torch
import torch.nn.functional as F
import numpy as np

from scipy.stats import binom
from scipy.optimize import bisect


def risk(Lis):
    return np.mean(Lis)


def binom_inv_cdf(Lis, n, delta):
    Rhat = risk(Lis)
    k = int(Rhat * n)

    def f_bin(p):
        return binom.cdf(k, n, p) - delta - 1e-5

    return bisect(f_bin, 1e-10, 1 - 1e-10, maxiter=1000, xtol=1e-8)


def binom_inv_cdf_uni(Rhat, n, delta):
    eta = n - 1
    delta = delta / (eta)
    k = int(Rhat * n)

    def f_bin(p):
        return binom.cdf(k, n, p) - delta - 1e-5

    return bisect(f_bin, 1e-10, 1 - 1e-10, maxiter=1000, xtol=1e-8) + 1 / n


# GET COVERAGE ROBUSTNESS
def robust_prediction_set(logits, lbd, nc_method, temp, bias, attack_radius):
    if nc_method == "lac_softmax":
        num_classes = logits.size(1)
        opt_outs = logits.unsqueeze(1).expand(-1, num_classes, -1) - attack_radius
        idx = torch.arange(num_classes, device=logits.device)
        opt_outs[:, idx, idx] += 2.0 * attack_radius
        opt_scores = F.softmax(opt_outs / temp, dim=-1)
        opt_ps = opt_scores >= lbd
        rps = torch.any(opt_ps, dim=1).float()
    elif nc_method == "lac_sigmoid":
        scores = F.sigmoid((logits - bias) / temp)
        lipconstant = 1 / (4 * temp)
        r_scores = scores + lipconstant * attack_radius
        rps = r_scores >= lbd
    else:
        raise NotImplementedError()
    return rps


def gamma_minmax(
    model: torch.nn.Module,
    nc_method: str,
    lbd: float,
    holdout_loader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_radius: float,
    delta: float,
    temp: float,
    bias: float,
):
    """

    Compute the upper and lower bounds for finite sample conformal coverage of vanilla CP under bounded adversarial attacks with probability 1-delta.

    args:
    - model: torch.nn.Module: the model to evaluate
    - nc_method: str: the method to compute the non-conformity score
    - lbd: float: the lambda parameter for the non-conformity score
    - holdout_loader: torch.utils.data.DataLoader: the holdout data loader
    - device: torch.device: the device to use
    - attack_radius: list: the radius of the attack
    - delta: float: the probability of the coverage
    - temp: float: the temperature for the softmax
    - bias: float: the bias for the non-conformity score

    returns:
    - gamma_min: float: the lower bound for the coverage
    - gamma_max: float: the upper bound for the coverage

    """
    assert delta > 0 and delta < 1, "Delta must be in (0, 1)"

    risk_min = np.array([])
    risk_max = np.array([])
    total = 0

    with torch.no_grad():
        for _, data in enumerate(holdout_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.cpu().numpy()
            batch_sz = len(labels)

            conservative_ps = (
                robust_prediction_set(
                    logits=outputs,
                    lbd=lbd,
                    nc_method=nc_method,
                    temp=temp,
                    bias=bias,
                    attack_radius=attack_radius,
                )
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            restrictive_ps = (
                robust_prediction_set(
                    logits=outputs,
                    lbd=lbd,
                    nc_method=nc_method,
                    temp=temp,
                    bias=bias,
                    attack_radius=-attack_radius,  # negative attack radius for rps
                )
                .float()
                .detach()
                .cpu()
                .numpy()
            )

            loss = restrictive_ps[np.arange(batch_sz), labels]
            risk_min = np.concatenate((risk_min, 1 - loss))
            loss = conservative_ps[np.arange(batch_sz), labels]
            risk_max = np.concatenate((risk_max, loss))
            total += len(labels)

    risk_min = np.mean(risk_min)
    risk_max = np.mean(risk_max)

    if np.mean(risk_min) != 1:
        gamma_min = 1 - binom_inv_cdf_uni(risk_min, total, delta)
    else:
        gamma_min = 0

    if np.mean(risk_max) != 1:
        gamma_max = binom_inv_cdf_uni(risk_max, total, delta)
    else:
        gamma_max = 1

    return gamma_min, gamma_max
