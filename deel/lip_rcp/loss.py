import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from typing import Optional
from torch import Tensor


def sigmoid_tau_cross_entropy(logits, gt, tau):
    assert logits.shape == gt.shape, "logits and ground truth must have the same shape."
    return F.binary_cross_entropy_with_logits(logits * tau, gt)


def hinge_loss(logits, gt, margin=1.0):
    assert logits.shape == gt.shape, "logits and ground truth must have the same shape."
    return F.hinge_embedding_loss(logits, gt, margin=margin)


class HSigLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.95,
        min_margin=1.0,
        temperature=1.0,
    ):
        """
        BCE from logits with a temperature factor with an additional Hinge margin constraint.

        Args:
            alpha (float): regularization factor (0 <= alpha <= 1),
                0 for BCE only, 1 for hinge only
            min_margin (float): margin to enforce.
            temperature (float): factor for sigmoid temperature
                (higher value increases the weight of the highest non y_true logits)
        """
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), "alpha must in [0,1]"
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.min_margin = min_margin
        self.temperature = temperature

    def forward(self, logits, gt):
        assert logits.shape == gt.shape, (
            "logits and ground truth must have the same shape."
        )
        gt = gt.float()
        bce_loss = sigmoid_tau_cross_entropy(logits, gt, self.temperature).mean()
        hinge_loss = F.hinge_embedding_loss(
            logits, 2 * gt - 1, margin=self.min_margin
        ).mean()
        return self.alpha * bce_loss + (1 - self.alpha) * hinge_loss


def apply_reduction(val: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "auto":
        reduction = "mean"
    red = getattr(torch, reduction, None)
    if red is None:
        return val
    return red(val)


class TauCCE(CrossEntropyLoss):
    def __init__(
        self,
        tau: float,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """
        The loss add a temperature (tau) factor to the CrossEntropyLoss
        CrossEntropyLoss(tau * input, target).

        See `CrossEntropyLoss` for more details on arguments.

        Args:
            tau (float): factor for  temperature
        """

        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape == target.shape:
            return super().forward(input * self.tau, target.to(torch.double)) / self.tau
        else:
            return super().forward(input * self.tau, target.to(torch.int64)) / self.tau


class LossXent(nn.Module):
    def __init__(self, n_classes, offset=2.12132, t=0.25):
        """
        A custom loss function class for cross-entropy calculation.

        This class initializes a cross-entropy loss criterion along with additional
        parameters, such as an offset and a temperature factor, to allow a finer control over
        the accuracy/robustness tradeoff during training.

        Attributes:
            criterion (nn.CrossEntropyLoss): The PyTorch cross-entropy loss criterion.
            n_classes (int): The number of classes present in the dataset.
            offset (float): An offset value for customizing the loss computation.
            temperature (float): A temperature factor for scaling logits during loss calculation.

        Parameters:
            n_classes (int): The number of classes in the dataset.
            offset (float, optional): The offset value for loss computation. Default is 2.12132.
            temperature (float, optional): The temperature scaling factor. Default is 0.25.
        """
        super(LossXent, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.offset = offset
        self.temperature = t

    def __call__(self, outputs, labels):
        if labels.dim() == 1:
            one_hot_labels = torch.nn.functional.one_hot(
                labels, num_classes=self.n_classes
            ).float()
            labels = labels.float()
        else:
            one_hot_labels = labels.float()
            labels = labels.argmax(1)
        offset_outputs = outputs - self.offset * one_hot_labels
        offset_outputs /= self.temperature
        loss = self.criterion(offset_outputs, labels) * self.temperature
        return loss


class SelfNormalizingCrossEntropy(nn.Module):
    def __init__(self, t=0.1):
        super().__init__()
        self.t = t
        self.eps = 1e-8

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        else:
            targets = targets.float()
        std_per_sample = logits.std(dim=1, keepdim=True) + self.eps
        scaled_logits = logits / (std_per_sample + self.t)
        shat = (F.softmax(scaled_logits - targets, dim=1)).float()
        loss = F.cross_entropy(shat, targets)
        return loss.mean()


class SoftHKRMulticlassLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.9,
        min_margin=1.0,
        alpha_mean=0.99,
        temperature=1.0,
        reduction: str = "mean",
    ):
        if (alpha >= 0) and (alpha <= 1):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            warnings.warn(
                f"Depreciated alpha should be between 0 and 1 replaced by \
                    {alpha / (alpha + 1.0)}"
            )
            self.alpha = torch.tensor(alpha / (alpha + 1.0), dtype=torch.float32)
        self.min_margin_v = min_margin
        self.alpha_mean = alpha_mean

        self.current_mean = torch.tensor((self.min_margin_v,), dtype=torch.float32)
        """    constraint=lambda x: torch.clamp(x, 0.005, 1000),
            name="current_mean",
        )"""

        self.temperature = temperature * self.min_margin_v
        if alpha == 1.0:  # alpha = 1.0 => hinge only
            self.fct = self.multiclass_hinge_soft
        else:
            if alpha == 0.0:  # alpha = 0.0 => KR only
                self.fct = self.kr_soft
            else:
                self.fct = self.hkr
        self.reduction = reduction

        super(SoftHKRMulticlassLoss, self).__init__()

    def clamp_current_mean(self, x):
        return torch.clamp(x, 0.005, 1000)

    def _update_mean(self, y_pred):
        self.current_mean = self.current_mean.to(y_pred.device)
        current_global_mean = torch.mean(torch.abs(y_pred)).to(
            dtype=self.current_mean.dtype
        )
        current_global_mean = (
            self.alpha_mean * self.current_mean
            + (1 - self.alpha_mean) * current_global_mean
        )
        self.current_mean = self.clamp_current_mean(current_global_mean).detach()
        total_mean = current_global_mean
        total_mean = torch.clamp(total_mean, self.min_margin_v, 20000)
        return total_mean

    def computeTemperatureSoftMax(self, y_true, y_pred):
        total_mean = self._update_mean(y_pred)
        current_temperature = (
            torch.clamp(self.temperature / total_mean, 0.005, 250)
            .to(dtype=y_pred.dtype)
            .detach()
        )
        min_value = torch.tensor(torch.finfo(torch.float32).min, dtype=y_pred.dtype).to(
            device=y_pred.device
        )
        opposite_values = torch.where(
            y_true > 0, min_value, current_temperature * y_pred
        )
        F_soft_KR = torch.softmax(opposite_values, dim=-1)
        one_value = torch.tensor(1.0, dtype=F_soft_KR.dtype).to(device=y_pred.device)
        F_soft_KR = torch.where(y_true > 0, one_value, F_soft_KR)
        return F_soft_KR

    def signed_y_pred(self, y_true, y_pred):
        """Return for each item sign(y_true)*y_pred."""
        sign_y_true = torch.where(y_true > 0, 1, -1)  # switch to +/-1
        sign_y_true = sign_y_true.to(dtype=y_pred.dtype)
        return y_pred * sign_y_true

    def multiclass_hinge_preproc(self, signed_y_pred, min_margin):
        """From multiclass_hinge(y_true, y_pred, min_margin)
        simplified to use precalculated signed_y_pred"""
        # compute the elementwise hinge term
        hinge = torch.nn.functional.relu(min_margin / 2.0 - signed_y_pred)
        return hinge

    def multiclass_hinge_soft_preproc(self, signed_y_pred, F_soft_KR):
        hinge = self.multiclass_hinge_preproc(signed_y_pred, self.min_margin_v)
        b = hinge * F_soft_KR
        b = torch.sum(b, axis=-1)
        return b

    def multiclass_hinge_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)

    def kr_soft_preproc(self, signed_y_pred, F_soft_KR):
        kr = -signed_y_pred
        a = kr * F_soft_KR
        a = torch.sum(a, axis=-1)
        return a

    def kr_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.kr_soft_preproc(signed_y_pred, F_soft_KR)

    def hkr(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)

        loss_softkr = self.kr_soft_preproc(signed_y_pred, F_soft_KR)

        loss_softhinge = self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)
        return (1 - self.alpha) * loss_softkr + self.alpha * loss_softhinge

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (isinstance(input, torch.Tensor)):  # required for dtype.max
            input = torch.Tensor(input, dtype=input.dtype)
        if not (isinstance(target, torch.Tensor)):
            target = torch.Tensor(target, dtype=input.dtype)
        loss_batch = self.fct(target, input)
        return apply_reduction(loss_batch, self.reduction)
