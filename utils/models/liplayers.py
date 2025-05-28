import math
import torch
import torch.nn as nn
from typing import Optional
import torch.distributed as dist


class BatchCentering(nn.Module):
    r"""
    Applies Batch Centering  over a 2D, 3D, 4D input.

    .. math::

        y = x - \mathrm{E}[x] + \beta

    The mean is calculated per-dimension over the mini-batchesa and
    other dimensions excepted the feature/channel dimension.
    This layer uses statistics computed from input data in
    training mode and  a constant in evaluation mode computed as
    the running mean on training samples.
    :math:`\beta` is a learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input).
    that can be applied after the mean subtraction.
    Unlike Batch Normalization, this layer is 1-Lipschitz

    Args:
        size: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((0, -2, -1))`` for a 4D tensor).
        momentum: the value used for the running mean computation
        bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`

    Shape:
        - Input: :math:`(N, size, *)`
        - Output: :math:`(N, size, *)` (same shape as input)

    """

    def __init__(
        self,
        num_features: int = 1,
        dim: Optional[tuple] = None,
        momentum: float = 0.05,
        bias: bool = True,
    ):
        super(BatchCentering, self).__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.first = True

    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        if self.training:
            mean = x.mean(dim=self.dim)
            with torch.no_grad():
                if self.first:
                    self.running_mean = mean
                    self.first = False
                else:
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mean
            if dist.is_initialized():
                dist.all_reduce(self.running_mean.detach(), op=dist.ReduceOp.SUM)
                self.running_mean /= dist.get_world_size()
        else:
            mean = self.running_mean
        if self.bias is not None:
            return x - mean.view(mean_shape) + self.bias.view(mean_shape)
        else:
            return x - mean.view(mean_shape)


class ScaleFactor(nn.Module):
    """
    Simply scale the input by a factor. Clamp this factor by max_value.
    """

    def __init__(self, max_value=100):
        super(ScaleFactor, self).__init__()
        self.max_value = max_value
        self.factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        factor = self.factor.clamp(min=0, max=self.max_value)
        return x * factor


class LipAvgPool2d(nn.AvgPool2d):
    """
    A 2D AvgPool layer that rescales its output so the overall operator
    is 1-Lipschitz under the L2 norm.
    """

    def __init__(self, kernel_size, *args, **kwargs):
        """
        Parameters
        ----------
        scale_factor : int
            Scale of the of the pooling kernel (= kernel_size and stride).
        """
        super(LipAvgPool2d, self).__init__(kernel_size, kernel_size, *args, **kwargs)
        self.scale_factor = kernel_size

        # Handle integer or tuple kernel_size
        k_h = self.scale_factor
        k_w = self.scale_factor

        # Compute the Lipschitz constant of the underlying avg-pool op
        self.lipschitz_const = 1.0 / math.sqrt(k_h * k_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the usual average pooling
        out = super(LipAvgPool2d, self).forward(x)
        # Divide by the Lipschitz constant so the resulting operation is 1-Lipschitz
        return out / self.lipschitz_const


class LipAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    An AdaptiveAvgPool2d that rescales its output so that the entire
    operation is 1-Lipschitz under the L2 norm.

    By default, we'll assume output_size=(1,1) for global pooling;
    otherwise, the Lipschitz constant depends on how many spatial
    elements each output pixel averages over.
    """

    def __init__(self, output_size=(1, 1)):
        assert output_size == (1, 1), "Only (1,1) case is implemented"
        super().__init__(output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with rescaling for 1-Lipschitz.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N, C, H, W)

        Returns
        -------
        torch.Tensor
            Pooled output of shape (N, C, outH, outW), scaled so
            that the map is 1-Lipschitz in the L2 norm.
        """
        # 1) Standard adaptive average pooling:
        out = super().forward(x)

        # 2) Determine how many spatial elements were averaged per output pixel
        N, C, H_in, W_in = x.shape
        H_out, W_out = 1, 1

        size_h = math.ceil(H_in / H_out)
        size_w = math.ceil(W_in / W_out)
        area = size_h * size_w

        # 3) The operator norm for averaging over `area` elements is 1/sqrt(area).
        scale = math.sqrt(area)

        # 4) Rescale
        out = out * scale
        return out
