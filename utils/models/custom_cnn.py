import torch
import torch.nn as nn
from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.layers.linear import UnitNormLinear, OrthoLinear
from orthogonium.layers.custom_activations import MaxMin
from orthogonium.model_factory.models_factory import StagedCNN
from .liplayers import (
    LipAvgPool2d,
    LipAdaptiveAvgPool2d,
    BatchCentering,
    ScaleFactor,
)

def CustomCNN(
    img_shape=(3, 32, 32),
    dim_repeats=[(64, 2), (128, 2)],
    dim_nb_dense=(256, 2),
    n_classes=10,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
    ),
    act=ClassParam(MaxMin),
    pool=LipAvgPool2d,
    lin=UnitNormLinear,
    norm=BatchCentering,
):
    layers = []
    in_channels = img_shape[0]

    # enrich the list of dim_repeats with the next number of channels in the list
    for i in range(len(dim_repeats) - 1):
        dim_repeats[i] = (dim_repeats[i][0], dim_repeats[i][1], dim_repeats[i + 1][0])
    dim_repeats[-1] = (dim_repeats[-1][0], dim_repeats[-1][1], None)

    # Create convolutional blocks
    for dim, repeats, next_dim in dim_repeats:
        # Add repeated conv layers
        for _ in range(repeats):
            layers.append(conv(in_channels=in_channels, out_channels=dim))
            layers.append(norm(num_features = dim) if norm is not None else nn.Identity())
            layers.append(act())
            in_channels = dim

        if next_dim is not None:
            # Add strided convolution to separate blocks
            layers.append(
                conv(
                    in_channels=dim,
                    out_channels=next_dim,
                    stride=2,
                )
            )
            layers.append(norm(num_features = next_dim) if norm is not None else nn.Identity())
            layers.append(act())
            in_channels = next_dim

    feat_shape = img_shape[-1] // (2 ** (len(dim_repeats) - 1))
    if pool is not None:
        layers.append(pool(kernel_size=feat_shape))
        feat_shape = 1

    # Flatten layer
    layers.append(LipAdaptiveAvgPool2d((1,1)))
    layers.append(nn.Flatten())

    nb_features = dim
    if dim_nb_dense is not None and len(dim_nb_dense) > 0:
        # Add linear layers
        dim, repeats = dim_nb_dense
        for _ in range(repeats):
            layers.append(lin(nb_features, dim))
            layers.append(norm(num_features = dim) if norm is not None else nn.Identity())
            layers.append(act())
            nb_features = dim
    else:
        dim = nb_features
    # Final linear layer for classification
    layers.append(lin(dim, n_classes))
    return nn.Sequential(*layers)
