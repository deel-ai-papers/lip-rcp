import torch.nn as nn
from utils.models.custom_cnn import CustomCNN
from utils.models.resnet import BatchCentering
from utils.models.liplayers import LipAvgPool2d
from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.layers.linear import UnitNormLinear
from orthogonium.layers.custom_activations import MaxMin
from orthogonium.layers.residual import PrescaledAdditiveResidual
from orthogonium.model_factory.models_factory import StagedCNN
from orthogonium.reparametrizers import QR_ORTHO_PARAMS


def cifar10_model(num_classes, p_dropout):
    model = StagedCNN(
        img_shape=(3, 32, 32),
        dim_repeats=[(512, 4), (1024, 6)],
        dim_nb_dense=(1024, 1),
        n_classes=num_classes,
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=True,
            padding_mode="zeros",
            kernel_size=3,
            padding=1,
        ),
        act=ClassParam(MaxMin),
        pool=LipAvgPool2d,
        lin=ClassParam(UnitNormLinear, bias=True),
        norm=BatchCentering,
    )
    return model


def cifar100_model(num_classes, p_dropout):
    model = StagedCNN(
        img_shape=(3, 32, 32),
        dim_repeats=[(512, 4), (1024, 6)],
        dim_nb_dense=(1024, 1),
        n_classes=num_classes,
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=True,
            padding_mode="zeros",
            kernel_size=3,
            padding=1,
        ),
        act=ClassParam(MaxMin),
        pool=LipAvgPool2d,
        lin=ClassParam(UnitNormLinear, bias=True),
        norm=BatchCentering,
    )
    return model


def imagenet_model(num_classes=1000, p_dropout=0.0):
    """
    Slightly smaller model than in paper for faster training with larger batch size
    """
    from orthogonium.model_factory.models_factory import MODELS
    model = MODELS["SplitConcatNet-M2"](
        img_shape=(3, 224, 224), n_classes=1000
    )
    return model
