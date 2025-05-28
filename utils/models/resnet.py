import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.distributed as dist

try:  # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except:  # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url

from orthogonium.layers import AdaptiveOrthoConv2d, MaxMin, UnitNormLinear
from orthogonium.model_factory.classparam import ClassParam
import orthogonium.layers as ol

from .liplayers import (
    LipAvgPool2d,
    LipAdaptiveAvgPool2d,
    BatchCentering,
    ScaleFactor,
)
from orthogonium.model_factory.classparam import ClassParam
from orthogonium.reparametrizers import OrthoParams
from orthogonium.reparametrizers import (
    DEFAULT_ORTHO_PARAMS,
    BJORCK_PASS_THROUGH_ORTHO_PARAMS,
    QR_ORTHO_PARAMS,
    EXP_ORTHO_PARAMS,
    CHOLESKY_ORTHO_PARAMS,
    BatchedPowerIteration,
    BatchedBjorckOrthogonalization,
)

CUSTOM_BJORCK_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-4),  # type: ignore
    orthogonalizer=ClassParam(BatchedBjorckOrthogonalization, beta=0.5, niters=10),
)

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return AdaptiveOrthoConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        ortho_params=CUSTOM_BJORCK_PARAMS,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return AdaptiveOrthoConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, ortho_params=CUSTOM_BJORCK_PARAMS
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        p_dropout=0,
    ):
        super(BasicBlock, self).__init__()
        norm_layer = BatchCentering
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.act1 = MaxMin()
        self.act2 = MaxMin()
        self.conv2 = conv3x3(planes, planes)
        self.bo = norm_layer(planes)
        self.stride = stride
        self.downsample = downsample
        self.coef = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout2d(p_dropout) if p_dropout != 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        alpha = F.sigmoid(self.coef)
        out = alpha * out + (1 - alpha) * identity
        out = self.dropout(out)
        out = self.bo(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        p_dropout=0,
    ):
        super(Bottleneck, self).__init__()
        norm_layer = BatchCentering
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bo = norm_layer(planes * self.expansion)
        self.stride = stride
        self.act1 = MaxMin()
        self.act2 = MaxMin()
        self.act3 = MaxMin()
        self.downsample = downsample
        self.coef = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout2d(p_dropout) if p_dropout != 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        alpha = F.sigmoid(self.coef)
        out = alpha * out + (1 - alpha) * identity
        out = self.dropout(out)
        out = self.bo(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes,
        p_dropout,
        scaling_factor=False,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
    ):
        super(ResNet, self).__init__()
        norm = BatchCentering
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.p_dropout = p_dropout

        self.groups = groups
        self.base_width = width_per_group
        self.feature_extractor = nn.Sequential(
            AdaptiveOrthoConv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, ortho_params=CUSTOM_BJORCK_PARAMS
            ),
            BatchCentering(self.inplanes),
            MaxMin(),
        )

        self.scaling = scaling_factor
        self.scale0 = ScaleFactor() if scaling_factor else nn.Identity()
        self.scale1 = ScaleFactor() if scaling_factor else nn.Identity()
        self.scale2 = ScaleFactor() if scaling_factor else nn.Identity()
        self.scale3 = ScaleFactor() if scaling_factor else nn.Identity()
        self.scale4 = ScaleFactor() if scaling_factor else nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.bln1 = (
            norm(64) if isinstance(block, BasicBlock) else norm(64 * block.expansion)
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.bln2 = (
            norm(128) if isinstance(block, BasicBlock) else norm(128 * block.expansion)
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.bln3 = (
            norm(256) if isinstance(block, BasicBlock) else norm(256 * block.expansion)
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bln4 = (
            norm(512) if isinstance(block, BasicBlock) else norm(512 * block.expansion)
        )

        if p_dropout == 0:
            self.dropout = nn.Identity()
        elif p_dropout < 0.3:
            self.dropout = nn.Dropout(p_dropout * 2)
        else:
            self.dropout = nn.Dropout(p_dropout)

        self.avgpool = LipAdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = UnitNormLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride),
                BatchCentering(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                self.p_dropout,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    p_dropout=self.p_dropout,
                )
            )

        return nn.Sequential(*layers)

    def lipconstant(self):
        lipconstant = (
            self.scale0.factor.clamp(min=0, max=self.scale0.max_value)
            * self.scale1.factor.clamp(min=0, max=self.scale1.max_value)
            * self.scale2.factor.clamp(min=0, max=self.scale2.max_value)
            * self.scale3.factor.clamp(min=0, max=self.scale3.max_value)
            * self.scale4.factor.clamp(min=0, max=self.scale4.max_value)
        )
        return lipconstant

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.scale0(x)

        x = self.layer1(x)
        x = self.scale1(x)
        x = self.bln1(x)

        x = self.layer2(x)
        x = self.scale2(x)
        x = self.bln2(x)

        x = self.layer3(x)
        x = self.scale3(x)
        x = self.bln3(x)

        x = self.layer4(x)
        x = self.scale4(x)
        x = self.bln4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        if self.scaling:
            lipconstant = self.lipconstant()
        else:
            lipconstant = 1.0

        return x / lipconstant


def _resnet(
    arch, block, layers, num_classes, p_dropout, pretrained, progress, **kwargs
):
    model = ResNet(block, layers, num_classes, p_dropout, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnet34(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnet50(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnet101(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnet152(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnext50_32x4d(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def resnext101_32x8d(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def wide_resnet50_2(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )


def wide_resnet101_2(num_classes, p_dropout, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        p_dropout,
        pretrained,
        progress,
        **kwargs,
    )
