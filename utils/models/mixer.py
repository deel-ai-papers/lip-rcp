import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import orthogonium.layers as ol
from orthogonium.layers.normalization import LipDyT
from orthogonium.reparametrizers import OrthoParams

from orthogonium.model_factory.classparam import ClassParam
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
    orthogonalizer=ClassParam(BatchedBjorckOrthogonalization, beta=0.5, niters=8),
)


class LipMean(nn.Module):
    def __init__(self, dim):
        super(LipMean, self).__init__()
        self.dim = dim

    def forward(self, x):
        og_size = x.size()
        x = x.mean(dim=self.dim) * math.sqrt(x.size(self.dim))
        return x


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        ortho_params: OrthoParams,
        p_dropout: float,
    ):
        """
        A simple MLP block: Linear -> GELU -> Linear.

        Args:
            in_features: The input/output dimension (equivalent to x.shape[-1] in Flax).
            mlp_dim: The hidden dimension in the intermediate MLP layer.
        """
        super().__init__()
        self.p_dropout = p_dropout
        self.fc1 = ol.OrthoLinear(in_features, mlp_dim, ortho_params=ortho_params)
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = ol.OrthoLinear(mlp_dim, in_features, ortho_params=ortho_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, sequence_length, in_features]
        """
        x = self.fc1(x)
        x = ol.MaxMin()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        seq_len: int,
        hidden_dim: int,
        ortho_params: OrthoParams,
        compression: float,
        p_dropout: float,
    ):
        """
        A single Mixer block, which performs token-mixing and channel-mixing.

        Args:
            tokens_mlp_dim: MLP dimension for token mixing.
            channels_mlp_dim: MLP dimension for channel mixing.
            seq_len: Number of tokens (e.g. patches).
            hidden_dim: The dimension of each token embedding.
        """
        super().__init__()

        # For token mixing: the MLP sees seq_len as "features"
        self.dyt1 = LipDyT(hidden_dim, compression=compression)
        self.token_mixing = MlpBlock(
            in_features=seq_len,
            mlp_dim=tokens_mlp_dim,
            ortho_params=ortho_params,
            p_dropout=p_dropout,
        )

        # For channel mixing: the MLP sees hidden_dim as "features"
        self.dyt2 = LipDyT(hidden_dim, compression=compression)
        self.channel_mixing = MlpBlock(
            in_features=hidden_dim,
            mlp_dim=channels_mlp_dim,
            ortho_params=ortho_params,
            p_dropout=p_dropout,
        )

        # Residual connection coefficients
        self.coefc = nn.Parameter(torch.ones(1))
        self.coeft = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, hidden_dim]
        """
        # Token mixing
        x = self.dyt1(x)
        y = x
        y = y.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        y = self.token_mixing(y)  # MLP on 'seq_len'
        y = y.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        alpha = F.sigmoid(self.coefc)
        x = (1 - alpha) * x + alpha * y  # Residual connection

        # Channel mixing
        x = self.dyt2(x)
        y = x
        y = self.channel_mixing(y)  # MLP on 'hidden_dim'
        beta = F.sigmoid(self.coeft)
        x = (1 - beta) * x + beta * y  # Residual connection
        return x


class LipMixer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        num_classes: int,
        num_blocks: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        model_name: str = None,
        ortho_method: str = "default",
        compression: float = 0.1,
        p_dropout: float = 0.0,
        p_stochastic: float = 0.0,
    ):
        """
        The MLP-Mixer model.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB).
            image_size: Input image resolution (height and width assumed equal).
            patches: An object with 'size' indicating patch size (e.g., 16).
            num_classes: Number of classes for the classification head.
            num_blocks: Number of MixerBlock layers.
            hidden_dim: The embedding dimension after the patch-embedding stem.
            tokens_mlp_dim: MLP dimension for token mixing.
            channels_mlp_dim: MLP dimension for channel mixing.
            model_name: (Optional) name of the model, purely for reference/logging.
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.depth = num_blocks
        self.p_stochastic_depth = p_stochastic

        dict_ortho = {
            "default": DEFAULT_ORTHO_PARAMS,
            "bjorck_pt": BJORCK_PASS_THROUGH_ORTHO_PARAMS,
            "qr": QR_ORTHO_PARAMS,
            "exp": EXP_ORTHO_PARAMS,
            "cholesky": CHOLESKY_ORTHO_PARAMS,
            "custom": CUSTOM_BJORCK_PARAMS,
        }
        if ortho_method not in dict_ortho.keys():
            raise ValueError("Invalid orthogonalization method")

        # Stem: Conv2d with kernel_size = stride = patches.size
        self.stem = ol.AdaptiveOrthoConv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )

        # Calculate how many patches in total = (H / patch_size) * (W / patch_size)
        num_patches = (image_size // patch_size) * (image_size // patch_size)

        # Stack MixerBlocks
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    seq_len=num_patches,
                    hidden_dim=hidden_dim,
                    ortho_params=dict_ortho[ortho_method],
                    compression=compression,
                    p_dropout=p_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # LayerNorm before the final head
        self.pre_head_dyt = LipDyT(hidden_dim)

        # Classification head
        if num_classes > 0:
            self.head = ol.UnitNormLinear(hidden_dim, num_classes)
        else:
            # if num_classes == 0, return embeddings
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Forward pass through the MLP-Mixer.

        Args:
            x: [batch_size, in_channels, height, width] input images.
            train: Flag to mimic the 'train' argument in Flax (unused in this PyTorch version).

        Returns:
            logits of shape [batch_size, num_classes] if num_classes > 0,
            else [batch_size, hidden_dim].
        """
        # Stem convolution -> [batch_size, hidden_dim, H/patch_size, W/patch_size]
        x = self.stem(x)

        # Flatten spatial dimensions and transpose -> [batch_size, num_patches, hidden_dim]
        batch_size, hidden_dim, h_prime, w_prime = x.shape
        x = x.view(
            batch_size, hidden_dim, h_prime * w_prime
        )  # [batch_size, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, hidden_dim]

        # Pass through Mixer blocks
        for block in self.blocks:
            if torch.rand((1)) < self.p_stochastic_depth:
                x = block(x)
            else:
                x = nn.Identity()(x)

        # LayerNorm, then average pool over tokens dimension
        x = self.pre_head_dyt(x)  # [batch_size, num_patches, hidden_dim]
        x = LipMean(dim=1)(x)  # [batch_size, hidden_dim]

        # Classification head (or identity if num_classes == 0)
        x = self.head(x)  # [batch_size, num_classes] or [batch_size, hidden_dim]
        lipconstant = self.get_lipconstant()

        return x / lipconstant

    def get_lipconstant(self):
        lipconstant = 1.0
        counter = 0
        for module in self.named_modules():
            if isinstance(module[1], LipDyT):
                lipconstant *= module[1].get_lipconstant()
                counter += 1
            else:
                pass
        assert counter == 2 * self.depth + 1, "Incorrect Lipschitz constant computation"
        return lipconstant


class ConvLipMixer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        kernel_size: int,
        patch_size: int,
        num_classes: int,
        num_blocks: int,
        hidden_dim_conv: int,
        hidden_dim_mixer: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        model_name: str = None,
        ortho_method: str = "default",
        compression: float = 0.1,
        p_dropout: float = 0.0,
        p_stochastic: float = 0.0,
    ):
        """
        The MLP-Mixer model.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB).
            image_size: Input image resolution (height and width assumed equal).
            patches: An object with 'size' indicating patch size (e.g., 16).
            num_classes: Number of classes for the classification head.
            num_blocks: Number of MixerBlock layers.
            hidden_dim: The embedding dimension after the patch-embedding stem.
            tokens_mlp_dim: MLP dimension for token mixing.
            channels_mlp_dim: MLP dimension for channel mixing.
            model_name: (Optional) name of the model, purely for reference/logging.
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.depth = num_blocks
        self.p_stochastic_depth = p_stochastic

        dict_ortho = {
            "default": DEFAULT_ORTHO_PARAMS,
            "bjorck_pt": BJORCK_PASS_THROUGH_ORTHO_PARAMS,
            "qr": QR_ORTHO_PARAMS,
            "exp": EXP_ORTHO_PARAMS,
            "cholesky": CHOLESKY_ORTHO_PARAMS,
            "custom": CUSTOM_BJORCK_PARAMS,
        }
        if ortho_method not in dict_ortho.keys():
            raise ValueError("Invalid orthogonalization method")

        # Stem: Conv2d with kernel_size = stride = patches.size
        self.conv_model = nn.Sequential(
            ol.AdaptiveOrthoConv2d(
                in_channels=in_channels,
                out_channels=hidden_dim_conv,
                kernel_size=2,
                stride=2,
                padding="valid",
            ),
            ol.MaxMin(),
            ol.AdaptiveOrthoConv2d(
                in_channels=hidden_dim_conv,
                out_channels=hidden_dim_conv,
                kernel_size=kernel_size,
            ),
            ol.AdaptiveOrthoConv2d(
                in_channels=hidden_dim_conv,
                out_channels=hidden_dim_conv,
                kernel_size=kernel_size,
                groups=hidden_dim_conv // 2,
            ),
            ol.MaxMin(),
            ol.AdaptiveOrthoConv2d(
                in_channels=hidden_dim_conv,
                out_channels=hidden_dim_conv,
                kernel_size=kernel_size,
            ),
            ol.AdaptiveOrthoConv2d(
                in_channels=hidden_dim_conv,
                out_channels=hidden_dim_conv,
                kernel_size=kernel_size,
                groups=hidden_dim_conv // 2,
            ),
            ol.MaxMin(),
            ol.AdaptiveOrthoConvTranspose2d(
                in_channels=hidden_dim_conv,
                out_channels=hidden_dim_conv,
                kernel_size=2,
                stride=2,
            ),
        )
        self.stem = ol.AdaptiveOrthoConv2d(
            in_channels=hidden_dim_conv,
            out_channels=hidden_dim_mixer,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )

        # Calculate how many patches in total = (H / patch_size) * (W / patch_size)
        num_patches = (image_size // patch_size) * (image_size // patch_size)

        # Stack MixerBlocks
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    seq_len=num_patches,
                    hidden_dim=hidden_dim_mixer,
                    ortho_params=dict_ortho[ortho_method],
                    compression=compression,
                    p_dropout=p_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # LayerNorm before the final head
        self.pre_head_dyt = LipDyT(hidden_dim_mixer)

        # Classification head
        if num_classes > 0:
            self.head = ol.UnitNormLinear(hidden_dim_mixer, num_classes)
        else:
            # if num_classes == 0, return embeddings
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Forward pass through the MLP-Mixer.

        Args:
            x: [batch_size, in_channels, height, width] input images.
            train: Flag to mimic the 'train' argument in Flax (unused in this PyTorch version).

        Returns:
            logits of shape [batch_size, num_classes] if num_classes > 0,
            else [batch_size, hidden_dim].
        """
        # Stem convolution -> [batch_size, hidden_dim, H/patch_size, W/patch_size]
        x = self.conv_model(x)
        x = self.stem(x)

        # Flatten spatial dimensions and transpose -> [batch_size, num_patches, hidden_dim]
        batch_size, hidden_dim, h_prime, w_prime = x.shape
        x = x.view(
            batch_size, hidden_dim, h_prime * w_prime
        )  # [batch_size, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, hidden_dim]

        # Pass through Mixer blocks
        for block in self.blocks:
            mask = (
                torch.rand(1, device=x.device) < self.p_stochastic_depth
            ) * self.training
            x = ~mask * block(x) + mask * x

        # LayerNorm, then average pool over tokens dimension
        x = self.pre_head_dyt(x)  # [batch_size, num_patches, hidden_dim]
        x = LipMean(dim=1)(x)  # [batch_size, hidden_dim]

        # Classification head (or identity if num_classes == 0)
        x = self.head(x)  # [batch_size, num_classes] or [batch_size, hidden_dim]
        lipconstant = self.get_lipconstant()

        return x / lipconstant

    def get_lipconstant(self):
        lipconstant = 1.0
        counter = 0
        for module in self.named_modules():
            if isinstance(module[1], LipDyT):
                lipconstant *= module[1].get_lipconstant()
                counter += 1
            else:
                pass
        assert counter == 2 * self.depth + 1, "Incorrect Lipschitz constant computation"
        return lipconstant
