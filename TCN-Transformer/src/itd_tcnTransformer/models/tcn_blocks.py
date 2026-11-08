import torch
import torch.nn.functional as F
from torch import nn

from src.itd_tcnTransformer.config.settings import (
    DROPOUT,
    TCN_ACTIVATION,
    TCN_KERNEL_SIZE,
    TCN_NORM_POSITION,
    TCN_USE_LAYER_NORM,
)


__all__ = ["TCNBlock"]


class TCNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = TCN_KERNEL_SIZE,
        dilation: int = 1,
        dropout: float = DROPOUT,
        activation_name: str = TCN_ACTIVATION,
        use_layer_norm: bool = TCN_USE_LAYER_NORM,
        norm_position: str = TCN_NORM_POSITION,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            input_channels,
            output_channels,
            kernel_size,
            dilation,
            dropout,
            activation_name,
            norm_position,
        )

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.causal_padding = (self.kernel_size - 1) * self.dilation
        self.activation_name = str(activation_name)
        self.use_layer_norm = bool(use_layer_norm)
        self.norm_position = str(norm_position)

        self.conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
        )
        self.activation = _create_activation(self.activation_name)
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm: nn.Module = (
            nn.LayerNorm(self.output_channels)
            if self.use_layer_norm
            else nn.Identity()
        )
        self.residual_projection: nn.Module = (
            nn.Identity()
            if self.input_channels == self.output_channels
            else nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.output_channels,
                kernel_size=1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        input_time_length = int(x.shape[1])

        x_t = x.transpose(1, 2)
        residual = self.residual_projection(x_t)

        out = F.pad(x_t, (self.causal_padding, 0))
        out = self.conv(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

        out = out + residual

        if out.shape[-1] != input_time_length:
            raise ValueError(
                "TCNBlock must preserve the temporal length. "
                f"Input length: {input_time_length}, "
                f"output length: {out.shape[-1]}."
            )

        return out.transpose(1, 2)

    def _validate_init_args(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation_name: str,
        norm_position: str,
    ) -> None:
        if input_channels <= 0:
            raise ValueError(
                f"input_channels must be positive. Current value: {input_channels}."
            )

        if output_channels <= 0:
            raise ValueError(
                f"output_channels must be positive. Current value: {output_channels}."
            )

        if kernel_size < 1:
            raise ValueError(
                f"kernel_size must be at least 1. Current value: {kernel_size}."
            )

        if dilation < 1:
            raise ValueError(
                f"dilation must be at least 1. Current value: {dilation}."
            )

        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"dropout must be between 0 and 1. Current value: {dropout}."
            )

        _create_activation(activation_name)

        if norm_position != "after_dropout_before_residual":
            raise ValueError(
                "norm_position currently supports only "
                '"after_dropout_before_residual". '
                f"Current value: {norm_position}."
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "TCNBlock expects input with shape (batch, time, channels). "
                f"Current shape: {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                "TCNBlock input channel count does not match input_channels. "
                f"Expected {self.input_channels}, got {x.shape[-1]}."
            )


def _create_activation(activation_name: str) -> nn.Module:
    if activation_name == "relu":
        return nn.ReLU()

    if activation_name == "gelu":
        return nn.GELU()

    raise ValueError(
        f"Unsupported TCN activation: {activation_name}. "
        'Expected "relu" or "gelu".'
    )
