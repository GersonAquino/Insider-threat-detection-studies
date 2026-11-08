import torch
from torch import nn

from src.itd_tcnTransformer.config.settings import (
    DROPOUT,
    HIDDEN_DIM,
    PREDICT_ALL_DAYS,
    POSITIONAL_ENCODING_TYPE,
    TCN_ACTIVATION,
    TCN_DILATIONS,
    TCN_KERNEL_SIZE,
    TCN_NORM_POSITION,
    TCN_USE_LAYER_NORM,
    TRANSFORMER_ACTIVATION,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    USE_CAUSAL_TRANSFORMER_MASK,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.models.causal_transformer import CausalTransformerEncoder
from src.itd_tcnTransformer.models.positional_encoding import PositionalEncoding
from src.itd_tcnTransformer.models.tcn_blocks import TCNBlock


__all__ = ["TCNTransformer", "TCNTransformerModel"]


class TCNTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = HIDDEN_DIM,
        tcn_dilations: list[int] = TCN_DILATIONS,
        kernel_size: int = TCN_KERNEL_SIZE,
        dropout: float = DROPOUT,
        tcn_activation: str = TCN_ACTIVATION,
        tcn_use_layer_norm: bool = TCN_USE_LAYER_NORM,
        tcn_norm_position: str = TCN_NORM_POSITION,
    ) -> None:
        super().__init__()

        self._validate_init_args(n_features, hidden_dim, tcn_dilations)

        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.tcn_dilations = [int(dilation) for dilation in tcn_dilations]
        self.tcn_activation = str(tcn_activation)
        self.tcn_use_layer_norm = bool(tcn_use_layer_norm)
        self.tcn_norm_position = str(tcn_norm_position)

        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock(
                    input_channels=(
                        self.n_features
                        if block_index == 0
                        else self.hidden_dim
                    ),
                    output_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation_name=self.tcn_activation,
                    use_layer_norm=self.tcn_use_layer_norm,
                    norm_position=self.tcn_norm_position,
                )
                for block_index, dilation in enumerate(self.tcn_dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        out = x

        for tcn_block in self.tcn_blocks:
            out = tcn_block(out)

        if out.shape[-1] != self.hidden_dim:
            raise ValueError(
                "TCNTransformer output channel count does not match hidden_dim. "
                f"Expected {self.hidden_dim}, got {out.shape[-1]}."
            )

        return out

    def _validate_init_args(
        self,
        n_features: int,
        hidden_dim: int,
        tcn_dilations: list[int],
    ) -> None:
        if n_features <= 0:
            raise ValueError(
                f"n_features must be positive. Current value: {n_features}."
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive. Current value: {hidden_dim}."
            )

        if not tcn_dilations:
            raise ValueError("tcn_dilations cannot be empty.")

        invalid_dilations = [
            dilation
            for dilation in tcn_dilations
            if dilation < 1
        ]

        if invalid_dilations:
            raise ValueError(
                "All TCN dilations must be at least 1. "
                f"Invalid values: {invalid_dilations}."
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "TCNTransformer expects input with shape "
                "(batch, time, features). "
                f"Current shape: {tuple(x.shape)}."
            )

        if x.shape[-1] != self.n_features:
            raise ValueError(
                "TCNTransformer input feature count does not match n_features. "
                f"Expected {self.n_features}, got {x.shape[-1]}."
            )


class TCNTransformerModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int = WINDOW_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        tcn_dilations: list[int] = TCN_DILATIONS,
        kernel_size: int = TCN_KERNEL_SIZE,
        tcn_activation: str = TCN_ACTIVATION,
        tcn_use_layer_norm: bool = TCN_USE_LAYER_NORM,
        tcn_norm_position: str = TCN_NORM_POSITION,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        feedforward_dim: int = TRANSFORMER_FF_DIM,
        transformer_activation: str = TRANSFORMER_ACTIVATION,
        positional_encoding_type: str = POSITIONAL_ENCODING_TYPE,
        dropout: float = DROPOUT,
        predict_all_days: bool = PREDICT_ALL_DAYS,
        use_causal_mask: bool = USE_CAUSAL_TRANSFORMER_MASK,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            n_features=n_features,
            window_size=window_size,
            hidden_dim=hidden_dim,
            tcn_dilations=tcn_dilations,
            kernel_size=kernel_size,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
        )

        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.tcn_dilations = [int(dilation) for dilation in tcn_dilations]
        self.kernel_size = int(kernel_size)
        self.tcn_activation = str(tcn_activation)
        self.tcn_use_layer_norm = bool(tcn_use_layer_norm)
        self.tcn_norm_position = str(tcn_norm_position)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.transformer_activation = str(transformer_activation)
        self.positional_encoding_type = str(positional_encoding_type)
        self.dropout = float(dropout)
        self.predict_all_days = bool(predict_all_days)
        self.use_causal_mask = bool(use_causal_mask)

        self.tcn_stack = TCNTransformer(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            tcn_dilations=self.tcn_dilations,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            tcn_activation=self.tcn_activation,
            tcn_use_layer_norm=self.tcn_use_layer_norm,
            tcn_norm_position=self.tcn_norm_position,
        )
        self.positional_encoding = PositionalEncoding(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            encoding_type=self.positional_encoding_type,
            dropout=self.dropout,
        )
        self.transformer_encoder = CausalTransformerEncoder(
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            activation_name=self.transformer_activation,
            use_causal_mask=self.use_causal_mask,
        )
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        out = self.tcn_stack(x)
        out = self.positional_encoding(out)
        out = self.transformer_encoder(out)
        logits = self.output_layer(out)

        if logits.shape != (x.shape[0], self.window_size, 1):
            raise ValueError(
                "TCNTransformerModel logits have invalid shape. "
                f"Expected {(x.shape[0], self.window_size, 1)}, "
                f"got {tuple(logits.shape)}."
            )

        if self.predict_all_days:
            return logits

        return logits[:, -1, 0]

    def _validate_init_args(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int,
        tcn_dilations: list[int],
        kernel_size: int,
        num_heads: int,
        feedforward_dim: int,
    ) -> None:
        if n_features <= 0:
            raise ValueError(
                f"n_features must be positive. Current value: {n_features}."
            )

        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive. Current value: {window_size}."
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive. Current value: {hidden_dim}."
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive. Current value: {num_heads}."
            )

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads. "
                f"hidden_dim={hidden_dim}, num_heads={num_heads}."
            )

        if not tcn_dilations:
            raise ValueError("tcn_dilations cannot be empty.")

        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive. Current value: {kernel_size}."
            )

        if feedforward_dim <= 0:
            raise ValueError(
                "feedforward_dim must be positive. "
                f"Current value: {feedforward_dim}."
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "TCNTransformerModel expects input with shape "
                "(batch, WINDOW_SIZE, n_features). "
                f"Current shape: {tuple(x.shape)}."
            )

        if x.shape[1] != self.window_size:
            raise ValueError(
                "TCNTransformerModel input temporal length does not match "
                "window_size. "
                f"Expected {self.window_size}, got {x.shape[1]}."
            )

        if x.shape[-1] != self.n_features:
            raise ValueError(
                "TCNTransformerModel input feature count does not match "
                "n_features. "
                f"Expected {self.n_features}, got {x.shape[-1]}."
            )
