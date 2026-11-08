import torch
from torch import nn

from src.itd_tcnTransformer.config.settings import (
    DROPOUT,
    HIDDEN_DIM,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_ACTIVATION,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    USE_CAUSAL_TRANSFORMER_MASK,
    WINDOW_SIZE,
)


__all__ = ["CausalTransformerEncoder", "create_causal_mask"]


class CausalTransformerEncoder(nn.Module):
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        feedforward_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        activation_name: str = TRANSFORMER_ACTIVATION,
        use_causal_mask: bool = USE_CAUSAL_TRANSFORMER_MASK,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            window_size,
            hidden_dim,
            num_layers,
            num_heads,
            feedforward_dim,
            dropout,
            activation_name,
        )

        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.activation_name = str(activation_name)
        self.use_causal_mask = bool(use_causal_mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=float(dropout),
            activation=self.activation_name,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )
        self.register_buffer(
            "causal_mask",
            create_causal_mask(self.window_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        attention_mask = (
            self.causal_mask
            if self.use_causal_mask
            else None
        )
        out = self.encoder(x, mask=attention_mask)

        if out.shape != x.shape:
            raise ValueError(
                "CausalTransformerEncoder must preserve the input shape. "
                f"Input shape: {tuple(x.shape)}, output shape: {tuple(out.shape)}."
            )

        return out

    def _validate_init_args(
        self,
        window_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation_name: str,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive. Current value: {window_size}."
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive. Current value: {hidden_dim}."
            )

        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive. Current value: {num_layers}."
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

        if feedforward_dim <= 0:
            raise ValueError(
                "feedforward_dim must be positive. "
                f"Current value: {feedforward_dim}."
            )

        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"dropout must be between 0 and 1. Current value: {dropout}."
            )

        if activation_name not in {"relu", "gelu"}:
            raise ValueError(
                f"Unsupported Transformer activation: {activation_name}. "
                'Expected "relu" or "gelu".'
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "CausalTransformerEncoder expects input with shape "
                "(batch, WINDOW_SIZE, HIDDEN_DIM). "
                f"Current shape: {tuple(x.shape)}."
            )

        if x.shape[1] != self.window_size:
            raise ValueError(
                "CausalTransformerEncoder input temporal length does not "
                "match window_size. "
                f"Expected {self.window_size}, got {x.shape[1]}."
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "CausalTransformerEncoder input hidden dimension does not "
                "match hidden_dim. "
                f"Expected {self.hidden_dim}, got {x.shape[-1]}."
            )


def create_causal_mask(
    sequence_length: int = WINDOW_SIZE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_sequence_length(sequence_length)

    return torch.triu(
        torch.ones(
            (int(sequence_length), int(sequence_length)),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def _validate_sequence_length(sequence_length: int) -> None:
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive. "
            f"Current value: {sequence_length}."
        )
