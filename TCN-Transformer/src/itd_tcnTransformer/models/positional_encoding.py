import torch
from torch import nn

from src.itd_tcnTransformer.config.settings import (
    DROPOUT,
    HIDDEN_DIM,
    POSITIONAL_ENCODING_TYPE,
    WINDOW_SIZE,
)


__all__ = ["PositionalEncoding"]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        encoding_type: str = POSITIONAL_ENCODING_TYPE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self._validate_init_args(window_size, hidden_dim, encoding_type, dropout)

        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.encoding_type = str(encoding_type)

        self.position_embedding: nn.Embedding | None = None
        if self.encoding_type == "learned":
            self.position_embedding = nn.Embedding(
                num_embeddings=self.window_size,
                embedding_dim=self.hidden_dim,
            )

        self.register_buffer(
            "position_indexes",
            torch.arange(self.window_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "sinusoidal_encoding",
            _create_sinusoidal_encoding(self.window_size, self.hidden_dim),
            persistent=False,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        if self.encoding_type == "learned":
            if self.position_embedding is None:
                raise ValueError("Learned positional encoding was not initialized.")
            positional_values = self.position_embedding(self.position_indexes)
        else:
            positional_values = self.sinusoidal_encoding

        out = self.dropout(x + positional_values.unsqueeze(0))

        if out.shape != x.shape:
            raise ValueError(
                "PositionalEncoding must preserve the input shape. "
                f"Input shape: {tuple(x.shape)}, output shape: {tuple(out.shape)}."
            )

        return out

    def _validate_init_args(
        self,
        window_size: int,
        hidden_dim: int,
        encoding_type: str,
        dropout: float,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive. Current value: {window_size}."
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive. Current value: {hidden_dim}."
            )

        if encoding_type not in {"sinusoidal", "learned"}:
            raise ValueError(
                f"Unsupported positional encoding type: {encoding_type}. "
                'Expected "sinusoidal" or "learned".'
            )

        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"dropout must be between 0 and 1. Current value: {dropout}."
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "PositionalEncoding expects input with shape "
                "(batch, WINDOW_SIZE, HIDDEN_DIM). "
                f"Current shape: {tuple(x.shape)}."
            )

        if x.shape[1] != self.window_size:
            raise ValueError(
                "PositionalEncoding input temporal length does not match "
                "window_size. "
                f"Expected {self.window_size}, got {x.shape[1]}."
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "PositionalEncoding input hidden dimension does not match "
                "hidden_dim. "
                f"Expected {self.hidden_dim}, got {x.shape[-1]}."
            )


def _create_sinusoidal_encoding(window_size: int, hidden_dim: int) -> torch.Tensor:
    positions = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)
    div_terms = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
    )

    encoding = torch.zeros(window_size, hidden_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_terms)

    if hidden_dim > 1:
        encoding[:, 1::2] = torch.cos(
            positions * div_terms[:encoding[:, 1::2].shape[1]]
        )

    return encoding
