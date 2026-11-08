import torch

from src.itd_tcnTransformer.config.settings import (
    HIDDEN_DIM,
    PREDICT_ALL_DAYS,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.models.causal_transformer import (
    CausalTransformerEncoder,
    create_causal_mask,
)
from src.itd_tcnTransformer.models.positional_encoding import PositionalEncoding
from src.itd_tcnTransformer.models.tcn_transformer import (
    TCNTransformer,
    TCNTransformerModel,
)


batch_size = 2
n_features = 10


def test_causal_mask_shape_and_visibility() -> None:
    causal_mask = create_causal_mask()

    assert causal_mask.shape == (WINDOW_SIZE, WINDOW_SIZE)
    assert causal_mask.dtype == torch.bool

    for current_day in range(WINDOW_SIZE):
        assert not causal_mask[current_day, current_day]

        for previous_day in range(current_day):
            assert not causal_mask[current_day, previous_day]

        for future_day in range(current_day + 1, WINDOW_SIZE):
            assert causal_mask[current_day, future_day]


def test_causal_transformer_encoder_shape() -> None:
    transformer_encoder = CausalTransformerEncoder()
    transformer_encoder.eval()
    x = torch.randn(batch_size, WINDOW_SIZE, HIDDEN_DIM)

    with torch.no_grad():
        out = transformer_encoder(x)

    assert out.shape == (batch_size, WINDOW_SIZE, HIDDEN_DIM)


def test_positional_encoding_shape() -> None:
    positional_encoding = PositionalEncoding()
    positional_encoding.eval()
    x = torch.randn(batch_size, WINDOW_SIZE, HIDDEN_DIM)

    with torch.no_grad():
        out = positional_encoding(x)

    assert out.shape == x.shape


def test_tcn_transformer_model_logits_shape() -> None:
    model = TCNTransformerModel(n_features=n_features)
    model.eval()
    x = torch.randn(batch_size, WINDOW_SIZE, n_features)

    with torch.no_grad():
        logits = model(x)

    if PREDICT_ALL_DAYS:
        assert logits.shape == (batch_size, WINDOW_SIZE, 1)
    else:
        assert logits.shape == (batch_size,)


def test_tcn_transformer_shape() -> None:
    tcn_transformer = TCNTransformer(n_features=n_features)
    tcn_transformer.eval()
    x = torch.randn(batch_size, WINDOW_SIZE, n_features)

    with torch.no_grad():
        out = tcn_transformer(x)

    assert out.shape == (batch_size, WINDOW_SIZE, HIDDEN_DIM)
