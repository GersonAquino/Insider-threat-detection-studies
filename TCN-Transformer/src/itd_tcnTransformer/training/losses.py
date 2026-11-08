from pathlib import Path

import torch
from torch import nn

from src.itd_tcnTransformer.config.paths import SEQUENCES_DIR
from src.itd_tcnTransformer.config.settings import (
    DEVICE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LOSS_TYPE,
    PREDICT_ALL_DAYS,
    USE_POS_WEIGHT,
    WINDOW_SIZE,
)


TRAIN_Y_FILE = "train_y.pt"

__all__ = [
    "BinaryFocalWithLogitsLoss",
    "BinarySequenceBCEWithLogitsLoss",
    "compute_pos_weight",
    "create_loss_fn",
]


class BinarySequenceBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        predict_all_days: bool = PREDICT_ALL_DAYS,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()

        self.predict_all_days = bool(predict_all_days)
        self.window_size = int(window_size)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _validate_loss_input_shapes(
            logits,
            y,
            self.predict_all_days,
            self.window_size,
        )

        return self.loss_fn(
            logits,
            y.to(device=logits.device, dtype=logits.dtype),
        )


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        gamma: float,
        reduction: str = "mean",
        predict_all_days: bool = PREDICT_ALL_DAYS,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()

        self._validate_init_args(alpha, gamma, reduction)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = str(reduction)
        self.predict_all_days = bool(predict_all_days)
        self.window_size = int(window_size)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _validate_loss_input_shapes(
            logits,
            y,
            self.predict_all_days,
            self.window_size,
        )
        y = y.to(device=logits.device, dtype=logits.dtype)
        _validate_binary_labels(y)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        p_t = torch.where(y.eq(1), probabilities, 1 - probabilities)
        alpha_t = torch.where(
            y.eq(1),
            torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device),
            torch.tensor(1 - self.alpha, dtype=logits.dtype, device=logits.device),
        )
        loss = alpha_t * torch.pow(1 - p_t, self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss

    def _validate_init_args(
        self,
        alpha: float,
        gamma: float,
        reduction: str,
    ) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(
                f"alpha must be greater than 0 and lower than 1. Current value: {alpha}."
            )

        if gamma < 0:
            raise ValueError(f"gamma cannot be negative. Current value: {gamma}.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Unsupported reduction: {reduction}. "
                'Expected "mean", "sum" or "none".'
            )

def compute_pos_weight(
    train_y_path: Path = SEQUENCES_DIR / TRAIN_Y_FILE,
    device: torch.device | str = DEVICE,
) -> torch.Tensor:
    train_y = _load_tensor_file(train_y_path)
    train_y = train_y.detach().to(device="cpu", dtype=torch.float32)

    _validate_binary_labels(train_y)

    positive_count = int(train_y.eq(1).sum().item())
    negative_count = int(train_y.eq(0).sum().item())

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Cannot compute pos_weight without both positive and negative "
            f"training labels. positive_count={positive_count}, "
            f"negative_count={negative_count}."
        )

    pos_weight = negative_count / positive_count

    return torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=torch.device(device),
    )


def create_loss_fn(
    train_y_path: Path = SEQUENCES_DIR / TRAIN_Y_FILE,
    loss_type: str = LOSS_TYPE,
    use_pos_weight: bool = USE_POS_WEIGHT,
    focal_alpha: float = FOCAL_ALPHA,
    focal_gamma: float = FOCAL_GAMMA,
    device: torch.device | str = DEVICE,
    predict_all_days: bool = PREDICT_ALL_DAYS,
    window_size: int = WINDOW_SIZE,
) -> nn.Module:
    if loss_type == "focal":
        if use_pos_weight:
            raise ValueError("use_pos_weight cannot be used when loss_type='focal'.")

        return BinaryFocalWithLogitsLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            predict_all_days=predict_all_days,
            window_size=window_size,
        )

    if loss_type != "bce":
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    pos_weight = (
        compute_pos_weight(train_y_path=train_y_path, device=device)
        if use_pos_weight
        else None
    )

    return BinarySequenceBCEWithLogitsLoss(
        pos_weight=pos_weight,
        predict_all_days=predict_all_days,
        window_size=window_size,
    )


def _validate_loss_input_shapes(
    logits: torch.Tensor,
    y: torch.Tensor,
    predict_all_days: bool,
    window_size: int,
) -> None:
    if logits.shape != y.shape:
        raise ValueError(
            "logits and y must have exactly the same shape. "
            f"logits shape: {tuple(logits.shape)}, "
            f"y shape: {tuple(y.shape)}."
        )

    if predict_all_days:
        expected_tail_shape = (window_size, 1)

        if logits.ndim != 3 or tuple(logits.shape[1:]) != expected_tail_shape:
            raise ValueError(
                "When PREDICT_ALL_DAYS=True, logits and y must have shape "
                "(batch, WINDOW_SIZE, 1). "
                f"Current shape: {tuple(logits.shape)}."
            )

        return

    if logits.ndim != 1:
        raise ValueError(
            "When PREDICT_ALL_DAYS=False, logits and y must have shape "
            "(batch,). "
            f"Current shape: {tuple(logits.shape)}."
        )


def _load_tensor_file(file_path: Path) -> torch.Tensor:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Training label tensor file not found: {file_path}")

    tensor = torch.load(file_path, map_location="cpu")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Training label file does not contain a tensor: {file_path}")

    return tensor


def _validate_binary_labels(labels: torch.Tensor) -> None:
    valid_label_mask = labels.eq(0) | labels.eq(1)

    if not bool(valid_label_mask.all()):
        raise ValueError("train_y.pt must contain only binary labels 0 and 1.")
