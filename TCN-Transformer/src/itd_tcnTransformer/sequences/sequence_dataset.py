from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.itd_tcnTransformer.config.paths import SEQUENCES_DIR
from src.itd_tcnTransformer.config.settings import (
    BATCH_SIZE,
    DEVICE,
    PREDICT_ALL_DAYS,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.utils.reproducibility import (
    create_torch_generator,
    seed_worker,
)


SPLIT_NAMES = ("train", "validation", "test")
ARTICLE_SPLIT_NAMES = (
    "normal_pretrain",
    "finetune_train",
    "finetune_validation",
    "test",
)
TRAIN_SPLIT_NAME = "train"
ARTICLE_SHUFFLE_SPLIT_NAMES = {"normal_pretrain", "finetune_train"}
SEQUENCE_X_FILE_SUFFIX = "_X.pt"
SEQUENCE_Y_FILE_SUFFIX = "_y.pt"
SEQUENCE_METADATA_FILE_SUFFIX = "_metadata.parquet"
DEFAULT_NUM_WORKERS = 0

__all__ = [
    "SequenceDataset",
    "create_sequence_dataloader",
    "create_sequence_dataloaders",
    "load_sequence_dataset",
]


class SequenceDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        metadata: pd.DataFrame | None = None,
        return_metadata: bool = False,
        predict_all_days: bool = PREDICT_ALL_DAYS,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.metadata = (
            metadata.reset_index(drop=True)
            if metadata is not None
            else None
        )
        self.return_metadata = return_metadata
        self.predict_all_days = predict_all_days
        self.window_size = int(window_size)

        self._validate_shapes()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(
        self,
        index: int | np.integer | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        item_index = _normalize_item_index(index)
        X_item = self.X[item_index]
        y_item = self.y[item_index]

        if not self.return_metadata:
            return X_item, y_item

        if self.metadata is None:
            raise ValueError("Metadata requested, but no metadata was provided.")

        metadata_item = {
            str(column): value
            for column, value in self.metadata.iloc[item_index].to_dict().items()
        }

        return X_item, y_item, metadata_item

    def _validate_shapes(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(
                "X must have shape (n_windows, WINDOW_SIZE, n_features). "
                f"Current shape: {tuple(self.X.shape)}"
            )

        if self.X.shape[1] != self.window_size:
            raise ValueError(
                "X must have WINDOW_SIZE rows per sequence. "
                f"Expected {self.window_size}, got {self.X.shape[1]}."
            )

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "X and y must have the same number of windows. "
                f"X has {self.X.shape[0]}, y has {self.y.shape[0]}."
            )

        if self.predict_all_days:
            self._validate_predict_all_days_y_shape()
        else:
            self._validate_last_day_y_shape()

        if self.metadata is not None and len(self.metadata.index) != len(self):
            raise ValueError(
                "Metadata must have one row per sequence window. "
                f"Metadata has {len(self.metadata.index)}, dataset has {len(self)}."
            )

    def _validate_predict_all_days_y_shape(self) -> None:
        expected_shape_tail = (self.window_size, 1)

        if self.y.ndim != 3 or tuple(self.y.shape[1:]) != expected_shape_tail:
            raise ValueError(
                "When PREDICT_ALL_DAYS=True, y must have shape "
                "(n_windows, WINDOW_SIZE, 1). "
                f"Current shape: {tuple(self.y.shape)}"
            )

    def _validate_last_day_y_shape(self) -> None:
        if self.y.ndim != 1:
            raise ValueError(
                "When PREDICT_ALL_DAYS=False, y must have shape (n_windows,). "
                f"Current shape: {tuple(self.y.shape)}"
            )


def create_sequence_dataloader(
    split_name: str,
    batch_size: int = BATCH_SIZE,
    sequences_dir: Path = SEQUENCES_DIR,
    return_metadata: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
    predict_all_days: bool = PREDICT_ALL_DAYS,
    seed: int = RANDOM_SEED,
    use_seeded_generator: bool = True,
    window_size: int = WINDOW_SIZE,
) -> DataLoader:
    dataset = load_sequence_dataset(
        split_name=split_name,
        sequences_dir=sequences_dir,
        return_metadata=return_metadata,
        predict_all_days=predict_all_days,
        window_size=window_size,
    )
    shuffle = _should_shuffle_split(split_name)
    generator = (
        create_torch_generator(seed)
        if shuffle and use_seeded_generator
        else None
    )
    worker_init_fn = seed_worker if int(num_workers) > 0 else None

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=_should_pin_memory(pin_memory),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def create_sequence_dataloaders(
    batch_size: int = BATCH_SIZE,
    sequences_dir: Path = SEQUENCES_DIR,
    return_metadata: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
    predict_all_days: bool = PREDICT_ALL_DAYS,
    seed: int = RANDOM_SEED,
    use_seeded_generator: bool = True,
    window_size: int = WINDOW_SIZE,
) -> dict[str, DataLoader]:
    return {
        split_name: create_sequence_dataloader(
            split_name=split_name,
            batch_size=batch_size,
            sequences_dir=sequences_dir,
            return_metadata=return_metadata,
            num_workers=num_workers,
            pin_memory=pin_memory,
            predict_all_days=predict_all_days,
            seed=seed,
            use_seeded_generator=use_seeded_generator,
            window_size=window_size,
        )
        for split_name in _get_split_names()
    }


def load_sequence_dataset(
    split_name: str,
    sequences_dir: Path = SEQUENCES_DIR,
    return_metadata: bool = False,
    predict_all_days: bool = PREDICT_ALL_DAYS,
    window_size: int = WINDOW_SIZE,
) -> SequenceDataset:
    _validate_split_name(split_name)

    X_path = sequences_dir / f"{split_name}{SEQUENCE_X_FILE_SUFFIX}"
    y_path = sequences_dir / f"{split_name}{SEQUENCE_Y_FILE_SUFFIX}"
    metadata_path = sequences_dir / f"{split_name}{SEQUENCE_METADATA_FILE_SUFFIX}"

    X = _load_tensor_file(X_path)
    y = _load_tensor_file(y_path)
    metadata = (
        _load_metadata_file(metadata_path)
        if return_metadata
        else None
    )

    return SequenceDataset(
        X=X,
        y=y,
        metadata=metadata,
        return_metadata=return_metadata,
        predict_all_days=predict_all_days,
        window_size=window_size,
    )


def _load_metadata_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Sequence metadata file not found: {file_path}")

    return pd.read_parquet(file_path)


def _load_tensor_file(file_path: Path) -> torch.Tensor:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Sequence tensor file not found: {file_path}")

    tensor = torch.load(file_path, map_location="cpu")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Sequence file does not contain a tensor: {file_path}")

    return tensor


def _normalize_item_index(index: int | np.integer | torch.Tensor) -> int:
    if isinstance(index, torch.Tensor):
        if index.ndim != 0:
            raise TypeError("SequenceDataset only supports scalar tensor indexes.")

        return int(index.item())

    return int(index)


def _should_pin_memory(pin_memory: bool | None) -> bool:
    if pin_memory is not None:
        return bool(pin_memory)

    return str(DEVICE).startswith("cuda")


def _should_shuffle_split(split_name: str) -> bool:
    _validate_split_name(split_name)

    if SPLIT_STRATEGY == "user_article_two_phase":
        return split_name in ARTICLE_SHUFFLE_SPLIT_NAMES

    return split_name == TRAIN_SPLIT_NAME


def _validate_split_name(split_name: str) -> None:
    split_names = _get_split_names()

    if split_name not in split_names:
        raise ValueError(
            f"Invalid split name: {split_name}. "
            f"Expected one of: {', '.join(split_names)}"
        )


def _get_split_names() -> tuple[str, ...]:
    if SPLIT_STRATEGY == "user_article_two_phase":
        return ARTICLE_SPLIT_NAMES

    return SPLIT_NAMES
