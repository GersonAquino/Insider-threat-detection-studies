from collections.abc import Sequence, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.itd_tcnTransformer.config.paths import MODELS_DIR
from src.itd_tcnTransformer.config.paths import SEQUENCES_DIR
from src.itd_tcnTransformer.config.settings import (
    CUBLAS_WORKSPACE_CONFIG,
    CUDNN_BENCHMARK,
    DEFAULT_THRESHOLD,
    DEVICE,
    DETERMINISTIC_TRAINING,
    DETERMINISTIC_WARN_ONLY,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EVALUATION_UNIT,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GRADIENT_CLIP_NORM,
    HIDDEN_DIM,
    LEARNING_RATE,
    LOSS_TYPE,
    MAX_EPOCHS,
    MODEL_DIM,
    PREDICT_ALL_DAYS,
    POSITIONAL_ENCODING_TYPE,
    PROFILE_FEATURE_MODE,
    OVERLAPPING_PREDICTION_AGGREGATION,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    TCN_ACTIVATION,
    TCN_DILATIONS,
    TCN_KERNEL_SIZE,
    TCN_NORM_POSITION,
    TCN_USE_LAYER_NORM,
    TRANSFORMER_ACTIVATION,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    TRAINING_MODE,
    USE_CAUSAL_TRANSFORMER_MASK,
    USE_POS_WEIGHT,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.evaluation.prediction_units import (
    build_evaluation_predictions,
    build_window_timestep_predictions,
)


BEST_CHECKPOINT_FILE = "best_model_checkpoint.pt"
PRETRAIN_CHECKPOINT_FILE = "pretrain_checkpoint.pt"
FINETUNE_BEST_CHECKPOINT_FILE = "finetune_best_checkpoint.pt"

CHECKPOINT_MONITOR_METRIC = "validation_auprc"
CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC = "best_validation_auprc"
CHECKPOINT_STRATEGY_LAST_EPOCH = "last_epoch"

VALID_CHECKPOINT_STRATEGIES = {
    CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
    CHECKPOINT_STRATEGY_LAST_EPOCH,
}

__all__ = [
    "CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC",
    "CHECKPOINT_STRATEGY_LAST_EPOCH",
    "Trainer",
    "TrainEpochMetrics",
    "TrainingHistory",
    "ValidationMetrics",
]


@dataclass
class TrainEpochMetrics:
    epoch: int
    train_loss: float
    batches: int
    examples: int
    validation: "ValidationMetrics | None" = None


@dataclass
class TrainingHistory:
    epochs: list[TrainEpochMetrics] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_auprc: float | None = None
    checkpoint_path: str | None = None
    early_stopped: bool = False
    stopped_epoch: int | None = None


@dataclass
class ValidationMetrics:
    validation_loss: float
    validation_auprc: float
    validation_auroc: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    evaluation_unit: str
    overlapping_prediction_aggregation: str
    prediction_count_before_aggregation: int
    prediction_count_after_aggregation: int
    threshold: float
    batches: int
    examples: int


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: torch.device | str = DEVICE,
        n_features: int | None = None,
        feature_columns: Sequence[str] | None = None,
        scaler_path: Path | str | None = None,
        checkpoint_path: Path | str = MODELS_DIR / BEST_CHECKPOINT_FILE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        gradient_clip_norm: float | None = GRADIENT_CLIP_NORM,
        threshold: float = DEFAULT_THRESHOLD,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        training_phase: str = "single_stage",
        validation_split_name: str = "validation",
        checkpoint_strategy: str = CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
        sequences_dir: Path | str = SEQUENCES_DIR,
        epoch_end_callback: Callable[[int, TrainEpochMetrics, Any], None] | None = None,
        loss_hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        self._validate_init_args(
            learning_rate,
            weight_decay,
            gradient_clip_norm,
            threshold,
            early_stopping_patience,
        )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.n_features = self._resolve_n_features(n_features)
        self.feature_columns = self._resolve_feature_columns(feature_columns)
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None
        self.checkpoint_path = Path(checkpoint_path)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip_norm = (
            None
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        )
        self.threshold = float(threshold)
        self.early_stopping_patience = int(early_stopping_patience)
        self.training_phase = str(training_phase)
        self.validation_split_name = str(validation_split_name)
        self.epoch_end_callback = epoch_end_callback
        self.loss_hyperparameters = (
            None
            if loss_hyperparameters is None
            else dict(loss_hyperparameters)
        )
        self.checkpoint_strategy = self._validate_checkpoint_strategy(
            checkpoint_strategy
        )
        self.sequences_dir = Path(sequences_dir)
        self.optimizer = self._create_optimizer()
        self.validation_labels = torch.empty(0, dtype=torch.float32)
        self.validation_probabilities = torch.empty(0, dtype=torch.float32)
        self.best_epoch: int | None = None
        self.best_validation_auprc: float | None = None

    def clip_gradients(self) -> torch.Tensor | None:
        if self.gradient_clip_norm is None:
            return None

        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip_norm,
        )

    def load_checkpoint(
        self,
        checkpoint_path: Path | str | None = None,
        restore_optimizer: bool = True,
    ) -> dict[str, Any]:
        checkpoint_file = (
            self.checkpoint_path
            if checkpoint_path is None
            else Path(checkpoint_path)
        )

        if not checkpoint_file.exists() or not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint payload must be a dictionary. "
                f"Current type: {type(checkpoint).__name__}."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if restore_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_epoch = int(checkpoint["best_epoch"])
        self.best_validation_auprc = float(
            checkpoint["best_metrics"][CHECKPOINT_MONITOR_METRIC]
        )

        return checkpoint

    def save_checkpoint(
        self,
        epoch: int,
        validation_metrics: ValidationMetrics,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        checkpoint_file = (
            self.checkpoint_path
            if checkpoint_path is None
            else Path(checkpoint_path)
        )
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            self._build_checkpoint_payload(epoch, validation_metrics),
            checkpoint_file,
        )

    def train(self, max_epochs: int = MAX_EPOCHS) -> TrainingHistory:
        self._validate_max_epochs(max_epochs)

        history = TrainingHistory(checkpoint_path=str(self.checkpoint_path))
        epochs_without_improvement = 0

        for epoch in range(1, int(max_epochs) + 1):
            train_epoch_metrics = self._run_train_epoch(epoch)
            validation_metrics = self.validate()
            epoch_metrics = TrainEpochMetrics(
                epoch=train_epoch_metrics.epoch,
                train_loss=train_epoch_metrics.train_loss,
                batches=train_epoch_metrics.batches,
                examples=train_epoch_metrics.examples,
                validation=validation_metrics,
            )
            history.epochs.append(epoch_metrics)

            if self.epoch_end_callback is not None:
                self.epoch_end_callback(epoch, epoch_metrics, self)

            if self.checkpoint_strategy == CHECKPOINT_STRATEGY_LAST_EPOCH:
                self.best_epoch = epoch
                self.best_validation_auprc = validation_metrics.validation_auprc
                self.save_checkpoint(epoch, validation_metrics)

                history.best_epoch = self.best_epoch
                history.best_validation_auprc = self.best_validation_auprc

                checkpoint_status = "last checkpoint saved"

            elif self._is_validation_auprc_improved(validation_metrics):
                self.best_epoch = epoch
                self.best_validation_auprc = validation_metrics.validation_auprc
                self.save_checkpoint(epoch, validation_metrics)

                history.best_epoch = self.best_epoch
                history.best_validation_auprc = self.best_validation_auprc
                epochs_without_improvement = 0

                checkpoint_status = "checkpoint saved"

            else:
                epochs_without_improvement += 1
                checkpoint_status = (
                    f"no improvement for {epochs_without_improvement}/"
                    f"{self.early_stopping_patience} epochs"
                )

            print(
                f"Epoch {epoch_metrics.epoch}: "
                f"train_loss={epoch_metrics.train_loss:.6f} "
                f"validation_loss={validation_metrics.validation_loss:.6f} "
                f"validation_auprc={validation_metrics.validation_auprc:.6f} "
                f"({checkpoint_status})"
            )

            if (
                self.checkpoint_strategy == CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                history.early_stopped = True
                history.stopped_epoch = epoch
                print(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epochs without validation "
                    "AUPRC improvement."
                )
                break

        self.load_checkpoint(self.checkpoint_path)

        return history
    
    def validate(self) -> ValidationMetrics:
        self.model.eval()

        total_loss = 0.0
        total_examples = 0
        total_batches = 0
        window_prediction_parts: list[pd.DataFrame] = []
        validation_metadata = self._load_validation_metadata()
        window_offset = 0

        with torch.no_grad():
            for batch in self.validation_dataloader:
                X, y = self._move_batch_to_device(batch)
                logits = self.model(X)
                self._validate_logit_label_shapes(logits, y)
                loss = self.loss_fn(logits, y)

                probabilities = torch.sigmoid(logits)

                batch_size = int(X.shape[0])
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_examples += batch_size
                total_batches += 1
                batch_metadata = validation_metadata.iloc[
                    window_offset:window_offset + batch_size
                ]
                window_prediction_parts.append(
                    build_window_timestep_predictions(
                        probabilities=probabilities.detach().cpu(),
                        labels=y.detach().cpu(),
                        metadata=batch_metadata,
                        predict_all_days=bool(PREDICT_ALL_DAYS),
                        split_name=self.validation_split_name,
                        window_offset=window_offset,
                    )
                )
                window_offset += batch_size

        if total_batches == 0 or total_examples == 0:
            raise ValueError("Validation DataLoader produced no batches.")

        window_predictions = pd.concat(window_prediction_parts, ignore_index=True)
        predictions_to_score = build_evaluation_predictions(
            window_predictions,
            evaluation_unit=EVALUATION_UNIT,
            aggregation=OVERLAPPING_PREDICTION_AGGREGATION,
        )
        self.validation_probabilities = torch.as_tensor(
            predictions_to_score["probability"].astype("float64").to_numpy().copy(),
            dtype=torch.float32,
        )
        self.validation_labels = torch.as_tensor(
            predictions_to_score["label"].astype("int64").to_numpy().copy(),
            dtype=torch.float32,
        )
        validation_loss = total_loss / total_examples

        return self._calculate_validation_metrics(
            validation_loss=float(validation_loss),
            total_batches=total_batches,
            total_examples=total_examples,
            probabilities=self.validation_probabilities,
            labels=self.validation_labels,
            prediction_count_before_aggregation=len(window_predictions.index),
            prediction_count_after_aggregation=len(predictions_to_score.index),
        )

    def _build_checkpoint_payload(
        self,
        epoch: int,
        validation_metrics: ValidationMetrics,
    ) -> dict[str, Any]:
        return {
            "best_epoch": int(epoch),
            "best_metrics": asdict(validation_metrics),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_hyperparameters": self._get_model_hyperparameters(),
            "n_features": int(self.n_features),
            "feature_columns": list(self.feature_columns),
            "scaler_path": (
                None
                if self.scaler_path is None
                else str(self.scaler_path)
            ),
            "loss_hyperparameters": self._get_loss_hyperparameters(),
            "training_hyperparameters": self._get_training_hyperparameters(),
            "reproducibility": self._get_reproducibility_configuration(),
            "configuration": self._get_relevant_configuration(),
            "training_mode": str(TRAINING_MODE),
            "split_strategy": str(SPLIT_STRATEGY),
            "training_phase": str(self.training_phase),
            "checkpoint_strategy": str(self.checkpoint_strategy),
            "evaluation_unit": str(EVALUATION_UNIT),
            "overlapping_prediction_aggregation": str(
                OVERLAPPING_PREDICTION_AGGREGATION
            ),
            "validation_split_name": str(self.validation_split_name),
            "validation_prediction_count_before_aggregation": int(
                validation_metrics.prediction_count_before_aggregation
            ),
            "validation_prediction_count_after_aggregation": int(
                validation_metrics.prediction_count_after_aggregation
            ),
        }

    def _calculate_validation_metrics(
        self,
        validation_loss: float,
        total_batches: int,
        total_examples: int,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        prediction_count_before_aggregation: int,
        prediction_count_after_aggregation: int,
    ) -> ValidationMetrics:
        y_score = probabilities.numpy()
        y_true_values = labels.numpy()
        self._validate_metric_labels(y_true_values)
        y_true = y_true_values.astype(np.int64)

        y_pred = (y_score >= self.threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        return ValidationMetrics(
            validation_loss=float(validation_loss),
            validation_auprc=float(average_precision_score(y_true, y_score)),
            validation_auroc=float(roc_auc_score(y_true, y_score)),
            validation_precision=float(precision),
            validation_recall=float(recall),
            validation_f1=float(f1),
            evaluation_unit=str(EVALUATION_UNIT),
            overlapping_prediction_aggregation=str(
                OVERLAPPING_PREDICTION_AGGREGATION
            ),
            prediction_count_before_aggregation=int(
                prediction_count_before_aggregation
            ),
            prediction_count_after_aggregation=int(
                prediction_count_after_aggregation
            ),
            threshold=float(self.threshold),
            batches=int(total_batches),
            examples=int(total_examples),
        )

    def _create_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _get_model_hyperparameters(self) -> dict[str, Any]:
        hidden_dim = int(self._get_model_attr("hidden_dim", HIDDEN_DIM))

        return {
            "n_features": int(self.n_features),
            "window_size": int(self._get_model_attr("window_size", WINDOW_SIZE)),
            "model_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "tcn_dilations": list(
                self._get_model_attr("tcn_dilations", TCN_DILATIONS)
            ),
            "kernel_size": int(self._get_model_attr("kernel_size", TCN_KERNEL_SIZE)),
            "tcn_activation": str(
                self._get_model_attr("tcn_activation", TCN_ACTIVATION)
            ),
            "tcn_use_layer_norm": bool(
                self._get_model_attr("tcn_use_layer_norm", TCN_USE_LAYER_NORM)
            ),
            "tcn_norm_position": str(
                self._get_model_attr("tcn_norm_position", TCN_NORM_POSITION)
            ),
            "num_layers": int(
                self._get_model_attr("num_layers", TRANSFORMER_NUM_LAYERS)
            ),
            "num_heads": int(
                self._get_model_attr("num_heads", TRANSFORMER_NUM_HEADS)
            ),
            "feedforward_dim": int(
                self._get_model_attr("feedforward_dim", TRANSFORMER_FF_DIM)
            ),
            "transformer_activation": str(
                self._get_model_attr(
                    "transformer_activation",
                    TRANSFORMER_ACTIVATION,
                )
            ),
            "positional_encoding_type": str(
                self._get_model_attr(
                    "positional_encoding_type",
                    POSITIONAL_ENCODING_TYPE,
                )
            ),
            "dropout": float(self._get_model_attr("dropout", DROPOUT)),
            "predict_all_days": bool(
                self._get_model_attr("predict_all_days", PREDICT_ALL_DAYS)
            ),
            "profile_feature_mode": str(PROFILE_FEATURE_MODE),
            "use_causal_mask": bool(
                self._get_model_attr(
                    "use_causal_mask",
                    USE_CAUSAL_TRANSFORMER_MASK,
                )
            ),
        }

    def _get_relevant_configuration(self) -> dict[str, Any]:
        hidden_dim = int(self._get_model_attr("hidden_dim", HIDDEN_DIM))
        loss_hyperparameters = self._get_loss_hyperparameters()

        return {
            "WINDOW_SIZE": int(WINDOW_SIZE),
            "MODEL_DIM": hidden_dim,
            "HIDDEN_DIM": hidden_dim,
            "TCN_DILATIONS": list(TCN_DILATIONS),
            "TCN_KERNEL_SIZE": int(TCN_KERNEL_SIZE),
            "TCN_ACTIVATION": str(TCN_ACTIVATION),
            "TCN_USE_LAYER_NORM": bool(TCN_USE_LAYER_NORM),
            "TCN_NORM_POSITION": str(TCN_NORM_POSITION),
            "TRANSFORMER_NUM_LAYERS": int(TRANSFORMER_NUM_LAYERS),
            "TRANSFORMER_NUM_HEADS": int(
                self._get_model_attr("num_heads", TRANSFORMER_NUM_HEADS)
            ),
            "TRANSFORMER_FF_DIM": int(
                self._get_model_attr("feedforward_dim", TRANSFORMER_FF_DIM)
            ),
            "TRANSFORMER_ACTIVATION": str(TRANSFORMER_ACTIVATION),
            "POSITIONAL_ENCODING_TYPE": str(POSITIONAL_ENCODING_TYPE),
            "DROPOUT": float(self._get_model_attr("dropout", DROPOUT)),
            "PREDICT_ALL_DAYS": bool(PREDICT_ALL_DAYS),
            "PROFILE_FEATURE_MODE": str(PROFILE_FEATURE_MODE),
            "USE_CAUSAL_TRANSFORMER_MASK": bool(USE_CAUSAL_TRANSFORMER_MASK),
            "LOSS_TYPE": str(loss_hyperparameters["loss_type"]),
            "FOCAL_ALPHA": float(loss_hyperparameters["focal_alpha"]),
            "FOCAL_GAMMA": float(loss_hyperparameters["focal_gamma"]),
            "USE_POS_WEIGHT": bool(loss_hyperparameters["use_pos_weight"]),
            "EVALUATION_UNIT": str(EVALUATION_UNIT),
            "OVERLAPPING_PREDICTION_AGGREGATION": str(
                OVERLAPPING_PREDICTION_AGGREGATION
            ),
            "RANDOM_SEED": int(RANDOM_SEED),
            "DETERMINISTIC_TRAINING": bool(DETERMINISTIC_TRAINING),
            "DETERMINISTIC_WARN_ONLY": bool(DETERMINISTIC_WARN_ONLY),
            "CUDNN_BENCHMARK": bool(CUDNN_BENCHMARK),
            "CUBLAS_WORKSPACE_CONFIG": str(CUBLAS_WORKSPACE_CONFIG),
        }

    def _get_loss_hyperparameters(self) -> dict[str, Any]:
        if self.loss_hyperparameters is not None:
            return dict(self.loss_hyperparameters)

        return {
            "loss_type": str(LOSS_TYPE),
            "focal_alpha": float(FOCAL_ALPHA),
            "focal_gamma": float(FOCAL_GAMMA),
            "use_pos_weight": bool(USE_POS_WEIGHT),
        }

    def _get_reproducibility_configuration(self) -> dict[str, Any]:
        return {
            "random_seed": int(RANDOM_SEED),
            "deterministic_training": bool(DETERMINISTIC_TRAINING),
            "deterministic_warn_only": bool(DETERMINISTIC_WARN_ONLY),
            "cudnn_benchmark": bool(CUDNN_BENCHMARK),
            "cublas_workspace_config": str(CUBLAS_WORKSPACE_CONFIG),
        }

    def _get_training_hyperparameters(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": self.gradient_clip_norm,
            "threshold": float(self.threshold),
            "early_stopping_patience": int(self.early_stopping_patience),
            "monitor_metric": CHECKPOINT_MONITOR_METRIC,
            "checkpoint_strategy": str(self.checkpoint_strategy),
            "validation_split_name": str(self.validation_split_name),
        }

    def _get_model_attr(self, attr_name: str, default: Any) -> Any:
        return getattr(self.model, attr_name, default)

    def _is_validation_auprc_improved(
        self,
        validation_metrics: ValidationMetrics,
    ) -> bool:
        validation_auprc = float(validation_metrics.validation_auprc)

        if not np.isfinite(validation_auprc):
            raise ValueError(
                "validation_auprc must be finite for checkpointing and "
                f"early stopping. Current value: {validation_auprc}."
            )

        return (
            self.best_validation_auprc is None
            or validation_auprc > self.best_validation_auprc
        )

    def _load_validation_metadata(self) -> pd.DataFrame:
        metadata_path = self.sequences_dir / f"{self.validation_split_name}_metadata.parquet"

        if not metadata_path.exists() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Validation metadata file not found: {metadata_path}"
            )

        metadata = pd.read_parquet(metadata_path)
        validation_dataset = self.validation_dataloader.dataset
        if not isinstance(validation_dataset, Sized):
            raise TypeError("Validation DataLoader dataset must expose __len__.")

        dataset_length = len(validation_dataset)

        if len(metadata.index) != dataset_length:
            raise ValueError(
                "Validation metadata must have one row per validation sequence. "
                f"Metadata rows: {len(metadata.index)}, sequences: {dataset_length}."
            )

        return metadata.reset_index(drop=True)

    def _move_batch_to_device(
        self,
        batch: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = batch[0].to(self.device, dtype=torch.float32)
        y = batch[1].to(self.device, dtype=torch.float32)

        return X, y

    def _run_train_epoch(self, epoch: int) -> TrainEpochMetrics:
        self.model.train()

        total_loss = 0.0
        total_examples = 0
        total_batches = 0

        for batch in self.train_dataloader:
            X, y = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()
            logits = self.model(X)
            self._validate_logit_label_shapes(logits, y)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.clip_gradients()
            self.optimizer.step()

            batch_size = int(X.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size
            total_batches += 1

        if total_batches == 0 or total_examples == 0:
            raise ValueError("Training DataLoader produced no batches.")

        return TrainEpochMetrics(
            epoch=int(epoch),
            train_loss=float(total_loss / total_examples),
            batches=int(total_batches),
            examples=int(total_examples),
        )

    def _resolve_feature_columns(
        self,
        feature_columns: Sequence[str] | None,
    ) -> list[str]:
        if feature_columns is None:
            return []

        resolved_feature_columns = [str(column) for column in feature_columns]

        if not resolved_feature_columns:
            raise ValueError("feature_columns cannot be empty when provided.")

        if len(resolved_feature_columns) != self.n_features:
            raise ValueError(
                "feature_columns length must match n_features. "
                f"feature_columns: {len(resolved_feature_columns)}, "
                f"n_features: {self.n_features}."
            )

        return resolved_feature_columns

    def _resolve_n_features(self, n_features: int | None) -> int:
        if n_features is None:
            if not hasattr(self.model, "n_features"):
                raise ValueError(
                    "n_features must be provided when the model does not expose "
                    "a n_features attribute."
                )

            n_features = int(getattr(self.model, "n_features"))

        if n_features <= 0:
            raise ValueError(f"n_features must be positive. Current value: {n_features}.")

        return int(n_features)

    @staticmethod
    def _validate_checkpoint_strategy(checkpoint_strategy: str) -> str:
        checkpoint_strategy = str(checkpoint_strategy)

        if checkpoint_strategy not in VALID_CHECKPOINT_STRATEGIES:
            raise ValueError(
                "checkpoint_strategy inválida: "
                f"{checkpoint_strategy}. "
                f"Opções válidas: {sorted(VALID_CHECKPOINT_STRATEGIES)}."
            )

        return checkpoint_strategy

    def _validate_init_args(
        self,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float | None,
        threshold: float,
        early_stopping_patience: int,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive. Current value: {learning_rate}."
            )

        if weight_decay < 0:
            raise ValueError(
                f"weight_decay cannot be negative. Current value: {weight_decay}."
            )

        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive when enabled. "
                f"Current value: {gradient_clip_norm}."
            )

        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"threshold must be between 0 and 1. Current value: {threshold}."
            )

        if early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive. "
                f"Current value: {early_stopping_patience}."
            )

    def _validate_logit_label_shapes(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if logits.shape != y.shape:
            raise ValueError(
                "logits and y must have exactly the same shape before loss. "
                f"logits shape: {tuple(logits.shape)}, y shape: {tuple(y.shape)}."
            )

    def _validate_max_epochs(self, max_epochs: int) -> None:
        if max_epochs <= 0:
            raise ValueError(
                f"max_epochs must be positive. Current value: {max_epochs}."
            )

    def _validate_metric_labels(self, labels: np.ndarray) -> None:
        unique_labels = set(labels.tolist())

        if unique_labels != {0, 1}:
            raise ValueError(
                "Validation metrics require both binary classes in y. "
                f"Current classes: {sorted(unique_labels)}."
            )
