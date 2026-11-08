import csv

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from src.itd_tcnTransformer.utils.io import save_json


__all__ = [
    "TrainingArtifactLog",
    "TrainingEpochLogRow",
    "TrainingEvaluationConfigurationLog",
    "TrainingLossConfigurationLog",
    "TrainingModelConfigurationLog",
    "TrainingOptimizationLog",
    "TrainingReproducibilityLog",
    "TrainingRunLog",
    "TrainingShapeLog",
    "save_training_epoch_logs",
    "save_training_run_log",
]


@dataclass
class TrainingArtifactLog:
    checkpoint_path: str
    scaler_path: str | None
    feature_columns_count: int


@dataclass
class TrainingEpochLogRow:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_auprc: float
    validation_auroc: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    evaluation_unit: str
    overlapping_prediction_aggregation: str
    validation_prediction_count_before_aggregation: int
    validation_prediction_count_after_aggregation: int
    threshold: float
    train_batches: int
    train_examples: int
    validation_batches: int
    validation_examples: int
    is_best_epoch: bool
    best_epoch: int | None
    best_validation_auprc: float | None
    early_stopped: bool
    stopped_epoch: int | None


@dataclass
class TrainingLossConfigurationLog:
    loss_type: str
    focal_alpha: float
    focal_gamma: float
    use_pos_weight: bool


@dataclass
class TrainingModelConfigurationLog:
    model_class: str
    n_features: int
    window_size: int
    model_dim: int
    hidden_dim: int
    tcn_dilations: list[int]
    tcn_kernel_size: int
    tcn_activation: str
    tcn_use_layer_norm: bool
    tcn_norm_position: str
    transformer_num_layers: int
    transformer_num_heads: int
    transformer_ff_dim: int
    transformer_activation: str
    positional_encoding_type: str
    profile_feature_mode: str
    dropout: float
    predict_all_days: bool
    use_causal_transformer_mask: bool


@dataclass
class TrainingOptimizationLog:
    max_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    threshold: float
    early_stopping_patience: int
    monitor_metric: str
    device: str


@dataclass
class TrainingReproducibilityLog:
    random_seed: int
    deterministic_training: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    torch_initial_seed: int
    cuda_available: bool
    cuda_device_count: int
    torch_version: str


@dataclass
class TrainingEvaluationConfigurationLog:
    evaluation_unit: str
    overlapping_prediction_aggregation: str
    validation_split_name: str
    threshold_selection_metric: str


@dataclass
class TrainingRunLog:
    training_mode: str
    split_strategy: str
    training_phase: str
    model: TrainingModelConfigurationLog
    loss: TrainingLossConfigurationLog
    evaluation: TrainingEvaluationConfigurationLog
    optimization: TrainingOptimizationLog
    reproducibility: TrainingReproducibilityLog
    artifacts: TrainingArtifactLog
    shapes: "TrainingShapeLog"
    params_override: dict[str, object] = field(default_factory=dict)
    environment: dict[str, object] | None = None


@dataclass
class TrainingShapeLog:
    train_X_batch_shape: list[int]
    train_y_batch_shape: list[int]
    validation_X_batch_shape: list[int]
    validation_y_batch_shape: list[int]
    model_input_shape: list[int]
    expected_model_output_shape: list[int]


def save_training_epoch_logs(
    epoch_logs: Sequence[TrainingEpochLogRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(TrainingEpochLogRow)]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch_log in epoch_logs:
            writer.writerow(asdict(epoch_log))


def save_training_run_log(
    training_run_log: TrainingRunLog,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(training_run_log, output_path)
