from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from torch.utils.data import DataLoader

from src.itd_tcnTransformer.config.paths import (
    HPO_BEST_PARAMS_PATH,
    HPO_BEST_TRIAL_REPORT_PATH,
    HPO_DIR,
    HPO_STUDY_DB_PATH,
    HPO_TRIALS_CSV_PATH,
    HPO_TRIALS_DIR,
    SCALERS_DIR,
    SEQUENCES_DIR,
    SPLITS_DIR,
)
from src.itd_tcnTransformer.config.settings import validate_device
from src.itd_tcnTransformer.config.settings import (
    BATCH_SIZE,
    CUBLAS_WORKSPACE_CONFIG,
    CUDNN_BENCHMARK,
    DETERMINISTIC_TRAINING,
    DETERMINISTIC_WARN_ONLY,
    DEVICE,
    DROPOUT,
    EVALUATION_UNIT,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOSS_TYPE,
    MODEL_DIM,
    OPTUNA_DIRECTION,
    OPTUNA_BATCH_SIZE_VALUES,
    OPTUNA_DROPOUT_MAX,
    OPTUNA_DROPOUT_MIN,
    OPTUNA_EARLY_STOPPING_PATIENCE,
    OPTUNA_ENQUEUE_CURRENT_SETTINGS_TRIAL,
    OPTUNA_FOCAL_ALPHA_MAX,
    OPTUNA_FOCAL_ALPHA_MIN,
    OPTUNA_FOCAL_GAMMA_MAX,
    OPTUNA_FOCAL_GAMMA_MIN,
    OPTUNA_GRADIENT_CLIP_NORM_VALUES,
    OPTUNA_LEARNING_RATE_LOG,
    OPTUNA_LEARNING_RATE_MAX,
    OPTUNA_LEARNING_RATE_MIN,
    OPTUNA_LOSS_TYPE_VALUES,
    OPTUNA_MAX_EPOCHS_PER_TRIAL,
    OPTUNA_MAX_PRETRAIN_EPOCHS_PER_TRIAL,
    OPTUNA_MIN_PRECISION,
    OPTUNA_MODEL_DIM_VALUES,
    OPTUNA_N_TRIALS,
    OPTUNA_OBJECTIVE_METRIC,
    OPTUNA_PRUNING_ENABLED,
    OPTUNA_SEED,
    OPTUNA_STUDY_NAME,
    OPTUNA_TIMEOUT_SECONDS,
    OPTUNA_TRANSFORMER_FF_DIM_VALUES,
    OPTUNA_TRANSFORMER_NUM_HEADS_VALUES,
    OPTUNA_TRIAL_SEED_OFFSET,
    OPTUNA_USE_FIXED_TRIAL_SEED,
    OPTUNA_USE_POS_WEIGHT_VALUES,
    OPTUNA_WEIGHT_DECAY_LOG,
    OPTUNA_WEIGHT_DECAY_MAX,
    OPTUNA_WEIGHT_DECAY_MIN,
    OVERLAPPING_PREDICTION_AGGREGATION,
    PREDICT_ALL_DAYS,
    POSITIONAL_ENCODING_TYPE,
    PROFILE_FEATURE_MODE,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    TCN_ACTIVATION,
    TCN_DILATIONS,
    TCN_KERNEL_SIZE,
    TCN_NORM_POSITION,
    TCN_USE_LAYER_NORM,
    TRAINING_MODE,
    TRANSFORMER_ACTIVATION,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    USE_CAUSAL_TRANSFORMER_MASK,
    USE_POS_WEIGHT,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.models.tcn_transformer import TCNTransformerModel
from src.itd_tcnTransformer.sequences.sequence_dataset import (
    SequenceDataset,
    create_sequence_dataloader,
)
from src.itd_tcnTransformer.training.losses import create_loss_fn
from src.itd_tcnTransformer.training.trainer import (
    CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
    CHECKPOINT_STRATEGY_LAST_EPOCH,
    Trainer,
    TrainEpochMetrics,
    ValidationMetrics,
)
from src.itd_tcnTransformer.utils.io import save_json
from src.itd_tcnTransformer.utils.reproducibility import (
    build_reproducibility_report,
    set_reproducibility,
)


FEATURE_COLUMNS_FILE = "feature_columns.json"
STANDARD_SCALER_FILE = "standard_scaler.joblib"
TRAIN_Y_FILE = "train_y.pt"
NORMAL_PRETRAIN_Y_FILE = "normal_pretrain_y.pt"
FINETUNE_TRAIN_Y_FILE = "finetune_train_y.pt"
TEMPORAL_VALIDATION_SPLIT_NAME = "validation"
ARTICLE_VALIDATION_SPLIT_NAME = "finetune_validation"
CURRENT_SETTINGS_BASELINE_TRIAL_KIND = "current_settings_baseline"
VALID_OBJECTIVE_METRICS = {
    "validation_auprc",
    "validation_f1",
    "recall_at_min_precision",
}

__all__ = [
    "build_optuna_epoch_callback",
    "build_trial_config",
    "compute_objective_score",
    "compute_recall_at_min_precision",
    "create_hpo_output_dir",
    "get_trial_seed",
    "run_optuna_search",
]


class TrialConfig(TypedDict):
    batch_size: int
    dropout: float
    focal_alpha: NotRequired[float]
    focal_gamma: NotRequired[float]
    gradient_clip_norm: float
    hidden_dim: int
    learning_rate: float
    loss_type: str
    model_dim: int
    transformer_ff_dim: int
    transformer_num_heads: int
    use_pos_weight: bool
    weight_decay: float


OptunaTrialParams = dict[str, bool | float | int | str]


@dataclass
class HpoBestTrialReport:
    best_trial_number: int
    best_value: float
    best_params: dict[str, object]
    objective_metric: str
    min_precision: float
    n_trials_total: int
    n_trials_complete: int
    n_trials_pruned: int
    n_trials_failed: int
    optuna_seed: int
    deterministic_training: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    timestamp: str


def build_optuna_epoch_callback(
    trial: optuna.Trial,
    objective_metric: str,
    min_precision: float,
    pruning_enabled: bool = OPTUNA_PRUNING_ENABLED,
):
    def callback(
        epoch: int,
        epoch_metrics: TrainEpochMetrics,
        trainer: Trainer,
    ) -> None:
        if epoch_metrics.validation is None:
            raise ValueError("Optuna callback requires validation metrics.")

        score = compute_objective_score(
            objective_metric=objective_metric,
            validation_metrics=epoch_metrics.validation,
            validation_labels=trainer.validation_labels,
            validation_probabilities=trainer.validation_probabilities,
            min_precision=min_precision,
        )
        _update_best_trial_score_attrs(trial, epoch, score)
        trial.report(score, step=int(epoch))

        if pruning_enabled and trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with score {score:.6f}."
            )

    return callback


def build_trial_config(trial: optuna.Trial) -> TrialConfig:
    model_dim = cast(
        int,
        trial.suggest_categorical("model_dim", OPTUNA_MODEL_DIM_VALUES),
    )
    transformer_num_heads = cast(
        int,
        trial.suggest_categorical(
            "transformer_num_heads",
            OPTUNA_TRANSFORMER_NUM_HEADS_VALUES,
        ),
    )
    transformer_ff_dim = cast(
        int,
        trial.suggest_categorical(
            "transformer_ff_dim",
            OPTUNA_TRANSFORMER_FF_DIM_VALUES,
        ),
    )

    if model_dim % transformer_num_heads != 0:
        raise optuna.TrialPruned(
            "model_dim must be divisible by transformer_num_heads."
        )

    if transformer_ff_dim <= 0:
        raise optuna.TrialPruned("transformer_ff_dim must be positive.")

    loss_type = cast(
        str,
        trial.suggest_categorical("loss_type", OPTUNA_LOSS_TYPE_VALUES),
    )
    trial_config: TrialConfig = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            OPTUNA_LEARNING_RATE_MIN,
            OPTUNA_LEARNING_RATE_MAX,
            log=OPTUNA_LEARNING_RATE_LOG,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            OPTUNA_WEIGHT_DECAY_MIN,
            OPTUNA_WEIGHT_DECAY_MAX,
            log=OPTUNA_WEIGHT_DECAY_LOG,
        ),
        "dropout": trial.suggest_float(
            "dropout",
            OPTUNA_DROPOUT_MIN,
            OPTUNA_DROPOUT_MAX,
        ),
        "model_dim": int(model_dim),
        "hidden_dim": int(model_dim),
        "transformer_ff_dim": int(transformer_ff_dim),
        "transformer_num_heads": int(transformer_num_heads),
        "batch_size": cast(
            int,
            trial.suggest_categorical(
                "batch_size",
                OPTUNA_BATCH_SIZE_VALUES,
            ),
        ),
        "gradient_clip_norm": float(
            cast(
                float | int,
                trial.suggest_categorical(
                    "gradient_clip_norm",
                    OPTUNA_GRADIENT_CLIP_NORM_VALUES,
                ),
            )
        ),
        "loss_type": str(loss_type),
        "use_pos_weight": False,
    }

    if loss_type == "focal":
        trial_config["focal_alpha"] = trial.suggest_float(
            "focal_alpha",
            OPTUNA_FOCAL_ALPHA_MIN,
            OPTUNA_FOCAL_ALPHA_MAX,
        )
        trial_config["focal_gamma"] = trial.suggest_float(
            "focal_gamma",
            OPTUNA_FOCAL_GAMMA_MIN,
            OPTUNA_FOCAL_GAMMA_MAX,
        )
        trial_config["use_pos_weight"] = False
    else:
        trial_config["use_pos_weight"] = cast(
            bool,
            trial.suggest_categorical(
                "use_pos_weight",
                OPTUNA_USE_POS_WEIGHT_VALUES,
            ),
        )

    return trial_config


def compute_objective_score(
    objective_metric: str,
    validation_metrics: ValidationMetrics,
    validation_labels: torch.Tensor,
    validation_probabilities: torch.Tensor,
    min_precision: float,
) -> float:
    if objective_metric == "validation_auprc":
        return float(validation_metrics.validation_auprc)

    if objective_metric == "validation_f1":
        return float(validation_metrics.validation_f1)

    if objective_metric == "recall_at_min_precision":
        return compute_recall_at_min_precision(
            validation_labels,
            validation_probabilities,
            min_precision,
        )

    raise ValueError(f"Unsupported objective_metric: {objective_metric}")


def compute_recall_at_min_precision(
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    min_precision: float,
) -> float:
    y_true_values = _to_numpy(y_true).astype(np.int64)
    y_prob_values = _to_numpy(y_prob).astype(np.float64)
    best_recall = 0.0

    for threshold in np.linspace(0.0, 1.0, 1001):
        y_pred = (y_prob_values >= threshold).astype(np.int64)
        true_positive = int(((y_true_values == 1) & (y_pred == 1)).sum())
        false_positive = int(((y_true_values == 0) & (y_pred == 1)).sum())
        false_negative = int(((y_true_values == 1) & (y_pred == 0)).sum())

        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)

        if precision >= min_precision:
            best_recall = max(best_recall, recall)

    return float(best_recall)


def create_hpo_output_dir() -> None:
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    HPO_TRIALS_DIR.mkdir(parents=True, exist_ok=True)


def get_trial_seed(
    trial: optuna.Trial,
    use_fixed_trial_seed: bool = OPTUNA_USE_FIXED_TRIAL_SEED,
) -> int:
    if use_fixed_trial_seed:
        return int(RANDOM_SEED)

    return int(RANDOM_SEED + OPTUNA_TRIAL_SEED_OFFSET + trial.number)


def run_optuna_search(
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int | None = OPTUNA_TIMEOUT_SECONDS,
    study_name: str = OPTUNA_STUDY_NAME,
    objective_metric: str = OPTUNA_OBJECTIVE_METRIC,
    min_precision: float = OPTUNA_MIN_PRECISION,
    max_epochs_per_trial: int = OPTUNA_MAX_EPOCHS_PER_TRIAL,
    early_stopping_patience: int = OPTUNA_EARLY_STOPPING_PATIENCE,
    pruning_enabled: bool = OPTUNA_PRUNING_ENABLED,
    seed: int = OPTUNA_SEED,
    use_fixed_trial_seed: bool = OPTUNA_USE_FIXED_TRIAL_SEED,
) -> optuna.Study:
    validate_device(DEVICE)
    _validate_run_args(
        n_trials=n_trials,
        objective_metric=objective_metric,
        min_precision=min_precision,
        max_epochs_per_trial=max_epochs_per_trial,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    create_hpo_output_dir()

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    pruner = (
        optuna.pruners.MedianPruner()
        if pruning_enabled
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        storage=f"sqlite:///{HPO_STUDY_DB_PATH.as_posix()}",
        study_name=str(study_name),
        direction=OPTUNA_DIRECTION,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    _enqueue_current_settings_trial(study)

    study.optimize(
        lambda trial: objective(
            trial=trial,
            objective_metric=objective_metric,
            min_precision=min_precision,
            max_epochs_per_trial=max_epochs_per_trial,
            early_stopping_patience=early_stopping_patience,
            pruning_enabled=pruning_enabled,
            use_fixed_trial_seed=use_fixed_trial_seed,
        ),
        n_trials=int(n_trials),
        timeout=timeout,
    )
    save_best_params(study)
    save_best_trial_report(
        study=study,
        objective_metric=objective_metric,
        min_precision=min_precision,
        optuna_seed=seed,
    )
    save_trials_csv(study)

    return study


def objective(
    trial: optuna.Trial,
    objective_metric: str,
    min_precision: float,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    pruning_enabled: bool,
    use_fixed_trial_seed: bool,
) -> float:
    trial_seed = get_trial_seed(trial, use_fixed_trial_seed)
    trial.set_user_attr("seed", int(trial_seed))
    set_reproducibility(
        seed=trial_seed,
        deterministic=DETERMINISTIC_TRAINING,
        warn_only=DETERMINISTIC_WARN_ONLY,
        cudnn_benchmark=CUDNN_BENCHMARK,
        cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
    )
    trial.set_user_attr(
        "reproducibility",
        build_reproducibility_report(
            seed=trial_seed,
            deterministic=DETERMINISTIC_TRAINING,
            warn_only=DETERMINISTIC_WARN_ONLY,
            cudnn_benchmark=CUDNN_BENCHMARK,
            cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
        ),
    )

    trial_config = build_trial_config(trial)
    dataloaders = _create_hpo_dataloaders(
        batch_size=trial_config["batch_size"],
        seed=trial_seed,
    )

    if TRAINING_MODE == "two_stage":
        score = _run_two_stage_trial(
            trial=trial,
            trial_config=trial_config,
            dataloaders=dataloaders,
            objective_metric=objective_metric,
            min_precision=min_precision,
            max_epochs_per_trial=max_epochs_per_trial,
            early_stopping_patience=early_stopping_patience,
            pruning_enabled=pruning_enabled,
        )
    else:
        score = _run_single_stage_trial(
            trial=trial,
            trial_config=trial_config,
            dataloaders=dataloaders,
            objective_metric=objective_metric,
            min_precision=min_precision,
            max_epochs_per_trial=max_epochs_per_trial,
            early_stopping_patience=early_stopping_patience,
            pruning_enabled=pruning_enabled,
        )

    trial.set_user_attr("objective_metric", str(objective_metric))
    trial.set_user_attr("min_precision", float(min_precision))
    trial.set_user_attr("objective_score", float(score))

    return float(score)


def save_best_params(study: optuna.Study) -> None:
    save_json(dict(study.best_params), HPO_BEST_PARAMS_PATH)


def save_best_trial_report(
    study: optuna.Study,
    objective_metric: str,
    min_precision: float,
    optuna_seed: int,
) -> None:
    trials = study.trials
    report = HpoBestTrialReport(
        best_trial_number=int(study.best_trial.number),
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        objective_metric=str(objective_metric),
        min_precision=float(min_precision),
        n_trials_total=len(trials),
        n_trials_complete=_count_trials_by_state(trials, TrialState.COMPLETE),
        n_trials_pruned=_count_trials_by_state(trials, TrialState.PRUNED),
        n_trials_failed=_count_trials_by_state(trials, TrialState.FAIL),
        optuna_seed=int(optuna_seed),
        deterministic_training=bool(DETERMINISTIC_TRAINING),
        deterministic_warn_only=bool(DETERMINISTIC_WARN_ONLY),
        cudnn_benchmark=bool(CUDNN_BENCHMARK),
        cublas_workspace_config=str(CUBLAS_WORKSPACE_CONFIG),
        timestamp=datetime.now(UTC).isoformat(),
    )
    save_json(report, HPO_BEST_TRIAL_REPORT_PATH)


def save_trials_csv(study: optuna.Study) -> None:
    HPO_TRIALS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(HPO_TRIALS_CSV_PATH, index=False)


def _build_current_settings_trial_params() -> OptunaTrialParams:
    return {
        "batch_size": int(BATCH_SIZE),
        "dropout": float(DROPOUT),
        "focal_alpha": float(FOCAL_ALPHA),
        "focal_gamma": float(FOCAL_GAMMA),
        "gradient_clip_norm": float(GRADIENT_CLIP_NORM),
        "learning_rate": float(LEARNING_RATE),
        "loss_type": str(LOSS_TYPE),
        "model_dim": int(MODEL_DIM),
        "transformer_ff_dim": int(TRANSFORMER_FF_DIM),
        "transformer_num_heads": int(TRANSFORMER_NUM_HEADS),
        "use_pos_weight": bool(USE_POS_WEIGHT),
        "weight_decay": float(WEIGHT_DECAY),
    }


def _build_model(
    n_features: int,
    trial_config: TrialConfig,
) -> TCNTransformerModel:
    return TCNTransformerModel(
        n_features=n_features,
        window_size=WINDOW_SIZE,
        hidden_dim=trial_config["model_dim"],
        tcn_dilations=TCN_DILATIONS,
        kernel_size=TCN_KERNEL_SIZE,
        tcn_activation=TCN_ACTIVATION,
        tcn_use_layer_norm=TCN_USE_LAYER_NORM,
        tcn_norm_position=TCN_NORM_POSITION,
        num_layers=TRANSFORMER_NUM_LAYERS,
        num_heads=trial_config["transformer_num_heads"],
        feedforward_dim=trial_config["transformer_ff_dim"],
        transformer_activation=TRANSFORMER_ACTIVATION,
        positional_encoding_type=POSITIONAL_ENCODING_TYPE,
        dropout=trial_config["dropout"],
        predict_all_days=PREDICT_ALL_DAYS,
        use_causal_mask=USE_CAUSAL_TRANSFORMER_MASK,
    )


def _build_trainer(
    model: TCNTransformerModel,
    loss_fn,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    n_features: int,
    feature_columns: list[str],
    checkpoint_path: Path,
    training_phase: str,
    validation_split_name: str,
    trial_config: TrialConfig,
    early_stopping_patience: int,
    checkpoint_strategy: str,
    epoch_end_callback=None,
    loss_hyperparameters: dict[str, object] | None = None,
) -> Trainer:
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=DEVICE,
        n_features=n_features,
        feature_columns=feature_columns,
        scaler_path=_get_standard_scaler_path(),
        checkpoint_path=checkpoint_path,
        learning_rate=trial_config["learning_rate"],
        weight_decay=trial_config["weight_decay"],
        gradient_clip_norm=trial_config["gradient_clip_norm"],
        early_stopping_patience=int(early_stopping_patience),
        training_phase=training_phase,
        validation_split_name=validation_split_name,
        checkpoint_strategy=checkpoint_strategy,
        epoch_end_callback=epoch_end_callback,
        loss_hyperparameters=(
            _get_trial_loss_hyperparameters(trial_config)
            if loss_hyperparameters is None
            else loss_hyperparameters
        ),
    )


def _build_trial_loss_fn(
    train_y_file: str,
    trial_config: TrialConfig,
    force_disable_pos_weight: bool = False,
):
    return create_loss_fn(
        train_y_path=SEQUENCES_DIR / train_y_file,
        loss_type=trial_config["loss_type"],
        use_pos_weight=(
            False
            if force_disable_pos_weight
            else bool(trial_config.get("use_pos_weight", False))
        ),
        focal_alpha=trial_config.get("focal_alpha", FOCAL_ALPHA),
        focal_gamma=trial_config.get("focal_gamma", FOCAL_GAMMA),
        device=DEVICE,
        predict_all_days=PREDICT_ALL_DAYS,
        window_size=WINDOW_SIZE,
    )


def _create_hpo_dataloaders(
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    if TRAINING_MODE == "two_stage":
        split_names = (
            "normal_pretrain",
            "finetune_train",
            "finetune_validation",
        )
    else:
        split_names = ("train", "validation")

    return {
        split_name: create_sequence_dataloader(
            split_name=split_name,
            batch_size=int(batch_size),
            seed=int(seed),
            use_seeded_generator=True,
        )
        for split_name in split_names
    }


def _count_trials_by_state(
    trials: list[optuna.trial.FrozenTrial],
    state: TrialState,
) -> int:
    return sum(1 for trial in trials if trial.state == state)


def _current_settings_baseline_exists(study: optuna.Study) -> bool:
    return any(
        trial.user_attrs.get("trial_kind") == CURRENT_SETTINGS_BASELINE_TRIAL_KIND
        for trial in study.trials
    )


def _enqueue_current_settings_trial(study: optuna.Study) -> None:
    if not OPTUNA_ENQUEUE_CURRENT_SETTINGS_TRIAL:
        return

    if _current_settings_baseline_exists(study):
        print("Current settings baseline trial already exists; skipping enqueue.")
        return

    params = _build_current_settings_trial_params()
    _validate_current_settings_trial_params(params)
    study.enqueue_trial(
        params,
        user_attrs={"trial_kind": CURRENT_SETTINGS_BASELINE_TRIAL_KIND},
    )
    print("Enqueued current settings baseline trial.")


def _get_n_features_from_dataloader(dataloader: DataLoader) -> int:
    dataset = dataloader.dataset

    if not isinstance(dataset, SequenceDataset):
        raise TypeError(
            "Expected DataLoader dataset to be SequenceDataset. "
            f"Current type: {type(dataset).__name__}."
        )

    if dataset.X.ndim != 3:
        raise ValueError(
            "Sequence X tensor must have shape "
            "(n_windows, WINDOW_SIZE, n_features)."
        )

    return int(dataset.X.shape[-1])


def _get_standard_scaler_path() -> Path:
    scaler_path = SCALERS_DIR / STANDARD_SCALER_FILE

    if not scaler_path.exists() or not scaler_path.is_file():
        raise FileNotFoundError(f"Standard scaler file not found: {scaler_path}")

    return scaler_path


def _get_trial_dir(trial: optuna.Trial) -> Path:
    trial_dir = HPO_TRIALS_DIR / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    return trial_dir


def _get_trial_loss_hyperparameters(
    trial_config: TrialConfig,
    force_disable_pos_weight: bool = False,
) -> dict[str, object]:
    return {
        "loss_type": trial_config["loss_type"],
        "focal_alpha": trial_config.get("focal_alpha", FOCAL_ALPHA),
        "focal_gamma": trial_config.get("focal_gamma", FOCAL_GAMMA),
        "use_pos_weight": (
            False
            if force_disable_pos_weight
            else bool(trial_config.get("use_pos_weight", False))
        ),
    }


def _load_feature_columns() -> list[str]:
    feature_columns_path = SPLITS_DIR / FEATURE_COLUMNS_FILE

    if not feature_columns_path.exists() or not feature_columns_path.is_file():
        raise FileNotFoundError(
            f"Feature columns report not found: {feature_columns_path}"
        )

    import json

    with feature_columns_path.open("r", encoding="utf-8") as file:
        feature_columns_report = json.load(file)

    feature_columns = feature_columns_report.get("feature_columns")

    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str)
        for column in feature_columns
    ):
        raise ValueError(
            "feature_columns.json must contain a feature_columns list of strings."
        )

    return feature_columns


def _run_single_stage_trial(
    trial: optuna.Trial,
    trial_config: TrialConfig,
    dataloaders: dict[str, DataLoader],
    objective_metric: str,
    min_precision: float,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    pruning_enabled: bool,
) -> float:
    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]
    n_features = _get_n_features_from_dataloader(train_dataloader)
    model = _build_model(n_features, trial_config)
    loss_fn = _build_trial_loss_fn(TRAIN_Y_FILE, trial_config)
    trial_dir = _get_trial_dir(trial)
    callback = build_optuna_epoch_callback(
        trial,
        objective_metric,
        min_precision,
        pruning_enabled,
    )
    trainer = _build_trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        n_features=n_features,
        feature_columns=_load_feature_columns(),
        checkpoint_path=trial_dir / "checkpoint.pt",
        training_phase="hpo_single_stage",
        validation_split_name=TEMPORAL_VALIDATION_SPLIT_NAME,
        trial_config=trial_config,
        early_stopping_patience=early_stopping_patience,
        checkpoint_strategy=CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
        epoch_end_callback=callback,
    )
    history = trainer.train(max_epochs=max_epochs_per_trial)
    validation_metrics = trainer.validate()
    score = compute_objective_score(
        objective_metric=objective_metric,
        validation_metrics=validation_metrics,
        validation_labels=trainer.validation_labels,
        validation_probabilities=trainer.validation_probabilities,
        min_precision=min_precision,
    )
    _set_final_trial_attrs(trial, history.best_epoch, validation_metrics, score)

    return _get_best_trial_score(trial, score)


def _run_two_stage_trial(
    trial: optuna.Trial,
    trial_config: TrialConfig,
    dataloaders: dict[str, DataLoader],
    objective_metric: str,
    min_precision: float,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    pruning_enabled: bool,
) -> float:
    pretrain_dataloader = dataloaders["normal_pretrain"]
    finetune_train_dataloader = dataloaders["finetune_train"]
    finetune_validation_dataloader = dataloaders["finetune_validation"]
    n_features = _get_n_features_from_dataloader(pretrain_dataloader)
    feature_columns = _load_feature_columns()
    trial_dir = _get_trial_dir(trial)
    model = _build_model(n_features, trial_config)

    pretrain_trainer = _build_trainer(
        model=model,
        loss_fn=_build_trial_loss_fn(
            NORMAL_PRETRAIN_Y_FILE,
            trial_config,
            force_disable_pos_weight=True,
        ),
        train_dataloader=pretrain_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        n_features=n_features,
        feature_columns=feature_columns,
        checkpoint_path=trial_dir / "pretrain_checkpoint.pt",
        training_phase="hpo_pretrain",
        validation_split_name=ARTICLE_VALIDATION_SPLIT_NAME,
        trial_config=trial_config,
        early_stopping_patience=early_stopping_patience,
        checkpoint_strategy=CHECKPOINT_STRATEGY_LAST_EPOCH,
        loss_hyperparameters=_get_trial_loss_hyperparameters(
            trial_config,
            force_disable_pos_weight=True,
        ),
    )
    pretrain_trainer.train(max_epochs=OPTUNA_MAX_PRETRAIN_EPOCHS_PER_TRIAL)

    callback = build_optuna_epoch_callback(
        trial,
        objective_metric,
        min_precision,
        pruning_enabled,
    )
    finetune_trainer = _build_trainer(
        model=model,
        loss_fn=_build_trial_loss_fn(FINETUNE_TRAIN_Y_FILE, trial_config),
        train_dataloader=finetune_train_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        n_features=n_features,
        feature_columns=feature_columns,
        checkpoint_path=trial_dir / "finetune_checkpoint.pt",
        training_phase="hpo_finetune",
        validation_split_name=ARTICLE_VALIDATION_SPLIT_NAME,
        trial_config=trial_config,
        early_stopping_patience=early_stopping_patience,
        checkpoint_strategy=CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
        epoch_end_callback=callback,
    )
    history = finetune_trainer.train(max_epochs=max_epochs_per_trial)
    validation_metrics = finetune_trainer.validate()
    score = compute_objective_score(
        objective_metric=objective_metric,
        validation_metrics=validation_metrics,
        validation_labels=finetune_trainer.validation_labels,
        validation_probabilities=finetune_trainer.validation_probabilities,
        min_precision=min_precision,
    )
    _set_final_trial_attrs(trial, history.best_epoch, validation_metrics, score)

    return _get_best_trial_score(trial, score)


def _set_final_trial_attrs(
    trial: optuna.Trial,
    best_epoch: int | None,
    validation_metrics: ValidationMetrics,
    score: float,
) -> None:
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr(
        "validation_auprc",
        float(validation_metrics.validation_auprc),
    )
    trial.set_user_attr(
        "validation_precision",
        float(validation_metrics.validation_precision),
    )
    trial.set_user_attr(
        "validation_recall",
        float(validation_metrics.validation_recall),
    )
    trial.set_user_attr("validation_f1", float(validation_metrics.validation_f1))
    trial.set_user_attr("final_objective_score", float(score))


def _get_best_trial_score(trial: optuna.Trial, fallback_score: float) -> float:
    return float(trial.user_attrs.get("best_objective_score", fallback_score))


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()

    return np.asarray(values)


def _update_best_trial_score_attrs(
    trial: optuna.Trial,
    epoch: int,
    score: float,
) -> None:
    current_best = trial.user_attrs.get("best_objective_score")

    if current_best is None or float(score) > float(current_best):
        trial.set_user_attr("best_objective_score", float(score))
        trial.set_user_attr("best_objective_epoch", int(epoch))


def _validate_current_settings_trial_params(params: OptunaTrialParams) -> None:
    if params["model_dim"] not in OPTUNA_MODEL_DIM_VALUES:
        raise ValueError(
            "MODEL_DIM must be included in OPTUNA_MODEL_DIM_VALUES before "
            "enqueuing the current settings baseline trial."
        )

    if params["transformer_num_heads"] not in OPTUNA_TRANSFORMER_NUM_HEADS_VALUES:
        raise ValueError(
            "TRANSFORMER_NUM_HEADS must be included in "
            "OPTUNA_TRANSFORMER_NUM_HEADS_VALUES before enqueuing the current "
            "settings baseline trial."
        )

    if params["transformer_ff_dim"] not in OPTUNA_TRANSFORMER_FF_DIM_VALUES:
        raise ValueError(
            "TRANSFORMER_FF_DIM must be included in "
            "OPTUNA_TRANSFORMER_FF_DIM_VALUES before enqueuing the current "
            "settings baseline trial."
        )

    if params["batch_size"] not in OPTUNA_BATCH_SIZE_VALUES:
        raise ValueError(
            "BATCH_SIZE must be included in OPTUNA_BATCH_SIZE_VALUES before "
            "enqueuing the current settings baseline trial."
        )

    if params["gradient_clip_norm"] not in OPTUNA_GRADIENT_CLIP_NORM_VALUES:
        raise ValueError(
            "GRADIENT_CLIP_NORM must be included in "
            "OPTUNA_GRADIENT_CLIP_NORM_VALUES before enqueuing the current "
            "settings baseline trial."
        )

    if params["loss_type"] not in OPTUNA_LOSS_TYPE_VALUES:
        raise ValueError(
            "LOSS_TYPE must be included in OPTUNA_LOSS_TYPE_VALUES before "
            "enqueuing the current settings baseline trial."
        )

    if params["use_pos_weight"] not in OPTUNA_USE_POS_WEIGHT_VALUES:
        raise ValueError(
            "USE_POS_WEIGHT must be included in OPTUNA_USE_POS_WEIGHT_VALUES "
            "before enqueuing the current settings baseline trial."
        )

    if int(params["model_dim"]) % int(params["transformer_num_heads"]) != 0:
        raise ValueError(
            "MODEL_DIM must be divisible by TRANSFORMER_NUM_HEADS before "
            "enqueuing the current settings baseline trial."
        )

    if params["loss_type"] == "focal" and bool(params["use_pos_weight"]):
        raise ValueError(
            "USE_POS_WEIGHT must be False when LOSS_TYPE='focal' before "
            "enqueuing the current settings baseline trial."
        )

    _validate_current_settings_trial_range(
        "LEARNING_RATE",
        float(params["learning_rate"]),
        OPTUNA_LEARNING_RATE_MIN,
        OPTUNA_LEARNING_RATE_MAX,
    )
    _validate_current_settings_trial_range(
        "WEIGHT_DECAY",
        float(params["weight_decay"]),
        OPTUNA_WEIGHT_DECAY_MIN,
        OPTUNA_WEIGHT_DECAY_MAX,
    )
    _validate_current_settings_trial_range(
        "DROPOUT",
        float(params["dropout"]),
        OPTUNA_DROPOUT_MIN,
        OPTUNA_DROPOUT_MAX,
    )

    if params["loss_type"] == "focal":
        _validate_current_settings_trial_range(
            "FOCAL_ALPHA",
            float(params["focal_alpha"]),
            OPTUNA_FOCAL_ALPHA_MIN,
            OPTUNA_FOCAL_ALPHA_MAX,
        )
        _validate_current_settings_trial_range(
            "FOCAL_GAMMA",
            float(params["focal_gamma"]),
            OPTUNA_FOCAL_GAMMA_MIN,
            OPTUNA_FOCAL_GAMMA_MAX,
        )


def _validate_current_settings_trial_range(
    name: str,
    value: float,
    min_value: float,
    max_value: float,
) -> None:
    if value < min_value or value > max_value:
        raise ValueError(
            f"{name}={value} must be inside the configured Optuna range "
            f"[{min_value}, {max_value}] before enqueuing the current settings "
            "baseline trial."
        )


def _validate_run_args(
    n_trials: int,
    objective_metric: str,
    min_precision: float,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    seed: int,
) -> None:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    if objective_metric not in VALID_OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported objective_metric: {objective_metric}")

    if min_precision < 0 or min_precision > 1:
        raise ValueError("min_precision must be between 0 and 1.")

    if max_epochs_per_trial <= 0:
        raise ValueError("max_epochs_per_trial must be positive.")

    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    if seed < 0:
        raise ValueError("seed must be greater than or equal to 0.")
