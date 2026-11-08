import json
import os

from pathlib import Path

from torch.utils.data import DataLoader

from src.itd_tcnTransformer.config.paths import (
    LOGS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    SCALERS_DIR,
    SEQUENCES_DIR,
    SPLITS_DIR,
)
from src.itd_tcnTransformer.config.settings import validate_device
from src.itd_tcnTransformer.config.settings import (
    BATCH_SIZE,
    CUBLAS_WORKSPACE_CONFIG,
    CUDNN_BENCHMARK,
    DEVICE,
    DETERMINISTIC_TRAINING,
    DETERMINISTIC_WARN_ONLY,
    DROPOUT,
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
    PRETRAIN_CHECKPOINT_STRATEGY,
    PRETRAIN_EPOCHS,
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
    THRESHOLD_SELECTION_METRIC,
    USE_CAUSAL_TRANSFORMER_MASK,
    USE_POS_WEIGHT,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from src.itd_tcnTransformer.models.tcn_transformer import TCNTransformerModel
from src.itd_tcnTransformer.sequences.sequence_dataset import (
    SequenceDataset,
    create_sequence_dataloaders,
)
from src.itd_tcnTransformer.training.losses import create_loss_fn
from src.itd_tcnTransformer.training.trainer import (
    BEST_CHECKPOINT_FILE,
    CHECKPOINT_MONITOR_METRIC,
    CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
    FINETUNE_BEST_CHECKPOINT_FILE,
    PRETRAIN_CHECKPOINT_FILE,
    Trainer,
    TrainingHistory,
)
from src.itd_tcnTransformer.utils.io import save_json
from src.itd_tcnTransformer.utils.environment import collect_training_environment
from src.itd_tcnTransformer.utils.logging_utils import (
    TrainingArtifactLog,
    TrainingEpochLogRow,
    TrainingEvaluationConfigurationLog,
    TrainingLossConfigurationLog,
    TrainingModelConfigurationLog,
    TrainingOptimizationLog,
    TrainingReproducibilityLog,
    TrainingRunLog,
    TrainingShapeLog,
    save_training_epoch_logs,
    save_training_run_log,
)
from src.itd_tcnTransformer.utils.reproducibility import (
    build_reproducibility_report,
    set_reproducibility,
)


FEATURE_COLUMNS_FILE = "feature_columns.json"
STANDARD_SCALER_FILE = "standard_scaler.joblib"
TRAINING_EPOCH_LOGS_FILE = "training_epoch_logs.csv"
TRAINING_RUN_LOG_FILE = "training_run_log.json"
TRAIN_METRICS_FILE = "training_metrics.json"
PRETRAIN_EPOCH_LOGS_FILE = "pretrain_training_epoch_logs.csv"
PRETRAIN_RUN_LOG_FILE = "pretrain_training_run_log.json"
PRETRAIN_METRICS_FILE = "pretrain_training_metrics.json"
FINETUNE_EPOCH_LOGS_FILE = "finetune_training_epoch_logs.csv"
FINETUNE_RUN_LOG_FILE = "finetune_training_run_log.json"
FINETUNE_METRICS_FILE = "finetune_training_metrics.json"
TEMPORAL_VALIDATION_SPLIT_NAME = "validation"
ARTICLE_VALIDATION_SPLIT_NAME = "finetune_validation"

TRAIN_Y_FILE = "train_y.pt"
NORMAL_PRETRAIN_Y_FILE = "normal_pretrain_y.pt"
FINETUNE_TRAIN_Y_FILE = "finetune_train_y.pt"
ALLOWED_PARAMS_OVERRIDE_KEYS = {
    "batch_size",
    "model_dim",
    "hidden_dim",
    "transformer_ff_dim",
    "transformer_num_heads",
    "dropout",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "loss_type",
    "focal_alpha",
    "focal_gamma",
    "use_pos_weight",
}

__all__ = ["load_hpo_params_json", "main"]


def main(
    params_override: dict[str, object] | None = None,
) -> TrainingHistory | dict[str, TrainingHistory]:
    validate_device(DEVICE)
    params_override = _validate_params_override(params_override)
    pythonhashseed_at_entry = os.environ.get("PYTHONHASHSEED")
    set_reproducibility(
        seed=RANDOM_SEED,
        deterministic=DETERMINISTIC_TRAINING,
        warn_only=DETERMINISTIC_WARN_ONLY,
        cudnn_benchmark=CUDNN_BENCHMARK,
        cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
    )
    reproducibility_report = build_reproducibility_report(
        seed=RANDOM_SEED,
        deterministic=DETERMINISTIC_TRAINING,
        warn_only=DETERMINISTIC_WARN_ONLY,
        cudnn_benchmark=CUDNN_BENCHMARK,
        cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
    )
    reproducibility_report["pythonhashseed_at_entry"] = pythonhashseed_at_entry

    print("Creating sequence dataloaders ...")
    dataloaders = create_sequence_dataloaders(
        batch_size=_get_batch_size(params_override),
        seed=RANDOM_SEED,
    )

    if TRAINING_MODE == "two_stage":
        return _run_two_stage_training(
            dataloaders,
            reproducibility_report,
            params_override,
        )

    return _run_single_stage_training(
        dataloaders,
        reproducibility_report,
        params_override,
    )


def load_hpo_params_json(input_path: Path | str) -> dict[str, object]:
    params_path = Path(input_path)

    if not params_path.exists() or not params_path.is_file():
        raise FileNotFoundError(f"HPO params file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as file:
        params = json.load(file)

    if not isinstance(params, dict):
        raise TypeError(
            "HPO params JSON must contain an object with hyperparameter values."
        )

    return _validate_params_override(params)


def _run_single_stage_training(
    dataloaders: dict[str, DataLoader],
    reproducibility_report: dict[str, object],
    params_override: dict[str, object],
) -> TrainingHistory:
    train_dataloader = dataloaders["train"]
    validation_dataloader = dataloaders["validation"]
    validation_split_name = TEMPORAL_VALIDATION_SPLIT_NAME
    n_features = _get_n_features_from_train_dataloader(train_dataloader)
    feature_columns = _load_feature_columns()
    _validate_feature_columns_match_n_features(feature_columns, n_features)
    scaler_path = _get_standard_scaler_path()

    print("Creating TCN Transformer model ...")
    model = _create_model(n_features, params_override)
    loss_fn = _create_loss_fn(TRAIN_Y_FILE, params_override)
    trainer = _create_trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        n_features=n_features,
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        training_phase="single_stage",
        validation_split_name=validation_split_name,
        params_override=params_override,
        loss_hyperparameters=_get_loss_params(params_override),
    )

    _save_training_run_log(
        trainer=trainer,
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        training_phase="single_stage",
        validation_split_name=validation_split_name,
        max_epochs=MAX_EPOCHS,
        reproducibility_report=reproducibility_report,
        params_override=params_override,
    )

    print("Starting training loop ...")
    history = trainer.train()
    _save_training_epoch_logs(history)
    _save_training_metrics(history)

    return history


def _run_two_stage_training(
    dataloaders: dict[str, DataLoader],
    reproducibility_report: dict[str, object],
    params_override: dict[str, object],
) -> dict[str, TrainingHistory]:
    required_splits = {
        "normal_pretrain",
        "finetune_train",
        "finetune_validation",
    }
    missing_splits = sorted(required_splits - set(dataloaders))
    if missing_splits:
        raise ValueError(
            "Two-stage training requires missing dataloaders: "
            f"{missing_splits}"
        )

    pretrain_dataloader = dataloaders["normal_pretrain"]
    finetune_train_dataloader = dataloaders["finetune_train"]
    finetune_validation_dataloader = dataloaders["finetune_validation"]
    validation_split_name = ARTICLE_VALIDATION_SPLIT_NAME
    n_features = _get_n_features_from_train_dataloader(pretrain_dataloader)
    feature_columns = _load_feature_columns()
    _validate_feature_columns_match_n_features(feature_columns, n_features)
    scaler_path = _get_standard_scaler_path()

    print("Creating TCN Transformer model ...")
    model = _create_model(n_features, params_override)
    pretrain_loss_fn = _create_loss_fn(
        NORMAL_PRETRAIN_Y_FILE,
        params_override,
        force_disable_pos_weight=True,
    )

    print("Starting pretrain phase ...")
    pretrain_trainer = _create_trainer(
        model=model,
        loss_fn=pretrain_loss_fn,
        train_dataloader=pretrain_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        n_features=n_features,
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        checkpoint_path=MODELS_DIR / PRETRAIN_CHECKPOINT_FILE,
        training_phase="pretrain",
        validation_split_name=validation_split_name,
        checkpoint_strategy=PRETRAIN_CHECKPOINT_STRATEGY,
        params_override=params_override,
        loss_hyperparameters=_get_loss_params(
            params_override,
            force_disable_pos_weight=True,
        ),
    )
    _save_training_run_log(
        trainer=pretrain_trainer,
        model=model,
        train_dataloader=pretrain_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        training_phase="pretrain",
        validation_split_name=validation_split_name,
        output_file=PRETRAIN_RUN_LOG_FILE,
        max_epochs=PRETRAIN_EPOCHS,
        reproducibility_report=reproducibility_report,
        params_override=params_override,
    )
    pretrain_history = pretrain_trainer.train(max_epochs=PRETRAIN_EPOCHS)
    _save_training_epoch_logs(pretrain_history, PRETRAIN_EPOCH_LOGS_FILE)
    _save_training_metrics(pretrain_history, PRETRAIN_METRICS_FILE)

    print("Starting finetune phase ...")
    finetune_loss_fn = _create_loss_fn(FINETUNE_TRAIN_Y_FILE, params_override)
    finetune_trainer = _create_trainer(
        model=model,
        loss_fn=finetune_loss_fn,
        train_dataloader=finetune_train_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        n_features=n_features,
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        checkpoint_path=MODELS_DIR / FINETUNE_BEST_CHECKPOINT_FILE,
        training_phase="finetune",
        validation_split_name=validation_split_name,
        checkpoint_strategy=CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
        params_override=params_override,
        loss_hyperparameters=_get_loss_params(params_override),
    )
    _save_training_run_log(
        trainer=finetune_trainer,
        model=model,
        train_dataloader=finetune_train_dataloader,
        validation_dataloader=finetune_validation_dataloader,
        training_phase="finetune",
        validation_split_name=validation_split_name,
        output_file=FINETUNE_RUN_LOG_FILE,
        max_epochs=MAX_EPOCHS,
        reproducibility_report=reproducibility_report,
        params_override=params_override,
    )
    finetune_history = finetune_trainer.train()
    _save_training_epoch_logs(finetune_history, FINETUNE_EPOCH_LOGS_FILE)
    _save_training_metrics(finetune_history, FINETUNE_METRICS_FILE)

    return {
        "pretrain": pretrain_history,
        "finetune": finetune_history,
    }


def _build_training_epoch_log_rows(
    history: TrainingHistory,
) -> list[TrainingEpochLogRow]:
    epoch_log_rows: list[TrainingEpochLogRow] = []

    for epoch_metrics in history.epochs:
        if epoch_metrics.validation is None:
            raise ValueError(
                "Cannot write epoch logs without validation metrics. "
                f"Missing validation metrics for epoch {epoch_metrics.epoch}."
            )

        validation = epoch_metrics.validation
        epoch_log_rows.append(
            TrainingEpochLogRow(
                epoch=int(epoch_metrics.epoch),
                train_loss=float(epoch_metrics.train_loss),
                validation_loss=float(validation.validation_loss),
                validation_auprc=float(validation.validation_auprc),
                validation_auroc=float(validation.validation_auroc),
                validation_precision=float(validation.validation_precision),
                validation_recall=float(validation.validation_recall),
                validation_f1=float(validation.validation_f1),
                evaluation_unit=str(validation.evaluation_unit),
                overlapping_prediction_aggregation=str(
                    validation.overlapping_prediction_aggregation
                ),
                validation_prediction_count_before_aggregation=int(
                    validation.prediction_count_before_aggregation
                ),
                validation_prediction_count_after_aggregation=int(
                    validation.prediction_count_after_aggregation
                ),
                threshold=float(validation.threshold),
                train_batches=int(epoch_metrics.batches),
                train_examples=int(epoch_metrics.examples),
                validation_batches=int(validation.batches),
                validation_examples=int(validation.examples),
                is_best_epoch=epoch_metrics.epoch == history.best_epoch,
                best_epoch=history.best_epoch,
                best_validation_auprc=history.best_validation_auprc,
                early_stopped=bool(history.early_stopped),
                stopped_epoch=history.stopped_epoch,
            )
        )

    return epoch_log_rows


def _create_model(
    n_features: int,
    params_override: dict[str, object],
) -> TCNTransformerModel:
    model_dim = _get_model_dim(params_override)

    return TCNTransformerModel(
        n_features=n_features,
        window_size=WINDOW_SIZE,
        hidden_dim=model_dim,
        tcn_dilations=TCN_DILATIONS,
        kernel_size=TCN_KERNEL_SIZE,
        tcn_activation=TCN_ACTIVATION,
        tcn_use_layer_norm=TCN_USE_LAYER_NORM,
        tcn_norm_position=TCN_NORM_POSITION,
        num_layers=TRANSFORMER_NUM_LAYERS,
        num_heads=_get_transformer_num_heads(params_override),
        feedforward_dim=_get_transformer_ff_dim(params_override),
        transformer_activation=TRANSFORMER_ACTIVATION,
        positional_encoding_type=POSITIONAL_ENCODING_TYPE,
        dropout=_get_dropout(params_override),
        predict_all_days=PREDICT_ALL_DAYS,
        use_causal_mask=USE_CAUSAL_TRANSFORMER_MASK,
    )


def _create_trainer(
    model: TCNTransformerModel,
    loss_fn,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    n_features: int,
    feature_columns: list[str],
    scaler_path,
    training_phase: str,
    validation_split_name: str,
    checkpoint_path=None,
    checkpoint_strategy: str = CHECKPOINT_STRATEGY_BEST_VALIDATION_AUPRC,
    params_override: dict[str, object] | None = None,
    loss_hyperparameters: dict[str, object] | None = None,
) -> Trainer:
    params_override = _validate_params_override(params_override)

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=DEVICE,
        n_features=n_features,
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        checkpoint_path=(
            MODELS_DIR / BEST_CHECKPOINT_FILE
            if checkpoint_path is None
            else checkpoint_path
        ),
        training_phase=training_phase,
        validation_split_name=validation_split_name,
        checkpoint_strategy=checkpoint_strategy,
        learning_rate=_get_learning_rate(params_override),
        weight_decay=_get_weight_decay(params_override),
        gradient_clip_norm=_get_gradient_clip_norm(params_override),
        loss_hyperparameters=loss_hyperparameters,
    )


def _create_loss_fn(
    train_y_file: str,
    params_override: dict[str, object],
    force_disable_pos_weight: bool = False,
):
    loss_params = _get_loss_params(
        params_override,
        force_disable_pos_weight=force_disable_pos_weight,
    )

    return create_loss_fn(
        train_y_path=SEQUENCES_DIR / train_y_file,
        loss_type=str(loss_params["loss_type"]),
        use_pos_weight=bool(loss_params["use_pos_weight"]),
        focal_alpha=float(loss_params["focal_alpha"]),
        focal_gamma=float(loss_params["focal_gamma"]),
        device=DEVICE,
        predict_all_days=PREDICT_ALL_DAYS,
        window_size=WINDOW_SIZE,
    )


def _build_training_run_log(
    trainer: Trainer,
    model: TCNTransformerModel,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    training_phase: str,
    validation_split_name: str,
    max_epochs: int,
    reproducibility_report: dict[str, object],
    params_override: dict[str, object],
) -> TrainingRunLog:
    train_X_shape, train_y_shape = _get_batch_shapes(train_dataloader, "Training")
    validation_X_shape, validation_y_shape = _get_batch_shapes(
        validation_dataloader,
        "Validation",
    )

    artifacts = TrainingArtifactLog(
        checkpoint_path=str(trainer.checkpoint_path),
        scaler_path=(
            None
            if trainer.scaler_path is None
            else str(trainer.scaler_path)
        ),
        feature_columns_count=len(trainer.feature_columns),
    )
    model_configuration = TrainingModelConfigurationLog(
        model_class=type(model).__name__,
        n_features=int(trainer.n_features),
        window_size=int(WINDOW_SIZE),
        model_dim=int(model.hidden_dim),
        hidden_dim=int(model.hidden_dim),
        tcn_dilations=list(TCN_DILATIONS),
        tcn_kernel_size=int(TCN_KERNEL_SIZE),
        tcn_activation=str(TCN_ACTIVATION),
        tcn_use_layer_norm=bool(TCN_USE_LAYER_NORM),
        tcn_norm_position=str(TCN_NORM_POSITION),
        transformer_num_layers=int(TRANSFORMER_NUM_LAYERS),
        transformer_num_heads=int(model.transformer_encoder.num_heads),
        transformer_ff_dim=int(model.transformer_encoder.feedforward_dim),
        transformer_activation=str(TRANSFORMER_ACTIVATION),
        positional_encoding_type=str(POSITIONAL_ENCODING_TYPE),
        profile_feature_mode=str(PROFILE_FEATURE_MODE),
        dropout=float(model.dropout),
        predict_all_days=bool(PREDICT_ALL_DAYS),
        use_causal_transformer_mask=bool(USE_CAUSAL_TRANSFORMER_MASK),
    )
    loss_params = (
        trainer.loss_hyperparameters
        if trainer.loss_hyperparameters is not None
        else _get_loss_params(params_override)
    )
    loss_configuration = TrainingLossConfigurationLog(
        loss_type=str(loss_params["loss_type"]),
        focal_alpha=float(loss_params["focal_alpha"]),
        focal_gamma=float(loss_params["focal_gamma"]),
        use_pos_weight=bool(loss_params["use_pos_weight"]),
    )
    evaluation_configuration = TrainingEvaluationConfigurationLog(
        evaluation_unit=str(EVALUATION_UNIT),
        overlapping_prediction_aggregation=str(
            OVERLAPPING_PREDICTION_AGGREGATION
        ),
        validation_split_name=str(validation_split_name),
        threshold_selection_metric=str(THRESHOLD_SELECTION_METRIC),
    )
    optimization = TrainingOptimizationLog(
        max_epochs=int(max_epochs),
        learning_rate=float(trainer.learning_rate),
        weight_decay=float(trainer.weight_decay),
        gradient_clip_norm=trainer.gradient_clip_norm,
        threshold=float(trainer.threshold),
        early_stopping_patience=int(trainer.early_stopping_patience),
        monitor_metric=CHECKPOINT_MONITOR_METRIC,
        device=str(trainer.device),
    )
    reproducibility = _build_training_reproducibility_log(
        reproducibility_report
    )
    shapes = TrainingShapeLog(
        train_X_batch_shape=train_X_shape,
        train_y_batch_shape=train_y_shape,
        validation_X_batch_shape=validation_X_shape,
        validation_y_batch_shape=validation_y_shape,
        model_input_shape=train_X_shape,
        expected_model_output_shape=train_y_shape,
    )

    return TrainingRunLog(
        training_mode=str(TRAINING_MODE),
        split_strategy=str(SPLIT_STRATEGY),
        training_phase=str(training_phase),
        model=model_configuration,
        loss=loss_configuration,
        evaluation=evaluation_configuration,
        optimization=optimization,
        reproducibility=reproducibility,
        artifacts=artifacts,
        shapes=shapes,
        params_override=dict(params_override),
        environment=collect_training_environment(
            device=str(trainer.device),
            pythonhashseed_at_entry=reproducibility_report.get("pythonhashseed_at_entry"),
        ),
    )


def _build_training_reproducibility_log(
    reproducibility_report: dict[str, object],
) -> TrainingReproducibilityLog:
    return TrainingReproducibilityLog(
        random_seed=int(reproducibility_report["random_seed"]),
        deterministic_training=bool(
            reproducibility_report["deterministic_training"]
        ),
        deterministic_warn_only=bool(
            reproducibility_report["deterministic_warn_only"]
        ),
        cudnn_deterministic=bool(
            reproducibility_report["cudnn_deterministic"]
        ),
        cudnn_benchmark=bool(reproducibility_report["cudnn_benchmark"]),
        cublas_workspace_config=str(
            reproducibility_report["cublas_workspace_config"]
        ),
        torch_initial_seed=int(reproducibility_report["torch_initial_seed"]),
        cuda_available=bool(reproducibility_report["cuda_available"]),
        cuda_device_count=int(reproducibility_report["cuda_device_count"]),
        torch_version=str(reproducibility_report["torch_version"]),
    )


def _get_batch_size(params_override: dict[str, object]) -> int:
    return int(params_override.get("batch_size", BATCH_SIZE))


def _get_batch_shapes(
    dataloader: DataLoader,
    source_name: str,
) -> tuple[list[int], list[int]]:
    dataset = dataloader.dataset

    if not isinstance(dataset, SequenceDataset):
        raise TypeError(
            f"Expected {source_name} DataLoader dataset to be SequenceDataset. "
            f"Current type: {type(dataset).__name__}."
        )

    if len(dataset) == 0:
        raise ValueError(f"{source_name} DataLoader dataset is empty.")

    dataloader_batch_size = dataloader.batch_size
    batch_size = (
        len(dataset)
        if dataloader_batch_size is None
        else min(int(dataloader_batch_size), len(dataset))
    )
    X_shape = [batch_size, *[int(dim) for dim in dataset.X.shape[1:]]]
    y_shape = [batch_size, *[int(dim) for dim in dataset.y.shape[1:]]]

    return X_shape, y_shape


def _get_dropout(params_override: dict[str, object]) -> float:
    return float(params_override.get("dropout", DROPOUT))


def _get_float_override(
    params_override: dict[str, object],
    key: str,
    default_value: float,
) -> float:
    return float(params_override.get(key, default_value))


def _get_gradient_clip_norm(
    params_override: dict[str, object],
) -> float | None:
    value = params_override.get("gradient_clip_norm", GRADIENT_CLIP_NORM)

    if value is None:
        return None

    return float(value)


def _get_learning_rate(params_override: dict[str, object]) -> float:
    return _get_float_override(params_override, "learning_rate", LEARNING_RATE)


def _get_loss_params(
    params_override: dict[str, object],
    force_disable_pos_weight: bool = False,
) -> dict[str, object]:
    loss_type = str(params_override.get("loss_type", LOSS_TYPE))
    use_pos_weight = bool(params_override.get("use_pos_weight", USE_POS_WEIGHT))

    if force_disable_pos_weight:
        use_pos_weight = False

    return {
        "loss_type": loss_type,
        "focal_alpha": float(params_override.get("focal_alpha", FOCAL_ALPHA)),
        "focal_gamma": float(params_override.get("focal_gamma", FOCAL_GAMMA)),
        "use_pos_weight": use_pos_weight,
    }


def _get_model_dim(params_override: dict[str, object]) -> int:
    return int(params_override.get("model_dim", MODEL_DIM))


def _get_n_features_from_train_dataloader(train_dataloader: DataLoader) -> int:
    dataset = train_dataloader.dataset

    if not isinstance(dataset, SequenceDataset):
        raise TypeError(
            "Expected train DataLoader dataset to be SequenceDataset. "
            f"Current type: {type(dataset).__name__}."
        )

    if dataset.X.ndim != 3:
        raise ValueError(
            "train_X.pt must have shape (n_windows, WINDOW_SIZE, n_features). "
            f"Current shape: {tuple(dataset.X.shape)}."
        )

    return int(dataset.X.shape[-1])


def _get_transformer_ff_dim(params_override: dict[str, object]) -> int:
    return int(params_override.get("transformer_ff_dim", TRANSFORMER_FF_DIM))


def _get_transformer_num_heads(params_override: dict[str, object]) -> int:
    return int(params_override.get("transformer_num_heads", TRANSFORMER_NUM_HEADS))


def _get_weight_decay(params_override: dict[str, object]) -> float:
    return _get_float_override(params_override, "weight_decay", WEIGHT_DECAY)


def _get_standard_scaler_path():
    scaler_path = SCALERS_DIR / STANDARD_SCALER_FILE

    if not scaler_path.exists() or not scaler_path.is_file():
        raise FileNotFoundError(f"Standard scaler file not found: {scaler_path}")

    return scaler_path


def _load_feature_columns() -> list[str]:
    feature_columns_path = SPLITS_DIR / FEATURE_COLUMNS_FILE

    if not feature_columns_path.exists() or not feature_columns_path.is_file():
        raise FileNotFoundError(
            f"Feature columns report not found: {feature_columns_path}"
        )

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


def _save_training_epoch_logs(
    history: TrainingHistory,
    output_file: str = TRAINING_EPOCH_LOGS_FILE,
) -> None:
    output_path = LOGS_DIR / output_file

    print(f"Saving training epoch logs to {output_path} ...")
    save_training_epoch_logs(_build_training_epoch_log_rows(history), output_path)


def _save_training_metrics(
    history: TrainingHistory,
    output_file: str = TRAIN_METRICS_FILE,
) -> None:
    output_path = METRICS_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving training metrics to {output_path} ...")
    save_json(history, output_path)


def _save_training_run_log(
    trainer: Trainer,
    model: TCNTransformerModel,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    training_phase: str,
    validation_split_name: str,
    output_file: str = TRAINING_RUN_LOG_FILE,
    max_epochs: int = MAX_EPOCHS,
    reproducibility_report: dict[str, object] | None = None,
    params_override: dict[str, object] | None = None,
) -> None:
    output_path = LOGS_DIR / output_file
    params_override = _validate_params_override(params_override)

    if reproducibility_report is None:
        reproducibility_report = build_reproducibility_report(
            seed=RANDOM_SEED,
            deterministic=DETERMINISTIC_TRAINING,
            warn_only=DETERMINISTIC_WARN_ONLY,
            cudnn_benchmark=CUDNN_BENCHMARK,
            cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
        )

    print(f"Saving training run log to {output_path} ...")
    save_training_run_log(
        _build_training_run_log(
            trainer=trainer,
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            training_phase=training_phase,
            validation_split_name=validation_split_name,
            max_epochs=max_epochs,
            reproducibility_report=reproducibility_report,
            params_override=params_override,
        ),
        output_path,
    )


def _validate_params_override(
    params_override: dict[str, object] | None,
) -> dict[str, object]:
    if params_override is None:
        return {}

    if not isinstance(params_override, dict):
        raise TypeError("params_override must be a dictionary or None.")

    unknown_keys = sorted(set(params_override) - ALLOWED_PARAMS_OVERRIDE_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown params_override keys: {unknown_keys}")

    normalized = dict(params_override)
    if "hidden_dim" in normalized and "model_dim" not in normalized:
        normalized["model_dim"] = int(normalized["hidden_dim"])

    if "model_dim" in normalized and "hidden_dim" not in normalized:
        normalized["hidden_dim"] = int(normalized["model_dim"])

    _validate_params_override_values(normalized)

    return normalized


def _validate_params_override_values(params_override: dict[str, object]) -> None:
    model_dim = int(params_override.get("model_dim", MODEL_DIM))
    hidden_dim = int(params_override.get("hidden_dim", model_dim))
    transformer_num_heads = int(
        params_override.get("transformer_num_heads", TRANSFORMER_NUM_HEADS)
    )
    transformer_ff_dim = int(
        params_override.get("transformer_ff_dim", TRANSFORMER_FF_DIM)
    )
    loss_type = str(params_override.get("loss_type", LOSS_TYPE))
    use_pos_weight = bool(params_override.get("use_pos_weight", USE_POS_WEIGHT))

    if hidden_dim != model_dim:
        raise ValueError("hidden_dim must be equal to model_dim.")

    if model_dim <= 0:
        raise ValueError("model_dim must be positive.")

    if transformer_num_heads <= 0:
        raise ValueError("transformer_num_heads must be positive.")

    if model_dim % transformer_num_heads != 0:
        raise ValueError("model_dim must be divisible by transformer_num_heads.")

    if transformer_ff_dim <= 0:
        raise ValueError("transformer_ff_dim must be positive.")

    if _get_batch_size(params_override) <= 0:
        raise ValueError("batch_size must be positive.")

    if _get_learning_rate(params_override) <= 0:
        raise ValueError("learning_rate must be positive.")

    if _get_weight_decay(params_override) < 0:
        raise ValueError("weight_decay cannot be negative.")

    gradient_clip_norm = _get_gradient_clip_norm(params_override)
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when enabled.")

    if _get_dropout(params_override) < 0 or _get_dropout(params_override) > 1:
        raise ValueError("dropout must be between 0 and 1.")

    if loss_type not in {"focal", "bce"}:
        raise ValueError(f"Unsupported loss_type override: {loss_type}")

    if loss_type == "focal" and use_pos_weight:
        raise ValueError("use_pos_weight cannot be True when loss_type='focal'.")

    focal_alpha = float(params_override.get("focal_alpha", FOCAL_ALPHA))
    if focal_alpha <= 0 or focal_alpha >= 1:
        raise ValueError("focal_alpha must be between 0 and 1.")

    focal_gamma = float(params_override.get("focal_gamma", FOCAL_GAMMA))
    if focal_gamma < 0:
        raise ValueError("focal_gamma cannot be negative.")


def _validate_feature_columns_match_n_features(
    feature_columns: list[str],
    n_features: int,
) -> None:
    if len(feature_columns) != n_features:
        raise ValueError(
            "feature_columns.json must match train_X.pt feature dimension. "
            f"feature_columns: {len(feature_columns)}, n_features: {n_features}."
        )
