from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.itd_tcnTransformer.config.paths import (
    DAILY_LABELS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SEQUENCES_DIR,
)
from src.itd_tcnTransformer.config.settings import validate_device
from src.itd_tcnTransformer.config.settings import (
    DEVICE,
    EVALUATION_UNIT,
    FIXED_EVALUATION_THRESHOLDS,
    OVERLAPPING_PREDICTION_AGGREGATION,
    SPLIT_STRATEGY,
    THRESHOLD_SELECTION_STEP,
    TRAINING_MODE,
)
from src.itd_tcnTransformer.evaluation.metrics import (
    TestEvaluationReport,
    add_predicted_labels,
    calculate_binary_classification_metrics,
    calculate_fixed_threshold_metrics,
    calculate_false_positive_breakdown,
    calculate_scenario_metrics,
    calculate_user_metrics,
    save_test_evaluation_report,
)
from src.itd_tcnTransformer.evaluation.prediction_units import (
    build_evaluation_predictions,
    build_window_timestep_predictions,
)
from src.itd_tcnTransformer.evaluation.thresholds import (
    load_threshold_selection_report,
    save_threshold_selection_report,
    select_threshold,
)
from src.itd_tcnTransformer.models.tcn_transformer import TCNTransformerModel
from src.itd_tcnTransformer.sequences.sequence_dataset import create_sequence_dataloader
from src.itd_tcnTransformer.training.trainer import (
    BEST_CHECKPOINT_FILE,
    FINETUNE_BEST_CHECKPOINT_FILE,
)


TEST_EVALUATION_REPORT_FILE = "test_evaluation_report.json"
DAILY_LABEL_METADATA_FILE = "daily_label_metadata.parquet"
TEST_PREDICTIONS_FILE = "test_predictions.parquet"
TEST_SPLIT_NAME = "test"
TEST_WINDOW_PREDICTIONS_FILE = "test_window_predictions.parquet"
VALIDATION_PREDICTIONS_FILE = "validation_predictions.parquet"
ARTICLE_VALIDATION_SPLIT_NAME = "finetune_validation"
VALIDATION_SPLIT_NAME = "validation"
VALIDATION_THRESHOLD_FILE = "validation_threshold.json"
VALIDATION_FINAL_POSITION_THRESHOLD_FILE = "validation_final_position_threshold.json"
VALIDATION_WINDOW_PREDICTIONS_FILE = "validation_window_predictions.parquet"

__all__ = ["EvaluationResult", "run_evaluation", "select_final_position_predictions"]


@dataclass
class EvaluationResult:
    test_predictions_count: int
    test_predictions_path: str
    test_report_path: str
    test_window_predictions_count: int
    test_window_predictions_path: str
    threshold_report_path: str
    validation_predictions_count: int
    validation_predictions_path: str
    validation_window_predictions_count: int
    validation_window_predictions_path: str
    final_position_threshold_report_path: str | None = None


def run_evaluation(
    checkpoint_path: Path | str | None = None,
    metrics_dir: Path | str = METRICS_DIR,
    predictions_dir: Path | str = PREDICTIONS_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    sequences_dir: Path | str = SEQUENCES_DIR,
    daily_labels_dir: Path | str = DAILY_LABELS_DIR,
    device: torch.device | str = DEVICE,
    aggregation: str = OVERLAPPING_PREDICTION_AGGREGATION,
) -> EvaluationResult:
    validate_device(device)
    checkpoint_file = _resolve_checkpoint_path(checkpoint_path)
    metrics_dir = Path(metrics_dir)
    predictions_dir = Path(predictions_dir)
    reports_dir = Path(reports_dir)
    sequences_dir = Path(sequences_dir)
    daily_labels_dir = Path(daily_labels_dir)
    device = torch.device(device)

    print("Loading best checkpoint ...")
    checkpoint = _load_checkpoint(checkpoint_file, device)
    _validate_checkpoint_loss_hyperparameters(checkpoint)
    _validate_checkpoint_evaluation_configuration(checkpoint)

    print("Reconstructing model ...")
    model = _reconstruct_model_from_checkpoint(checkpoint, device)
    predict_all_days = bool(checkpoint["model_hyperparameters"]["predict_all_days"])
    window_size = int(checkpoint["model_hyperparameters"]["window_size"])
    validation_split_name = _resolve_validation_split_name()

    print("Generating validation predictions ...")
    validation_window_predictions, validation_predictions = _evaluate_split_predictions(
        model=model,
        split_name=validation_split_name,
        sequences_dir=sequences_dir,
        device=device,
        predict_all_days=predict_all_days,
        aggregation=aggregation,
    )

    validation_window_predictions_path = (
        predictions_dir / VALIDATION_WINDOW_PREDICTIONS_FILE
    )
    validation_predictions_path = predictions_dir / VALIDATION_PREDICTIONS_FILE
    _save_predictions(validation_window_predictions, validation_window_predictions_path)
    _save_predictions(validation_predictions, validation_predictions_path)

    print("Selecting validation threshold ...")
    threshold_report = select_threshold(
        validation_predictions,
        evaluation_unit=EVALUATION_UNIT,
        threshold_step=THRESHOLD_SELECTION_STEP,
        allow_unavailable=True,
    )
    threshold_report_path = metrics_dir / VALIDATION_THRESHOLD_FILE
    save_threshold_selection_report(threshold_report, threshold_report_path)

    print("Loading selected validation threshold ...")
    threshold_report = load_threshold_selection_report(threshold_report_path)

    validation_final_predictions = select_final_position_predictions(
        validation_window_predictions, window_size,
    )
    final_threshold_report = select_threshold(
        validation_final_predictions,
        evaluation_unit="window_timestep",
        threshold_step=THRESHOLD_SELECTION_STEP,
        allow_unavailable=True,
        prediction_scope="final_position",
        temporal_position=window_size - 1,
        window_size=window_size,
    )
    final_threshold_report_path = metrics_dir / VALIDATION_FINAL_POSITION_THRESHOLD_FILE
    save_threshold_selection_report(final_threshold_report, final_threshold_report_path)

    print("Generating test predictions ...")
    test_window_predictions, test_predictions = _evaluate_split_predictions(
        model=model,
        split_name=TEST_SPLIT_NAME,
        sequences_dir=sequences_dir,
        device=device,
        predict_all_days=predict_all_days,
        aggregation=aggregation,
    )
    test_predictions = add_predicted_labels(
        test_predictions,
        threshold_report.selected_threshold,
    )
    test_predictions = _add_scenario_metadata(test_predictions, daily_labels_dir)

    print("Calculating test metrics ...")
    test_metrics = calculate_binary_classification_metrics(
        test_predictions,
        threshold_report.selected_threshold,
    )
    scenario_metrics = calculate_scenario_metrics(
        test_predictions,
        threshold_report.selected_threshold,
    ) if threshold_report.selected_threshold is not None else None
    fixed_threshold_metrics = calculate_fixed_threshold_metrics(
        test_predictions,
        FIXED_EVALUATION_THRESHOLDS,
    )
    user_metrics = calculate_user_metrics(
        test_predictions,
        threshold_report.selected_threshold,
    ) if threshold_report.selected_threshold is not None else None
    final_predictions = select_final_position_predictions(
        test_window_predictions, window_size,
    )
    test_report = TestEvaluationReport(
        split_name=TEST_SPLIT_NAME,
        evaluation_unit=str(EVALUATION_UNIT),
        overlapping_prediction_aggregation=_resolve_report_aggregation(aggregation),
        threshold=threshold_report.selected_threshold,
        predictions_file=str(predictions_dir / TEST_PREDICTIONS_FILE),
        validation_predictions_file=str(validation_predictions_path),
        test_window_predictions_file=str(
            predictions_dir / TEST_WINDOW_PREDICTIONS_FILE
        ),
        threshold_file=str(threshold_report_path),
        threshold_selection_criterion=threshold_report.selection_criterion,
        threshold_selection_metric=threshold_report.threshold_selection_metric,
        prediction_count_before_aggregation=int(len(test_window_predictions.index)),
        prediction_count_after_aggregation=int(len(test_predictions.index)),
        positive_count=int(test_predictions["label"].eq(1).sum()),
        negative_count=int(test_predictions["label"].eq(0).sum()),
        false_positive_breakdown=(
            calculate_false_positive_breakdown(test_predictions)
            if threshold_report.selected_threshold is not None else None
        ),
        metrics=test_metrics,
        fixed_threshold_metrics=fixed_threshold_metrics,
        scenario_metrics=scenario_metrics,
        top_false_positive_users=sorted(
            user_metrics,
            key=lambda report: (-report.false_positives, report.user),
        )[:20] if user_metrics is not None else None,
        top_false_negative_users=sorted(
            user_metrics,
            key=lambda report: (-report.false_negatives, report.user),
        )[:20] if user_metrics is not None else None,
        threshold_selection_status=threshold_report.status,
        threshold_selection_reason=threshold_report.reason,
        final_position={
            "evaluation_unit": "window_timestep",
            "prediction_scope": "final_position",
            "window_size": window_size,
            "temporal_position": window_size - 1,
            "position": window_size,
            "filter": f"temporal_position == {window_size - 1}",
            "interpretation": (
                "Current observed user-day after its features are available, "
                f"using the preceding {window_size - 1} observed user-day records. "
                "This does not measure within-day detection or real-time latency."
            ),
            "test_window_predictions_file": str(predictions_dir / TEST_WINDOW_PREDICTIONS_FILE),
            "validation_window_predictions_file": str(validation_window_predictions_path),
            "global_threshold": {
                "threshold_file": str(threshold_report_path),
                "threshold_selection_status": threshold_report.status,
                "threshold_selection_reason": threshold_report.reason,
                "metrics": calculate_binary_classification_metrics(
                    final_predictions, threshold_report.selected_threshold,
                ),
            },
            "final_position_threshold": {
                "threshold_file": str(final_threshold_report_path),
                "threshold_selection_status": final_threshold_report.status,
                "threshold_selection_reason": final_threshold_report.reason,
                "metrics": calculate_binary_classification_metrics(
                    final_predictions, final_threshold_report.selected_threshold,
                ),
            },
        },
    )

    print("Saving test predictions and report ...")
    test_window_predictions_path = predictions_dir / TEST_WINDOW_PREDICTIONS_FILE
    test_predictions_path = predictions_dir / TEST_PREDICTIONS_FILE
    test_report_path = reports_dir / TEST_EVALUATION_REPORT_FILE
    _save_predictions(test_window_predictions, test_window_predictions_path)
    _save_predictions(test_predictions, test_predictions_path)
    save_test_evaluation_report(test_report, test_report_path)

    return EvaluationResult(
        test_predictions_count=int(len(test_predictions.index)),
        test_predictions_path=str(test_predictions_path),
        test_report_path=str(test_report_path),
        test_window_predictions_count=int(len(test_window_predictions.index)),
        test_window_predictions_path=str(test_window_predictions_path),
        threshold_report_path=str(threshold_report_path),
        validation_predictions_count=int(len(validation_predictions.index)),
        validation_predictions_path=str(validation_predictions_path),
        validation_window_predictions_count=int(len(validation_window_predictions.index)),
        validation_window_predictions_path=str(validation_window_predictions_path),
        final_position_threshold_report_path=str(final_threshold_report_path),
    )


def select_final_position_predictions(
    window_predictions: pd.DataFrame, window_size: int,
) -> pd.DataFrame:
    """Select current-day scores before any aggregation of overlapping windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    required = {"temporal_position", "window_index", "day", "window_end_day"}
    missing = required - set(window_predictions.columns)
    if missing:
        raise ValueError(f"Window predictions are missing columns: {sorted(missing)}")
    if window_predictions.empty:
        return window_predictions.copy().reset_index(drop=True)
    positions = window_predictions["temporal_position"]
    if not pd.api.types.is_numeric_dtype(positions):
        raise ValueError("Temporal positions must be numeric integers.")
    valid_positions = (
        positions.notna()
        & positions.between(0, window_size - 1)
        & positions.eq(positions.round())
    )
    if not valid_positions.all():
        raise ValueError("Invalid temporal positions in window predictions.")
    final = window_predictions.loc[positions.eq(window_size - 1)].copy()
    if final["window_index"].duplicated().any():
        raise ValueError("Multiple final-position predictions for the same window.")
    if not pd.to_datetime(final["day"]).eq(pd.to_datetime(final["window_end_day"])).all():
        raise ValueError("Final-position day must equal window_end_day.")
    return final.reset_index(drop=True)


def _build_model_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_hyperparameters = checkpoint["model_hyperparameters"]
    _validate_checkpoint_model_hyperparameters(model_hyperparameters)

    return {
        "n_features": int(checkpoint["n_features"]),
        "window_size": int(model_hyperparameters["window_size"]),
        "hidden_dim": int(model_hyperparameters["hidden_dim"]),
        "tcn_dilations": list(model_hyperparameters["tcn_dilations"]),
        "kernel_size": int(model_hyperparameters["kernel_size"]),
        "tcn_activation": str(model_hyperparameters["tcn_activation"]),
        "tcn_use_layer_norm": bool(model_hyperparameters["tcn_use_layer_norm"]),
        "tcn_norm_position": str(model_hyperparameters["tcn_norm_position"]),
        "num_layers": int(model_hyperparameters["num_layers"]),
        "num_heads": int(model_hyperparameters["num_heads"]),
        "feedforward_dim": int(model_hyperparameters["feedforward_dim"]),
        "transformer_activation": str(
            model_hyperparameters["transformer_activation"]
        ),
        "positional_encoding_type": str(
            model_hyperparameters["positional_encoding_type"]
        ),
        "dropout": float(model_hyperparameters["dropout"]),
        "predict_all_days": bool(model_hyperparameters["predict_all_days"]),
        "use_causal_mask": bool(model_hyperparameters["use_causal_mask"]),
    }


def _evaluate_split_predictions(
    model: TCNTransformerModel,
    split_name: str,
    sequences_dir: Path,
    device: torch.device,
    predict_all_days: bool,
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataloader = create_sequence_dataloader(
        split_name=split_name,
        sequences_dir=sequences_dir,
        pin_memory=str(device).startswith("cuda"),
    )
    metadata = _load_split_metadata(sequences_dir, split_name)
    _validate_metadata_matches_dataloader(metadata, dataloader, split_name)

    window_predictions = _generate_window_predictions(
        model=model,
        dataloader=dataloader,
        metadata=metadata,
        device=device,
        predict_all_days=predict_all_days,
        split_name=split_name,
    )
    predictions_to_score = build_evaluation_predictions(
        window_predictions,
        evaluation_unit=EVALUATION_UNIT,
        aggregation=aggregation,
    )

    return window_predictions, predictions_to_score


def _generate_window_predictions(
    model: TCNTransformerModel,
    dataloader,
    metadata: pd.DataFrame,
    device: torch.device,
    predict_all_days: bool,
    split_name: str,
) -> pd.DataFrame:
    model.eval()
    prediction_parts: list[pd.DataFrame] = []
    window_offset = 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device, dtype=torch.float32)
            y = batch[1].to(device, dtype=torch.float32)
            logits = model(X)
            _validate_logit_label_shapes(logits, y)
            probabilities = torch.sigmoid(logits).detach().cpu()
            labels = y.detach().cpu()
            batch_size = int(X.shape[0])
            batch_metadata = metadata.iloc[window_offset:window_offset + batch_size]

            prediction_parts.append(
                build_window_timestep_predictions(
                    probabilities=probabilities,
                    labels=labels,
                    metadata=batch_metadata,
                    predict_all_days=predict_all_days,
                    split_name=split_name,
                    window_offset=window_offset,
                )
            )

            window_offset += batch_size

    if not prediction_parts:
        raise ValueError(f"{split_name} prediction generation produced no rows.")

    return pd.concat(prediction_parts, ignore_index=True)


def _load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str,
) -> dict[str, Any]:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint payload must be a dictionary. "
            f"Current type: {type(checkpoint).__name__}."
        )

    return checkpoint


def _load_split_metadata(sequences_dir: Path, split_name: str) -> pd.DataFrame:
    metadata_path = sequences_dir / f"{split_name}_metadata.parquet"

    if not metadata_path.exists() or not metadata_path.is_file():
        raise FileNotFoundError(f"{split_name} metadata file not found: {metadata_path}")

    return pd.read_parquet(metadata_path)


def _reconstruct_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device | str,
) -> TCNTransformerModel:
    model = TCNTransformerModel(**_build_model_kwargs(checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    return model


def _resolve_checkpoint_path(checkpoint_path: Path | str | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    if TRAINING_MODE == "two_stage":
        return MODELS_DIR / FINETUNE_BEST_CHECKPOINT_FILE

    return MODELS_DIR / BEST_CHECKPOINT_FILE


def _resolve_report_aggregation(aggregation: str) -> str:
    if EVALUATION_UNIT == "window_timestep":
        return "not_applicable"

    return str(aggregation)


def _resolve_validation_split_name() -> str:
    if SPLIT_STRATEGY == "user_article_two_phase":
        return ARTICLE_VALIDATION_SPLIT_NAME

    return VALIDATION_SPLIT_NAME


def _add_scenario_metadata(
    predictions: pd.DataFrame,
    daily_labels_dir: Path = DAILY_LABELS_DIR,
) -> pd.DataFrame:
    label_metadata_path = daily_labels_dir / DAILY_LABEL_METADATA_FILE

    if not label_metadata_path.exists() or not label_metadata_path.is_file():
        predictions = predictions.copy()
        predictions["scenario_id"] = pd.NA
        predictions["is_malicious_user"] = False
        return predictions

    label_metadata = pd.read_parquet(
        label_metadata_path,
        columns=["user", "scenario_id"],
    )
    label_metadata = label_metadata.copy()
    label_metadata["user"] = label_metadata["user"].astype("string").str.strip()
    label_metadata["scenario_id"] = label_metadata["scenario_id"].astype("int64")
    scenario_counts = label_metadata.groupby("user")["scenario_id"].nunique()
    multi_scenario_users = scenario_counts[scenario_counts > 1]
    if not multi_scenario_users.empty:
        raise ValueError(
            "Scenario metadata must have one scenario per malicious user. "
            f"Sample users: {multi_scenario_users.index[:10].tolist()}."
        )

    user_scenarios = (
        label_metadata
        .drop_duplicates("user")
        .rename(columns={"scenario_id": "user_scenario_id"})
    )
    predictions = predictions.copy()
    predictions["user"] = predictions["user"].astype("string").str.strip()
    predictions["day"] = pd.to_datetime(predictions["day"]).dt.floor("D")

    predictions = predictions.merge(user_scenarios, on="user", how="left")
    predictions["is_malicious_user"] = predictions["user_scenario_id"].notna()
    predictions["scenario_id"] = predictions["user_scenario_id"]
    predictions = predictions.drop(columns=["user_scenario_id"])

    return predictions


def _save_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )


def _validate_logit_label_shapes(logits: torch.Tensor, y: torch.Tensor) -> None:
    if logits.shape != y.shape:
        raise ValueError(
            "logits and y must have exactly the same shape during evaluation. "
            f"logits shape: {tuple(logits.shape)}, y shape: {tuple(y.shape)}."
        )


def _validate_checkpoint_model_hyperparameters(
    model_hyperparameters: dict[str, Any]
) -> None:
    required_hyperparameters = {
        "window_size",
        "hidden_dim",
        "tcn_dilations",
        "kernel_size",
        "tcn_activation",
        "tcn_use_layer_norm",
        "tcn_norm_position",
        "num_layers",
        "num_heads",
        "feedforward_dim",
        "transformer_activation",
        "positional_encoding_type",
        "dropout",
        "predict_all_days",
        "use_causal_mask",
    }
    missing_hyperparameters = sorted(
        required_hyperparameters - set(model_hyperparameters)
    )

    if missing_hyperparameters:
        raise ValueError(
            "Checkpoint is missing model hyperparameters required to reconstruct "
            f"the model: {missing_hyperparameters}"
        )


def _validate_checkpoint_loss_hyperparameters(checkpoint: dict[str, Any]) -> None:
    if "loss_hyperparameters" not in checkpoint:
        raise ValueError("Checkpoint is missing loss_hyperparameters.")

    loss_hyperparameters = checkpoint["loss_hyperparameters"]
    if not isinstance(loss_hyperparameters, dict):
        raise TypeError(
            "checkpoint['loss_hyperparameters'] must be a dictionary. "
            f"Current type: {type(loss_hyperparameters).__name__}."
        )

    required_hyperparameters = {
        "loss_type",
        "focal_alpha",
        "focal_gamma",
        "use_pos_weight",
    }
    missing_hyperparameters = sorted(
        required_hyperparameters - set(loss_hyperparameters)
    )

    if missing_hyperparameters:
        raise ValueError(
            "Checkpoint is missing loss hyperparameters: "
            f"{missing_hyperparameters}"
        )


def _validate_checkpoint_evaluation_configuration(checkpoint: dict[str, Any]) -> None:
    required_fields = {
        "evaluation_unit",
        "overlapping_prediction_aggregation",
        "validation_split_name",
    }
    missing_fields = sorted(required_fields - set(checkpoint))

    if missing_fields:
        raise ValueError(
            "Checkpoint is missing evaluation configuration fields: "
            f"{missing_fields}"
        )

    checkpoint_evaluation_unit = str(checkpoint["evaluation_unit"])
    checkpoint_aggregation = str(checkpoint["overlapping_prediction_aggregation"])

    if checkpoint_evaluation_unit != EVALUATION_UNIT:
        raise ValueError(
            "Checkpoint evaluation_unit does not match current EVALUATION_UNIT. "
            f"Checkpoint: {checkpoint_evaluation_unit}, current: {EVALUATION_UNIT}."
        )

    if (
        EVALUATION_UNIT == "user_day_aggregated"
        and checkpoint_aggregation != OVERLAPPING_PREDICTION_AGGREGATION
    ):
        raise ValueError(
            "Checkpoint overlapping prediction aggregation does not match current "
            "OVERLAPPING_PREDICTION_AGGREGATION. "
            f"Checkpoint: {checkpoint_aggregation}, "
            f"current: {OVERLAPPING_PREDICTION_AGGREGATION}."
        )


def _validate_metadata_matches_dataloader(
    metadata: pd.DataFrame,
    dataloader,
    split_name: str,
) -> None:
    dataset_length = len(dataloader.dataset)

    if len(metadata.index) != dataset_length:
        raise ValueError(
            f"{split_name} metadata must have one row per sequence. "
            f"Metadata rows: {len(metadata.index)}, sequences: {dataset_length}."
        )
