import json
import math

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.itd_tcnTransformer.config.settings import (
    EVALUATION_UNIT,
    MIN_THRESHOLD_SELECTION_PRECISION,
    THRESHOLD_SELECTION_METRIC,
    THRESHOLD_SELECTION_STEP,
)
from src.itd_tcnTransformer.utils.io import save_json


DEFAULT_THRESHOLD_STEP = THRESHOLD_SELECTION_STEP

__all__ = [
    "ThresholdCandidate",
    "ThresholdSelectionReport",
    "load_threshold_selection_report",
    "save_threshold_selection_report",
    "select_threshold",
]


@dataclass
class ThresholdCandidate:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive_count: int
    predicted_negative_count: int


@dataclass
class ThresholdSelectionReport:
    selected_threshold: float | None
    selection_criterion: str
    threshold_selection_metric: str
    threshold_selection_step: float
    evaluation_unit: str
    selected_precision: float | None
    selected_recall: float | None
    selected_f1: float | None
    min_recall: float | None
    min_precision: float | None
    validation_rows: int
    validation_positive_rows: int
    validation_negative_rows: int
    prediction_count: int
    positive_count: int
    negative_count: int
    selected_candidate: ThresholdCandidate | None
    candidates: list[ThresholdCandidate] = field(default_factory=list)
    status: str = "available"
    reason: str | None = None
    prediction_scope: str = "all_positions"
    temporal_position: int | None = None
    window_size: int | None = None


def load_threshold_selection_report(input_path: Path) -> ThresholdSelectionReport:
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Threshold selection report not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    selected_candidate_data = data.get("selected_candidate")
    selected_candidate = (
        None
        if selected_candidate_data is None
        else ThresholdCandidate(**selected_candidate_data)
    )
    candidates = [
        ThresholdCandidate(**candidate)
        for candidate in data["candidates"]
    ]

    return ThresholdSelectionReport(
        selected_threshold=(
            None
            if data.get("selected_threshold") is None
            else float(data["selected_threshold"])
        ),
        selection_criterion=str(data["selection_criterion"]),
        threshold_selection_metric=str(data["threshold_selection_metric"]),
        threshold_selection_step=float(
            data.get("threshold_selection_step", DEFAULT_THRESHOLD_STEP)
        ),
        evaluation_unit=str(data["evaluation_unit"]),
        selected_precision=(
            None
            if data.get("selected_precision") is None
            else float(data["selected_precision"])
        ),
        selected_recall=(
            None
            if data.get("selected_recall") is None
            else float(data["selected_recall"])
        ),
        selected_f1=(
            None
            if data.get("selected_f1") is None
            else float(data["selected_f1"])
        ),
        min_recall=(
            None
            if data["min_recall"] is None
            else float(data["min_recall"])
        ),
        min_precision=(
            None
            if data.get("min_precision") is None
            else float(data["min_precision"])
        ),
        validation_rows=int(data["validation_rows"]),
        validation_positive_rows=int(data["validation_positive_rows"]),
        validation_negative_rows=int(data["validation_negative_rows"]),
        prediction_count=int(data["prediction_count"]),
        positive_count=int(data["positive_count"]),
        negative_count=int(data["negative_count"]),
        selected_candidate=selected_candidate,
        candidates=candidates,
        status=str(data.get("status", "available")),
        reason=(None if data.get("reason") is None else str(data["reason"])),
        prediction_scope=str(data.get("prediction_scope", "all_positions")),
        temporal_position=(
            None
            if data.get("temporal_position") is None
            else int(data["temporal_position"])
        ),
        window_size=(
            None
            if data.get("window_size") is None
            else int(data["window_size"])
        ),
    )


def save_threshold_selection_report(
    report: ThresholdSelectionReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output_path)


def select_threshold(
    validation_predictions: pd.DataFrame,
    threshold_selection_metric: str = THRESHOLD_SELECTION_METRIC,
    evaluation_unit: str = EVALUATION_UNIT,
    min_recall: float | None = None,
    min_precision: float | None = MIN_THRESHOLD_SELECTION_PRECISION,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
    allow_unavailable: bool = False,
    prediction_scope: str = "all_positions",
    temporal_position: int | None = None,
    window_size: int | None = None,
) -> ThresholdSelectionReport:
    """Select a validation threshold, optionally reporting unavailable selection.

    ``allow_unavailable`` is an opt-in for final evaluation subsets that cannot
    support threshold selection. Training and HPO retain strict failure
    semantics through its default ``False`` value.
    """
    effective_min_precision = (
        min_precision
        if threshold_selection_metric == "recall_at_min_precision"
        else None
    )
    _validate_threshold_selection_inputs(
        validation_predictions,
        threshold_selection_metric,
        evaluation_unit,
        min_recall,
        effective_min_precision,
        threshold_step,
    )

    if validation_predictions.empty:
        if allow_unavailable:
            return _build_unavailable_report(
                validation_predictions,
                threshold_selection_metric,
                evaluation_unit,
                min_recall,
                effective_min_precision,
                threshold_step,
                "Validation predictions are empty.",
                prediction_scope,
                temporal_position,
                window_size,
            )
        raise ValueError("Validation predictions cannot be empty.")

    candidates = _build_threshold_candidates(
        validation_predictions,
        threshold_step,
    )
    if validation_predictions["label"].nunique() < 2 and allow_unavailable:
        return _build_unavailable_report(
            validation_predictions,
            threshold_selection_metric,
            evaluation_unit,
            min_recall,
            effective_min_precision,
            threshold_step,
            "Validation predictions contain a single label class.",
            prediction_scope,
            temporal_position,
            window_size,
            candidates,
        )

    selected_candidate = _select_candidate(
        candidates,
        threshold_selection_metric,
        min_recall,
        effective_min_precision,
        allow_unavailable,
    )

    if selected_candidate is None:
        reason = _get_unavailable_selection_reason(
            candidates,
            threshold_selection_metric,
            min_recall,
            effective_min_precision,
        )
        return _build_unavailable_report(
            validation_predictions,
            threshold_selection_metric,
            evaluation_unit,
            min_recall,
            effective_min_precision,
            threshold_step,
            reason,
            prediction_scope,
            temporal_position,
            window_size,
            candidates,
        )

    return ThresholdSelectionReport(
        selected_threshold=float(selected_candidate.threshold),
        selection_criterion=_get_selection_criterion(
            threshold_selection_metric,
            min_recall,
            effective_min_precision,
        ),
        threshold_selection_metric=threshold_selection_metric,
        threshold_selection_step=float(threshold_step),
        evaluation_unit=str(evaluation_unit),
        selected_precision=float(selected_candidate.precision),
        selected_recall=float(selected_candidate.recall),
        selected_f1=float(selected_candidate.f1),
        min_recall=min_recall,
        min_precision=effective_min_precision,
        validation_rows=int(len(validation_predictions.index)),
        validation_positive_rows=int(validation_predictions["label"].eq(1).sum()),
        validation_negative_rows=int(validation_predictions["label"].eq(0).sum()),
        prediction_count=int(len(validation_predictions.index)),
        positive_count=int(validation_predictions["label"].eq(1).sum()),
        negative_count=int(validation_predictions["label"].eq(0).sum()),
        selected_candidate=selected_candidate,
        candidates=candidates,
        prediction_scope=prediction_scope,
        temporal_position=temporal_position,
        window_size=window_size,
    )


def _build_unavailable_report(
    validation_predictions: pd.DataFrame,
    threshold_selection_metric: str,
    evaluation_unit: str,
    min_recall: float | None,
    min_precision: float | None,
    threshold_step: float,
    reason: str,
    prediction_scope: str,
    temporal_position: int | None,
    window_size: int | None,
    candidates: list[ThresholdCandidate] | None = None,
) -> ThresholdSelectionReport:
    labels = validation_predictions["label"]
    positive_count = int(labels.eq(1).sum())
    negative_count = int(labels.eq(0).sum())
    return ThresholdSelectionReport(
        selected_threshold=None,
        selection_criterion=_get_selection_criterion(
            threshold_selection_metric, min_recall, min_precision
        ),
        threshold_selection_metric=threshold_selection_metric,
        threshold_selection_step=float(threshold_step),
        evaluation_unit=str(evaluation_unit),
        selected_precision=None,
        selected_recall=None,
        selected_f1=None,
        min_recall=min_recall,
        min_precision=min_precision,
        validation_rows=int(len(validation_predictions.index)),
        validation_positive_rows=positive_count,
        validation_negative_rows=negative_count,
        prediction_count=int(len(validation_predictions.index)),
        positive_count=positive_count,
        negative_count=negative_count,
        selected_candidate=None,
        candidates=[] if candidates is None else candidates,
        status="unavailable",
        reason=reason,
        prediction_scope=prediction_scope,
        temporal_position=temporal_position,
        window_size=window_size,
    )


def _get_unavailable_selection_reason(
    candidates: list[ThresholdCandidate],
    threshold_selection_metric: str,
    min_recall: float | None,
    min_precision: float | None,
) -> str:
    if min_recall is not None and not any(
        candidate.recall >= min_recall for candidate in candidates
    ):
        return (
            "No threshold candidate satisfies the requested minimum recall. "
            f"Requested min_recall: {min_recall}."
        )

    if threshold_selection_metric == "recall_at_min_precision":
        return (
            "No threshold candidate satisfies the requested minimum precision. "
            f"Requested min_precision: {min_precision}."
        )

    return "No threshold candidate satisfies the requested selection constraints."


def _build_threshold_candidates(
    validation_predictions: pd.DataFrame,
    threshold_step: float,
) -> list[ThresholdCandidate]:
    y_true = validation_predictions["label"].astype("int64").to_numpy()
    y_score = validation_predictions["probability"].astype("float64").to_numpy()
    thresholds = _build_threshold_grid(threshold_step)
    candidates: list[ThresholdCandidate] = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype("int64")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        candidates.append(
            ThresholdCandidate(
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                predicted_positive_count=int(y_pred.sum()),
                predicted_negative_count=int((y_pred == 0).sum()),
            )
        )

    return candidates


def _build_threshold_grid(threshold_step: float) -> list[float]:
    step = Decimal(str(threshold_step))
    threshold_count = int(
        (Decimal("1") / step).to_integral_value(rounding=ROUND_FLOOR)
    )
    thresholds = [
        float(step * index)
        for index in range(threshold_count + 1)
    ]

    if thresholds[-1] < 1.0:
        thresholds.append(1.0)
    else:
        thresholds[-1] = 1.0

    return thresholds


def _get_selection_criterion(
    threshold_selection_metric: str,
    min_recall: float | None,
    min_precision: float | None,
) -> str:
    if threshold_selection_metric == "recall_at_min_precision":
        if min_precision is None:
            raise ValueError(
                "min_precision is required when threshold_selection_metric is "
                "'recall_at_min_precision'."
            )
        return f"best_recall_at_min_precision_{float(min_precision):.2f}"

    if threshold_selection_metric == "f1" and min_recall is None:
        return "best_f1"

    return f"best_{threshold_selection_metric}_with_min_recall_{min_recall}"


def _select_candidate(
    candidates: list[ThresholdCandidate],
    threshold_selection_metric: str,
    min_recall: float | None,
    min_precision: float | None,
    allow_unavailable: bool = False,
) -> ThresholdCandidate | None:
    valid_candidates = candidates

    if min_recall is not None:
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.recall >= min_recall
        ]

        if not valid_candidates:
            if allow_unavailable:
                return None
            raise ValueError(
                "No threshold candidate satisfies the requested minimum recall. "
                f"Requested min_recall: {min_recall}."
            )

    if threshold_selection_metric == "f1":
        return max(
            valid_candidates,
            key=lambda candidate: (
                candidate.f1,
                candidate.precision,
                candidate.recall,
                -candidate.threshold,
            ),
        )

    if threshold_selection_metric == "recall_at_min_precision":
        if min_precision is None:
            raise ValueError(
                "min_precision is required when threshold_selection_metric is "
                "'recall_at_min_precision'."
            )

        valid_candidates = [
            candidate
            for candidate in valid_candidates
            if candidate.precision >= min_precision
        ]

        if not valid_candidates:
            if allow_unavailable:
                return None
            raise ValueError(
                "No threshold candidate satisfies the requested minimum precision. "
                f"Requested min_precision: {min_precision}."
            )

        return max(
            valid_candidates,
            key=lambda candidate: (
                candidate.recall,
                candidate.precision,
                candidate.f1,
                candidate.threshold,
            ),
        )

    raise ValueError(
        f"Unsupported threshold selection metric: {threshold_selection_metric}."
    )


def _validate_threshold_selection_inputs(
    validation_predictions: pd.DataFrame,
    threshold_selection_metric: str,
    evaluation_unit: str,
    min_recall: float | None,
    min_precision: float | None,
    threshold_step: float,
) -> None:
    required_columns = {"probability", "label"}
    missing_columns = required_columns - set(validation_predictions.columns)

    if missing_columns:
        raise ValueError(
            "Validation predictions are missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if threshold_selection_metric not in {"f1", "recall_at_min_precision"}:
        raise ValueError(
            f"Unsupported threshold selection metric: {threshold_selection_metric}."
        )

    if evaluation_unit not in {"window_timestep", "user_day_aggregated"}:
        raise ValueError(f"Unsupported evaluation_unit: {evaluation_unit}.")

    if min_recall is not None and (
        not math.isfinite(min_recall) or min_recall < 0 or min_recall > 1
    ):
        raise ValueError(
            f"min_recall must be between 0 and 1. Current value: {min_recall}."
        )

    if min_precision is not None and (
        not math.isfinite(min_precision) or min_precision < 0 or min_precision > 1
    ):
        raise ValueError(
            "min_precision must be between 0 and 1. "
            f"Current value: {min_precision}."
        )

    if (
        threshold_selection_metric == "recall_at_min_precision"
        and min_precision is None
    ):
        raise ValueError(
            "min_precision is required when threshold_selection_metric is "
            "'recall_at_min_precision'."
        )

    if not math.isfinite(threshold_step) or not 0 < threshold_step <= 1:
        raise ValueError(
            "threshold_step must be greater than 0 and at most 1. "
            f"Current value: {threshold_step}."
        )

    if validation_predictions.empty:
        return

    labels = validation_predictions["label"]
    if not labels.isin([0, 1]).all():
        raise ValueError("Validation labels must contain only the values 0 and 1.")

    probabilities = validation_predictions["probability"]
    if not pd.api.types.is_numeric_dtype(probabilities):
        raise ValueError("Validation probabilities must be numeric values in [0, 1].")

    probability_values = probabilities.astype("float64").to_numpy()
    if (
        not all(math.isfinite(value) for value in probability_values)
        or (probability_values < 0).any()
        or (probability_values > 1).any()
    ):
        raise ValueError(
            "Validation probabilities must be finite values between 0 and 1."
        )
