from typing import Any

import numpy as np
import pandas as pd
import torch


VALID_EVALUATION_UNITS = {"window_timestep", "user_day_aggregated"}
VALID_OVERLAPPING_PREDICTION_AGGREGATIONS = {"mean", "max", "min", "last"}

__all__ = [
    "aggregate_user_day_predictions",
    "build_evaluation_predictions",
    "build_window_timestep_predictions",
    "validate_evaluation_unit",
]


def aggregate_user_day_predictions(
    window_predictions: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    _validate_aggregation(aggregation)
    _validate_window_prediction_columns(window_predictions)
    _validate_repeated_labels(window_predictions)

    ordered_predictions = window_predictions.sort_values(
        ["user", "day", "window_end_day", "window_start_day", "temporal_position"]
    )
    grouped_predictions = ordered_predictions.groupby(["user", "day"], sort=True)
    base_aggregation = grouped_predictions.agg(
        split_name=("split_name", "first"),
        label=("label", "first"),
        prediction_count=("probability", "size"),
        first_window_start_day=("window_start_day", "min"),
        last_window_end_day=("window_end_day", "max"),
    ).reset_index()

    if aggregation == "last":
        probabilities = ordered_predictions.drop_duplicates(
            ["user", "day"],
            keep="last",
        )[["user", "day", "probability"]].reset_index(drop=True)
    else:
        probabilities = (
            grouped_predictions["probability"]
            .agg(aggregation)
            .rename("probability")
            .reset_index()
        )

    aggregated_predictions = base_aggregation.merge(
        probabilities,
        on=["user", "day"],
        how="left",
    )
    aggregated_predictions["has_overlapping_predictions"] = (
        aggregated_predictions["prediction_count"] > 1
    )
    aggregated_predictions["aggregation"] = aggregation

    return aggregated_predictions[
        [
            "split_name",
            "user",
            "day",
            "probability",
            "label",
            "prediction_count",
            "has_overlapping_predictions",
            "aggregation",
            "first_window_start_day",
            "last_window_end_day",
        ]
    ].sort_values(["user", "day"], ignore_index=True)


def build_evaluation_predictions(
    window_predictions: pd.DataFrame,
    evaluation_unit: str,
    aggregation: str,
) -> pd.DataFrame:
    validate_evaluation_unit(evaluation_unit, aggregation)

    if evaluation_unit == "window_timestep":
        return window_predictions.sort_values(
            ["user", "day", "window_end_day", "window_start_day", "temporal_position"],
            ignore_index=True,
        )

    return aggregate_user_day_predictions(window_predictions, aggregation)


def build_window_timestep_predictions(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    metadata: pd.DataFrame,
    predict_all_days: bool,
    split_name: str,
    window_offset: int = 0,
) -> pd.DataFrame:
    if len(metadata.index) != int(probabilities.shape[0]):
        raise ValueError(
            "metadata must have one row per prediction window. "
            f"Metadata rows: {len(metadata.index)}, windows: {probabilities.shape[0]}."
        )

    prediction_rows: list[dict[str, object]] = []

    for batch_index, (_, metadata_row) in enumerate(metadata.iterrows()):
        window_index = int(window_offset + batch_index)

        if predict_all_days:
            prediction_rows.extend(
                _build_predict_all_days_prediction_rows(
                    metadata_row=metadata_row,
                    probabilities=probabilities,
                    labels=labels,
                    batch_index=batch_index,
                    split_name=split_name,
                    window_index=window_index,
                )
            )
        else:
            prediction_rows.extend(
                _build_last_day_prediction_rows(
                    metadata_row=metadata_row,
                    probabilities=probabilities,
                    labels=labels,
                    batch_index=batch_index,
                    split_name=split_name,
                    window_index=window_index,
                )
            )

    if not prediction_rows:
        raise ValueError(f"{split_name} prediction generation produced no rows.")

    return _normalize_prediction_dates(pd.DataFrame(prediction_rows))


def validate_evaluation_unit(evaluation_unit: str, aggregation: str) -> None:
    if evaluation_unit not in VALID_EVALUATION_UNITS:
        raise ValueError(
            f"Invalid evaluation_unit: {evaluation_unit}. "
            f"Expected one of: {sorted(VALID_EVALUATION_UNITS)}."
        )

    if (
        evaluation_unit == "user_day_aggregated"
        and aggregation not in VALID_OVERLAPPING_PREDICTION_AGGREGATIONS
    ):
        raise ValueError(
            f"Invalid overlapping prediction aggregation: {aggregation}. "
            f"Expected one of: {sorted(VALID_OVERLAPPING_PREDICTION_AGGREGATIONS)}."
        )


def _build_last_day_prediction_rows(
    metadata_row: pd.Series,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    batch_index: int,
    split_name: str,
    window_index: int,
) -> list[dict[str, object]]:
    window_days = _normalize_window_days(metadata_row["window_days"])
    temporal_position = len(window_days) - 1

    return [
        _build_prediction_row(
            metadata_row=metadata_row,
            day=pd.Timestamp(metadata_row["end_day"]),
            split_name=split_name,
            temporal_position=temporal_position,
            probability=float(probabilities[batch_index].item()),
            label=float(labels[batch_index].item()),
            window_index=window_index,
        )
    ]


def _build_predict_all_days_prediction_rows(
    metadata_row: pd.Series,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    batch_index: int,
    split_name: str,
    window_index: int,
) -> list[dict[str, object]]:
    window_days = _normalize_window_days(metadata_row["window_days"])
    prediction_rows: list[dict[str, object]] = []

    for temporal_position, day in enumerate(window_days):
        prediction_rows.append(
            _build_prediction_row(
                metadata_row=metadata_row,
                day=day,
                split_name=split_name,
                temporal_position=temporal_position,
                probability=float(
                    probabilities[batch_index, temporal_position, 0].item()
                ),
                label=float(labels[batch_index, temporal_position, 0].item()),
                window_index=window_index,
            )
        )

    return prediction_rows


def _build_prediction_row(
    metadata_row: pd.Series,
    day: pd.Timestamp,
    split_name: str,
    temporal_position: int,
    probability: float,
    label: float,
    window_index: int,
) -> dict[str, object]:
    return {
        "split_name": str(split_name),
        "user": str(metadata_row["user"]),
        "day": pd.Timestamp(day).floor("D"),
        "window_start_day": pd.Timestamp(metadata_row["start_day"]).floor("D"),
        "window_end_day": pd.Timestamp(metadata_row["end_day"]).floor("D"),
        "temporal_position": int(temporal_position),
        "probability": float(probability),
        "label": int(label),
        "window_index": int(window_index),
    }


def _normalize_prediction_dates(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["day"] = pd.to_datetime(predictions["day"]).dt.floor("D")
    predictions["window_start_day"] = pd.to_datetime(
        predictions["window_start_day"]
    ).dt.floor("D")
    predictions["window_end_day"] = pd.to_datetime(
        predictions["window_end_day"]
    ).dt.floor("D")

    return predictions.sort_values(
        ["user", "day", "window_end_day", "window_start_day", "temporal_position"],
        ignore_index=True,
    )


def _normalize_window_days(window_days: Any) -> list[pd.Timestamp]:
    if isinstance(window_days, np.ndarray):
        raw_days = window_days.tolist()
    elif isinstance(window_days, (list, tuple)):
        raw_days = list(window_days)
    else:
        raise TypeError(
            "window_days must be stored as a list-like value in sequence metadata. "
            f"Current type: {type(window_days).__name__}."
        )

    return [pd.Timestamp(day).floor("D") for day in raw_days]


def _validate_aggregation(aggregation: str) -> None:
    if aggregation not in VALID_OVERLAPPING_PREDICTION_AGGREGATIONS:
        raise ValueError(
            f"Invalid overlapping prediction aggregation: {aggregation}. "
            f"Expected one of: {sorted(VALID_OVERLAPPING_PREDICTION_AGGREGATIONS)}."
        )


def _validate_repeated_labels(predictions: pd.DataFrame) -> None:
    label_counts = predictions.groupby(["user", "day"])["label"].nunique()
    conflicting_labels = label_counts[label_counts > 1]

    if not conflicting_labels.empty:
        raise ValueError(
            "Repeated predictions for the same user + day have conflicting labels. "
            f"Conflicting pairs: {len(conflicting_labels.index)}."
        )


def _validate_window_prediction_columns(predictions: pd.DataFrame) -> None:
    required_columns = {
        "split_name",
        "user",
        "day",
        "window_start_day",
        "window_end_day",
        "temporal_position",
        "probability",
        "label",
        "window_index",
    }
    missing_columns = sorted(required_columns - set(predictions.columns))

    if missing_columns:
        raise ValueError(
            "Window predictions are missing required columns: "
            f"{missing_columns}."
        )

    if predictions.empty:
        raise ValueError("Window predictions cannot be empty.")
