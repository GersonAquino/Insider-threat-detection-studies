from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.itd_tcnTransformer.utils.io import save_json


__all__ = [
    "BinaryClassificationMetrics",
    "ConfusionMatrixReport",
    "FalsePositiveBreakdownReport",
    "FixedThresholdMetricsReport",
    "ScenarioMetricReport",
    "TestEvaluationReport",
    "UserMetricReport",
    "add_predicted_labels",
    "calculate_binary_classification_metrics",
    "calculate_fixed_threshold_metrics",
    "calculate_false_positive_breakdown",
    "calculate_scenario_metrics",
    "calculate_user_metrics",
    "save_test_evaluation_report",
]


@dataclass
class BinaryClassificationMetrics:
    threshold: float | None
    rows: int
    positive_rows: int
    negative_rows: int
    predicted_positive_rows: int | None
    predicted_negative_rows: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    auroc: float | None
    auprc: float | None
    confusion_matrix: "ConfusionMatrixReport | None"
    prevalence: float | None = None
    status: str = "available"
    reason: str | None = None


@dataclass
class ConfusionMatrixReport:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass
class FalsePositiveBreakdownReport:
    total_false_positives: int
    normal_user_false_positives: int
    malicious_user_non_attack_false_positives: int


@dataclass
class ScenarioMetricReport:
    scenario_id: int
    evaluation_definition: str
    rows: int
    positive_count: int
    negative_count: int
    true_positives: int
    false_negatives: int
    false_positives_on_negative_rows: int
    false_positives_normal_users: int
    false_positives_malicious_users_non_attack: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass
class UserMetricReport:
    user: str
    is_malicious_user: bool
    scenario_id: int | None
    rows: int
    positive_count: int
    negative_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass
class FixedThresholdMetricsReport:
    threshold: float
    metrics: BinaryClassificationMetrics
    false_positive_breakdown: FalsePositiveBreakdownReport
    scenario_metrics: list[ScenarioMetricReport]


@dataclass
class TestEvaluationReport:
    split_name: str
    evaluation_unit: str
    overlapping_prediction_aggregation: str
    threshold: float | None
    predictions_file: str
    validation_predictions_file: str
    test_window_predictions_file: str
    threshold_file: str
    threshold_selection_criterion: str
    threshold_selection_metric: str
    prediction_count_before_aggregation: int
    prediction_count_after_aggregation: int
    positive_count: int
    negative_count: int
    false_positive_breakdown: FalsePositiveBreakdownReport | None
    metrics: BinaryClassificationMetrics
    fixed_threshold_metrics: list[FixedThresholdMetricsReport]
    scenario_metrics: list[ScenarioMetricReport] | None
    top_false_positive_users: list[UserMetricReport] | None
    top_false_negative_users: list[UserMetricReport] | None
    final_position: dict | None = None
    threshold_selection_status: str = "available"
    threshold_selection_reason: str | None = None


def add_predicted_labels(
    predictions: pd.DataFrame,
    threshold: float | None,
) -> pd.DataFrame:
    _validate_prediction_inputs(predictions, threshold)

    predictions = predictions.copy()
    if threshold is None:
        predictions["predicted_label"] = pd.Series(
            pd.NA,
            index=predictions.index,
            dtype="Int64",
        )
    else:
        predictions["predicted_label"] = (
            predictions["probability"].astype("float64") >= float(threshold)
        ).astype("int64")

    return predictions


def calculate_binary_classification_metrics(
    predictions: pd.DataFrame,
    threshold: float | None,
) -> BinaryClassificationMetrics:
    predictions = add_predicted_labels(predictions, threshold)
    y_true = predictions["label"].astype("int64").to_numpy()
    y_score = predictions["probability"].astype("float64").to_numpy()
    rows = int(len(predictions.index))
    positive_rows = int((y_true == 1).sum())
    negative_rows = int((y_true == 0).sum())
    has_both_classes = positive_rows > 0 and negative_rows > 0

    auroc = (
        float(roc_auc_score(y_true, y_score)) if has_both_classes else None
    )
    auprc = (
        float(average_precision_score(y_true, y_score))
        if positive_rows > 0
        else None
    )

    if threshold is None:
        predicted_positive_rows = None
        predicted_negative_rows = None
        precision = recall = f1 = None
        matrix = None
    else:
        y_pred = predictions["predicted_label"].astype("int64").to_numpy()
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
        if has_both_classes:
            precision_value, recall_value, f1_value, _ = (
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average="binary",
                    zero_division=0,
                )
            )
            precision = float(precision_value)
            recall = float(recall_value)
            f1 = float(f1_value)
        else:
            precision = (
                float(tp / (tp + fp)) if tp + fp > 0 else None
            )
            recall = float(tp / (tp + fn)) if tp + fn > 0 else None
            f1 = (
                float((2 * tp) / (2 * tp + fp + fn))
                if 2 * tp + fp + fn > 0
                else None
            )
        predicted_positive_rows = int((y_pred == 1).sum())
        predicted_negative_rows = int((y_pred == 0).sum())
        matrix = (
            None
            if rows == 0
            else ConfusionMatrixReport(
                true_negative=int(tn),
                false_positive=int(fp),
                false_negative=int(fn),
                true_positive=int(tp),
            )
        )

    unavailable_reasons: list[str] = []
    if rows == 0:
        unavailable_reasons.append("No prediction rows are available.")
    elif not has_both_classes:
        unavailable_reasons.append(
            "Only one label class is present, so AUROC is undefined."
        )
        if positive_rows == 0:
            unavailable_reasons.append(
                "No positive labels are present, so AUPRC is undefined."
            )
    if threshold is None:
        unavailable_reasons.append(
            "No threshold was provided, so threshold-dependent metrics are unavailable."
        )

    return BinaryClassificationMetrics(
        threshold=None if threshold is None else float(threshold),
        rows=rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        predicted_positive_rows=predicted_positive_rows,
        predicted_negative_rows=predicted_negative_rows,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        auprc=auprc,
        confusion_matrix=matrix,
        prevalence=(None if rows == 0 else positive_rows / rows),
        status=(
            "available"
            if not unavailable_reasons
            else "unavailable" if rows == 0 else "partial"
        ),
        reason=" ".join(unavailable_reasons) or None,
    )


def calculate_scenario_metrics(
    predictions: pd.DataFrame,
    threshold: float,
) -> list[ScenarioMetricReport]:
    _validate_prediction_inputs(predictions, threshold)

    if "scenario_id" not in predictions.columns:
        return []

    predictions = add_predicted_labels(predictions, threshold)
    _validate_scenario_metadata(predictions)
    scenario_values = (
        predictions.loc[predictions["scenario_id"].notna(), "scenario_id"]
        .astype("int64")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    scenario_metrics: list[ScenarioMetricReport] = []

    for scenario_id in scenario_values:
        scenario_predictions = predictions[
            (
                (predictions["label"].eq(1)) &
                (predictions["scenario_id"].eq(scenario_id))
            )
            | predictions["label"].eq(0)
        ]
        y_true = (
            scenario_predictions["label"].eq(1)
            & scenario_predictions["scenario_id"].eq(scenario_id)
        ).astype("int64")
        y_pred = scenario_predictions["predicted_label"].astype("int64")
        true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
        false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
        false_positive_mask = (y_true == 0) & (y_pred == 1)
        false_positives_on_negative_rows = int(false_positive_mask.sum())
        false_positives_normal_users = int(
            (
                false_positive_mask &
                scenario_predictions["is_malicious_user"].eq(False)
            ).sum()
        )
        false_positives_malicious_users_non_attack = int(
            (
                false_positive_mask &
                scenario_predictions["is_malicious_user"].eq(True)
            ).sum()
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.to_numpy(),
            y_pred.to_numpy(),
            average="binary",
            zero_division=0,
        )

        scenario_metrics.append(
            ScenarioMetricReport(
                scenario_id=int(scenario_id),
                evaluation_definition=(
                    "scenario_id positives versus all negative rows; negative "
                    "rows include normal users and non-attack rows from "
                    "malicious users. Positives from other scenarios are "
                    "ignored for this scenario metric."
                ),
                rows=int(len(scenario_predictions.index)),
                positive_count=int(y_true.eq(1).sum()),
                negative_count=int(y_true.eq(0).sum()),
                true_positives=true_positives,
                false_negatives=false_negatives,
                false_positives_on_negative_rows=false_positives_on_negative_rows,
                false_positives_normal_users=false_positives_normal_users,
                false_positives_malicious_users_non_attack=(
                    false_positives_malicious_users_non_attack
                ),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
            )
        )

    return scenario_metrics


def calculate_false_positive_breakdown(
    predictions: pd.DataFrame,
) -> FalsePositiveBreakdownReport:
    _validate_prediction_columns(
        predictions,
        {"label", "predicted_label", "is_malicious_user"},
    )

    false_positive_mask = (
        predictions["label"].eq(0) &
        predictions["predicted_label"].eq(1)
    )

    return FalsePositiveBreakdownReport(
        total_false_positives=int(false_positive_mask.sum()),
        normal_user_false_positives=int(
            (
                false_positive_mask &
                predictions["is_malicious_user"].eq(False)
            ).sum()
        ),
        malicious_user_non_attack_false_positives=int(
            (
                false_positive_mask &
                predictions["is_malicious_user"].eq(True)
            ).sum()
        ),
    )


def calculate_fixed_threshold_metrics(
    predictions: pd.DataFrame,
    thresholds: list[float],
) -> list[FixedThresholdMetricsReport]:
    _validate_fixed_thresholds(thresholds)

    reports: list[FixedThresholdMetricsReport] = []
    for threshold in sorted(thresholds):
        threshold_predictions = add_predicted_labels(predictions, threshold)
        reports.append(
            FixedThresholdMetricsReport(
                threshold=float(threshold),
                metrics=calculate_binary_classification_metrics(
                    threshold_predictions,
                    threshold,
                ),
                false_positive_breakdown=calculate_false_positive_breakdown(
                    threshold_predictions
                ),
                scenario_metrics=calculate_scenario_metrics(
                    threshold_predictions,
                    threshold,
                ),
            )
        )

    return reports


def calculate_user_metrics(
    predictions: pd.DataFrame,
    threshold: float,
) -> list[UserMetricReport]:
    _validate_prediction_inputs(predictions, threshold)
    _validate_prediction_columns(
        predictions,
        {"user", "is_malicious_user", "scenario_id"},
    )

    predictions = add_predicted_labels(predictions, threshold)
    user_reports: list[UserMetricReport] = []

    for user, user_predictions in predictions.groupby("user", sort=True):
        y_true = user_predictions["label"].astype("int64")
        y_pred = user_predictions["predicted_label"].astype("int64")
        true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
        false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
        false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
        true_negatives = int(((y_true == 0) & (y_pred == 0)).sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.to_numpy(),
            y_pred.to_numpy(),
            average="binary",
            zero_division=0,
        )
        scenario_values = (
            user_predictions["scenario_id"]
            .dropna()
            .astype("int64")
            .drop_duplicates()
            .tolist()
        )

        user_reports.append(
            UserMetricReport(
                user=str(user),
                is_malicious_user=bool(
                    user_predictions["is_malicious_user"].iloc[0]
                ),
                scenario_id=(
                    None
                    if not scenario_values
                    else int(sorted(scenario_values)[0])
                ),
                rows=int(len(user_predictions.index)),
                positive_count=int(y_true.eq(1).sum()),
                negative_count=int(y_true.eq(0).sum()),
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                true_negatives=true_negatives,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
            )
        )

    return user_reports


def save_test_evaluation_report(
    report: TestEvaluationReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output_path)


def _validate_prediction_inputs(
    predictions: pd.DataFrame,
    threshold: float | None,
) -> None:
    required_columns = {"probability", "label"}
    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if threshold is not None and (
        not isinstance(threshold, (int, float, np.number))
        or not np.isfinite(threshold)
        or threshold < 0
        or threshold > 1
    ):
        raise ValueError(
            "threshold must be a finite value between 0 and 1, or None. "
            f"Current value: {threshold}."
        )

    try:
        probabilities = pd.to_numeric(predictions["probability"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Predictions contain invalid probability values.") from error
    if (
        probabilities.isna().any()
        or not np.isfinite(probabilities.astype("float64")).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be finite values between 0 and 1."
        )

    try:
        labels = pd.to_numeric(predictions["label"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Labels must contain only 0 and 1.") from error
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            "Labels must contain only exact binary values 0 and 1."
        )


def _validate_prediction_columns(
    predictions: pd.DataFrame,
    required_columns: set[str],
) -> None:
    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            f"{sorted(missing_columns)}."
        )


def _validate_fixed_thresholds(thresholds: list[float]) -> None:
    if not thresholds:
        raise ValueError("Fixed threshold list cannot be empty.")

    if any(
        not isinstance(threshold, (int, float, np.number))
        or not np.isfinite(threshold)
        or threshold < 0
        or threshold > 1
        for threshold in thresholds
    ):
        raise ValueError("Fixed thresholds must be finite values between 0 and 1.")

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Fixed thresholds cannot contain duplicate values.")


def _validate_scenario_metadata(predictions: pd.DataFrame) -> None:
    _validate_prediction_columns(
        predictions,
        {"scenario_id", "is_malicious_user"},
    )
    malicious_user_missing_scenario = (
        predictions["is_malicious_user"].eq(True) &
        predictions["scenario_id"].isna()
    )
    normal_user_with_scenario = (
        predictions["is_malicious_user"].eq(False) &
        predictions["scenario_id"].notna()
    )

    if malicious_user_missing_scenario.any():
        raise ValueError(
            "Scenario metrics require scenario_id for every malicious user row."
        )

    if normal_user_with_scenario.any():
        raise ValueError(
            "Scenario metrics require scenario_id to be null for normal user rows."
        )
