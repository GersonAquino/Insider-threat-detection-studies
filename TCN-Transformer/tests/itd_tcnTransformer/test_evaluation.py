"""Synthetic, CPU-only coverage for the evaluation protocol."""

import importlib
import json
from unittest.mock import patch

import pandas as pd
import pytest
import torch


@pytest.fixture(scope="module")
def evaluation_modules():
    """Import settings once without requiring a CUDA-capable test host."""
    with patch.object(torch.cuda, "is_available", return_value=True):
        thresholds = importlib.import_module("src.itd_tcnTransformer.evaluation.thresholds")
        metrics = importlib.import_module("src.itd_tcnTransformer.evaluation.metrics")
        evaluate = importlib.import_module("src.itd_tcnTransformer.evaluation.evaluate")
    return thresholds, metrics, evaluate


def _predictions(labels, probabilities):
    return pd.DataFrame({"label": labels, "probability": probabilities})


def _window_predictions(window_size=3, final_labels=(0, 0)):
    rows = []
    for window_index, final_label in enumerate(final_labels):
        end_day = pd.Timestamp("2025-01-03") + pd.Timedelta(days=window_index)
        for position in range(window_size):
            rows.append({
                "window_index": window_index,
                "temporal_position": position,
                "day": end_day - pd.Timedelta(days=window_size - 1 - position),
                "window_end_day": end_day,
                "label": final_label if position == window_size - 1 else position % 2,
                "probability": 0.8 if position == window_size - 1 else 0.2 + position / (10 * window_size),
                "user": f"U{window_index}",
            })
    return pd.DataFrame(rows)


def test_threshold_report_round_trip_and_legacy_loading(evaluation_modules, tmp_path):
    thresholds, _, _ = evaluation_modules
    report = thresholds.select_threshold(
        _predictions([0, 1], [0.1, 0.9]),
        threshold_selection_metric="f1",
        threshold_step=0.5,
        prediction_scope="final_position",
        temporal_position=29,
        window_size=30,
    )
    path = tmp_path / "threshold.json"
    thresholds.save_threshold_selection_report(report, path)
    loaded = thresholds.load_threshold_selection_report(path)

    assert loaded.selected_threshold == report.selected_threshold
    assert loaded.prediction_scope == "final_position"
    assert loaded.temporal_position == 29
    assert loaded.window_size == 30

    legacy = json.loads(path.read_text(encoding="utf-8"))
    for key in ("status", "reason", "prediction_scope", "temporal_position", "window_size"):
        legacy.pop(key, None)
    legacy_path = tmp_path / "legacy_threshold.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    legacy_loaded = thresholds.load_threshold_selection_report(legacy_path)

    assert legacy_loaded.status == "available"
    assert legacy_loaded.reason is None
    assert legacy_loaded.prediction_scope == "all_positions"
    assert legacy_loaded.temporal_position is None
    assert legacy_loaded.window_size is None


@pytest.mark.parametrize(
    ("predictions", "kwargs", "reason_fragment"),
    [
        (_predictions([], []), {}, "empty"),
        (_predictions([0, 0], [0.2, 0.8]), {}, "single"),
        (
            _predictions([0, 1], [0.8, 0.2]),
            {"threshold_selection_metric": "recall_at_min_precision", "min_precision": 1.0},
            "No threshold candidate",
        ),
    ],
)
def test_threshold_selection_can_report_unavailable(evaluation_modules, predictions, kwargs, reason_fragment, tmp_path):
    thresholds, _, _ = evaluation_modules
    report = thresholds.select_threshold(
        predictions,
        allow_unavailable=True,
        threshold_step=0.5,
        **kwargs,
    )

    assert report.status == "unavailable"
    assert report.selected_threshold is None
    assert report.selected_candidate is None
    assert reason_fragment.lower() in report.reason.lower()
    path = tmp_path / "unavailable.json"
    thresholds.save_threshold_selection_report(report, path)
    assert thresholds.load_threshold_selection_report(path) == report


def test_threshold_selection_is_strict_by_default(evaluation_modules):
    thresholds, _, _ = evaluation_modules
    with pytest.raises(ValueError, match="empty"):
        thresholds.select_threshold(_predictions([], []))
    with pytest.raises(ValueError, match="minimum precision"):
        thresholds.select_threshold(_predictions([0, 1], [0.8, 0.2]), threshold_step=0.5)


def test_binary_metrics_known_counts_and_thresholdless_ranking(evaluation_modules):
    _, metrics, _ = evaluation_modules
    predictions = _predictions([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])

    result = metrics.calculate_binary_classification_metrics(predictions, 0.5)
    assert result.confusion_matrix.true_negative == 1
    assert result.confusion_matrix.false_positive == 1
    assert result.confusion_matrix.false_negative == 1
    assert result.confusion_matrix.true_positive == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.auroc == pytest.approx(0.75)
    assert result.auprc == pytest.approx(5 / 6)
    assert result.prevalence == pytest.approx(0.5)

    ranking_only = metrics.calculate_binary_classification_metrics(predictions, None)
    assert ranking_only.threshold is None
    assert ranking_only.predicted_positive_rows is None
    assert ranking_only.precision is None
    assert ranking_only.confusion_matrix is None
    assert ranking_only.auroc == pytest.approx(0.75)


def test_binary_metrics_support_empty_and_single_class_inputs(evaluation_modules):
    _, metrics, _ = evaluation_modules
    empty = metrics.calculate_binary_classification_metrics(_predictions([], []), None)
    assert empty.rows == 0
    assert empty.prevalence is None
    assert empty.auroc is None
    assert empty.auprc is None
    assert empty.confusion_matrix is None

    negatives = metrics.calculate_binary_classification_metrics(
        _predictions([0, 0], [0.1, 0.8]), 0.5,
    )
    assert negatives.auroc is None
    assert negatives.auprc is None
    assert negatives.confusion_matrix.false_positive == 1
    assert negatives.precision == 0.0
    assert negatives.recall is None
    assert negatives.f1 == 0.0

    positives = metrics.calculate_binary_classification_metrics(
        _predictions([1, 1], [0.1, 0.8]), 0.5,
    )
    assert positives.auroc is None
    assert positives.auprc == pytest.approx(1.0)
    assert positives.precision == pytest.approx(1.0)
    assert positives.recall == pytest.approx(0.5)
    assert positives.f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "predictions",
    [
        _predictions([0, 2], [0.2, 0.8]),
        _predictions([0, 0.5], [0.2, 0.8]),
        _predictions([0, 1], [-0.1, 0.8]),
        _predictions([0, 1], [0.2, float("nan")]),
    ],
)
def test_binary_metrics_reject_invalid_data(evaluation_modules, predictions):
    thresholds, metrics, _ = evaluation_modules
    with pytest.raises(ValueError):
        metrics.calculate_binary_classification_metrics(predictions, 0.5)
    with pytest.raises(ValueError):
        thresholds.select_threshold(predictions, allow_unavailable=True)


@pytest.mark.parametrize("window_size", [3, 30])
def test_final_position_filter_uses_dynamic_window_size(evaluation_modules, window_size):
    _, _, evaluate = evaluation_modules
    selected = evaluate.select_final_position_predictions(
        _window_predictions(window_size=window_size, final_labels=(0, 1)), window_size,
    )

    assert selected["temporal_position"].tolist() == [window_size - 1] * 2
    assert selected["window_index"].tolist() == [0, 1]
    assert selected["day"].tolist() == selected["window_end_day"].tolist()


@pytest.mark.parametrize("availability", ["both", "global_only", "final_only", "neither"])
def test_run_evaluation_preserves_global_and_final_position_reports(
    evaluation_modules, monkeypatch, tmp_path, availability,
):
    _, metrics, evaluate = evaluation_modules
    global_available = availability in {"both", "global_only"}
    final_available = availability in {"both", "final_only"}
    validation_window = _window_predictions(final_labels=(0, 1))
    final_mask = validation_window["temporal_position"].eq(2)
    validation_window.loc[final_mask, "probability"] = (
        [0.1, 0.6] if final_available else [0.8, 0.2]
    )
    test_window = _window_predictions(final_labels=(0, 1))
    validation_scored = _predictions(
        [0, 1], [0.1, 0.9] if global_available else [0.9, 0.1],
    )
    test_scored = _predictions([0, 1], [0.1, 0.9])
    test_scored["user"] = ["U0", "U1"]
    test_scored["day"] = pd.to_datetime(["2025-01-03", "2025-01-04"])
    calls = []
    saved_prediction_paths = []
    scenario_metadata_paths = []
    sentinel_model = object()
    checkpoint = {
        "model_hyperparameters": {"predict_all_days": True, "window_size": 3},
    }

    monkeypatch.setattr(evaluate, "_load_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(evaluate, "_validate_checkpoint_loss_hyperparameters", lambda *args: None)
    monkeypatch.setattr(evaluate, "_validate_checkpoint_evaluation_configuration", lambda *args: None)
    monkeypatch.setattr(evaluate, "_reconstruct_model_from_checkpoint", lambda *args: sentinel_model)
    monkeypatch.setattr(evaluate, "_resolve_validation_split_name", lambda: "validation")
    monkeypatch.setattr(evaluate, "THRESHOLD_SELECTION_STEP", 0.1)

    def fake_evaluate_split_predictions(**kwargs):
        assert kwargs["model"] is sentinel_model
        calls.append(kwargs["split_name"])
        if kwargs["split_name"] == "validation":
            return validation_window.copy(), validation_scored.copy()
        assert kwargs["split_name"] == "test"
        return test_window.copy(), test_scored.copy()

    monkeypatch.setattr(evaluate, "_evaluate_split_predictions", fake_evaluate_split_predictions)
    monkeypatch.setattr(evaluate, "_save_predictions", lambda _, path: saved_prediction_paths.append(path))
    def fake_add_scenario_metadata(frame, daily_labels_dir):
        scenario_metadata_paths.append(daily_labels_dir)
        return frame.assign(scenario_id=pd.NA, is_malicious_user=False)

    monkeypatch.setattr(evaluate, "_add_scenario_metadata", fake_add_scenario_metadata)
    archived_daily_labels_dir = tmp_path / "archived" / "daily_labels"

    result = evaluate.run_evaluation(
        checkpoint_path=tmp_path / "checkpoint.pt",
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "predictions",
        reports_dir=tmp_path / "reports",
        sequences_dir=tmp_path / "sequences",
        daily_labels_dir=archived_daily_labels_dir,
        device="cpu",
    )

    assert calls == ["validation", "test"]
    assert scenario_metadata_paths == [archived_daily_labels_dir]
    assert len(saved_prediction_paths) == 4
    assert all(path.parent == tmp_path / "predictions" for path in saved_prediction_paths)
    assert (tmp_path / "metrics" / evaluate.VALIDATION_THRESHOLD_FILE).is_file()
    assert (tmp_path / "metrics" / evaluate.VALIDATION_FINAL_POSITION_THRESHOLD_FILE).is_file()
    report_data = json.loads((tmp_path / "reports" / evaluate.TEST_EVALUATION_REPORT_FILE).read_text())
    assert result.test_report_path == str(tmp_path / "reports" / evaluate.TEST_EVALUATION_REPORT_FILE)
    if global_available:
        assert report_data["metrics"]["confusion_matrix"] == {
            "true_negative": 1, "false_positive": 0, "false_negative": 0, "true_positive": 1,
        }
    else:
        assert report_data["metrics"]["confusion_matrix"] is None
        assert report_data["scenario_metrics"] is None
        assert report_data["top_false_positive_users"] is None
        assert report_data["false_positive_breakdown"] is None
    assert report_data["metrics"]["auprc"] == 1.0
    assert report_data["metrics"]["prevalence"] == 0.5
    assert report_data["fixed_threshold_metrics"]
    final = report_data["final_position"]
    for key, available, expected_threshold in (
        ("global_threshold", global_available, 0.9),
        ("final_position_threshold", final_available, 0.6),
    ):
        assert final[key]["threshold_selection_status"] == ("available" if available else "unavailable")
        subset_metrics = final[key]["metrics"]
        assert subset_metrics["rows"] == 2
        assert subset_metrics["prevalence"] == 0.5
        assert subset_metrics["auroc"] == 0.5
        assert subset_metrics["auprc"] == 0.5
        if available:
            assert subset_metrics["threshold"] == pytest.approx(expected_threshold)
        else:
            assert subset_metrics["threshold"] is None
            assert subset_metrics["precision"] is None
            assert subset_metrics["confusion_matrix"] is None
        expected = metrics.calculate_binary_classification_metrics(
            test_window[test_window["temporal_position"].eq(2)],
            expected_threshold if available else None,
        )
        assert subset_metrics["precision"] == expected.precision
    final_selection = json.loads((tmp_path / "metrics" / evaluate.VALIDATION_FINAL_POSITION_THRESHOLD_FILE).read_text())
    assert final_selection["validation_rows"] == 2
    assert final_selection["min_precision"] == 0.7
    assert final_selection["temporal_position"] == 2
