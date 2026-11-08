from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    read_jsonl,
    resolve_path,
    sanitize_for_prompt,
    stable_int,
    volume_class,
    write_json,
)


OLLAMA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "prediction",
        "confidence",
        "justification",
    ],
    "additionalProperties": False,
    "properties": {
        "prediction": {
            "type": "string",
            "enum": ["malicious", "benign"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "justification": {
            "type": "string",
        },
    },
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout or completed.stderr).strip()


def git_metadata(project_root: Path) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"], cwd=project_root)
    status = command_output(["git", "status", "--porcelain"], cwd=project_root)
    return {
        "commit": commit,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected an object from {url}")
    return parsed


def ollama_runtime_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    ollama_cfg = cfg["ollama"]
    base_url = str(ollama_cfg["base_url"]).rstrip("/")
    model_name = str(ollama_cfg["model"])
    result: dict[str, Any] = {"model": model_name, "base_url": base_url}
    errors: list[str] = []

    try:
        version = http_json(base_url + "/api/version")
        result["version"] = version.get("version")
    except Exception as exc:
        errors.append(f"version: {exc}")

    try:
        tags = http_json(base_url + "/api/tags")
        models = tags.get("models") if isinstance(tags.get("models"), list) else []
        selected = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and str(item.get("name") or item.get("model") or "") == model_name
            ),
            None,
        )
        if selected:
            result["digest"] = selected.get("digest")
            result["modified_at"] = selected.get("modified_at")
            result["size_bytes"] = selected.get("size")
            result["details"] = selected.get("details")
    except Exception as exc:
        errors.append(f"tags: {exc}")

    try:
        shown = http_json(base_url + "/api/show", {"model": model_name})
        result["capabilities"] = shown.get("capabilities")
        result["model_details"] = shown.get("details")
        result["model_info_sha256"] = sha256_text(canonical_json(shown.get("model_info") or {}))
        result["template_sha256"] = sha256_text(str(shown.get("template") or ""))
        result["parameters_sha256"] = sha256_text(str(shown.get("parameters") or ""))
    except Exception as exc:
        errors.append(f"show: {exc}")

    if errors:
        result["metadata_errors"] = errors
    return result


def environment_metadata() -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for package_name in ("duckdb", "PyYAML"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "package_versions": package_versions,
    }


def source_counts(sample: dict[str, Any]) -> dict[str, int]:
    counts = sample.get("metadata", {}).get("event_counts", {}).get("by_source")
    if isinstance(counts, dict):
        return {str(key): int(value or 0) for key, value in counts.items()}
    return dict(Counter(str(event.get("source") or "") for event in sample.get("events", [])))


def event_flags(sample: dict[str, Any]) -> dict[str, bool]:
    counts = sample.get("metadata", {}).get("event_counts", {})
    if isinstance(counts, dict) and "has_device" in counts:
        return {
            "has_device": bool(counts.get("has_device")),
            "has_file": bool(counts.get("has_file")),
            "has_ldap": bool(counts.get("has_ldap")),
            "has_psychometric": bool(counts.get("has_psychometric")),
            "has_profile": bool(counts.get("has_profile")),
            "has_external_email": bool(counts.get("has_external_email")),
            "has_external_http": bool(counts.get("has_external_http")),
            "has_off_hours": bool(counts.get("has_off_hours")),
            "has_off_hours_logon": bool(counts.get("has_off_hours_logon")),
            "has_suspicious_http": bool(counts.get("has_suspicious_http")),
        }
    events = sample.get("events", [])
    return {
        "has_device": any(event.get("source") == "device" for event in events),
        "has_file": any(event.get("source") == "file" for event in events),
        "has_ldap": any(event.get("source") == "ldap" for event in events),
        "has_psychometric": any(event.get("source") == "psychometric" for event in events),
        "has_profile": any(event.get("is_profile_record") for event in events),
        "has_external_email": any(event.get("external_email") for event in events),
        "has_external_http": any(event.get("external_http") for event in events),
        "has_off_hours": any(event.get("off_hours") for event in events),
        "has_off_hours_logon": any(event.get("source") == "logon" and event.get("off_hours") for event in events),
        "has_suspicious_http": any(event.get("suspicious_http_keyword") for event in events),
    }


def total_events(sample: dict[str, Any]) -> int:
    counts = sample.get("metadata", {}).get("event_counts", {})
    if isinstance(counts, dict) and "total" in counts:
        return int(counts.get("total") or 0)
    return len(sample.get("events", []))


def sample_volume_bin(sample: dict[str, Any], bins: list[int]) -> str:
    counts = sample.get("metadata", {}).get("event_counts", {})
    if isinstance(counts, dict) and counts.get("volume_bin"):
        return str(counts["volume_bin"])
    return volume_class(total_events(sample), bins)


def similarity_score(train_sample: dict[str, Any], test_sample: dict[str, Any], cfg: dict[str, Any]) -> float:
    bins = [int(value) for value in cfg["sampling"].get("volume_bins", [20, 100, 300])]
    train_counts = source_counts(train_sample)
    test_counts = source_counts(test_sample)
    train_types = {key for key, value in train_counts.items() if value > 0}
    test_types = {key for key, value in test_counts.items() if value > 0}
    union = train_types | test_types
    jaccard = (len(train_types & test_types) / len(union)) if union else 0.0

    score = 4.0 * jaccard
    if sample_volume_bin(train_sample, bins) == sample_volume_bin(test_sample, bins):
        score += 3.0

    train_flags = event_flags(train_sample)
    test_flags = event_flags(test_sample)
    for key in sorted(test_flags):
        if train_flags.get(key) == test_flags.get(key):
            score += 0.8

    count_delta = abs(total_events(train_sample) - total_events(test_sample))
    score -= min(2.0, count_delta / 150.0)

    if train_sample.get("user") != test_sample.get("user"):
        score += 1.0
    else:
        score -= 5.0
    return score


def choose_examples(train_samples: list[dict[str, Any]], test_sample: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    llm_cfg = cfg["llm"]
    k = int(llm_cfg.get("k_examples", 5))
    prefer_different = bool(llm_cfg.get("prefer_different_users", True))
    pool = train_samples
    if prefer_different:
        different_user_pool = [sample for sample in train_samples if sample.get("user") != test_sample.get("user")]
        if len(different_user_pool) >= k:
            pool = different_user_pool

    ranked = [
        {
            "sample": sample,
            "similarity": similarity_score(sample, test_sample, cfg),
            "hash": stable_int(str(sample.get("case_id")), int(cfg["sampling"]["random_seed"])),
        }
        for sample in pool
    ]
    ranked.sort(key=lambda item: (-item["similarity"], item["hash"]))

    if not bool(llm_cfg.get("balance_example_labels", True)):
        return [item["sample"] | {"_similarity": item["similarity"]} for item in ranked[:k]]

    positives = [item for item in ranked if item["sample"].get("label") == "malicious"]
    negatives = [item for item in ranked if item["sample"].get("label") == "benign"]
    wanted_pos = min(len(positives), max(1, k // 2)) if positives else 0
    wanted_neg = min(len(negatives), k - wanted_pos)
    selected = positives[:wanted_pos] + negatives[:wanted_neg]
    selected_ids = {item["sample"]["case_id"] for item in selected}
    for item in ranked:
        if len(selected) >= k:
            break
        if item["sample"]["case_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["sample"]["case_id"])
    selected.sort(key=lambda item: (-item["similarity"], item["hash"]))
    return [item["sample"] | {"_similarity": item["similarity"]} for item in selected[:k]]


def compact_case_for_prompt(sample: dict[str, Any], *, keep_label: bool, max_events: int) -> dict[str, Any]:
    clean = sanitize_for_prompt(sample, keep_label=keep_label)
    if "metadata" in clean:
        clean["metadata"] = {
            "event_counts": clean["metadata"].get("event_counts", {}),
            "events_in_payload": clean["metadata"].get("events_in_payload"),
            "truncation": clean["metadata"].get("truncation", {}),
        }
    events = clean.get("events", [])
    if len(events) > max_events:
        clean["events"] = events[:max_events]
        clean.setdefault("metadata", {})["prompt_event_limit"] = {
            "original_payload_events": len(events),
            "events_sent": max_events,
        }
    return clean


def build_prompt(
    test_sample: dict[str, Any],
    examples: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> str:
    llm_cfg = cfg["llm"]
    max_example_events = int(llm_cfg.get("max_events_per_example", 120))
    max_test_events = int(llm_cfg.get("max_events_for_test", 180))
    example_payloads = [
        compact_case_for_prompt(example, keep_label=True, max_events=max_example_events)
        for example in examples
    ]
    for payload, example in zip(example_payloads, examples):
        payload["similarity_to_test"] = round(float(example.get("_similarity") or 0.0), 4)

    test_payload = compact_case_for_prompt(test_sample, keep_label=False, max_events=max_test_events)
    prompt_style = str(llm_cfg.get("prompt_style") or "schema_json").strip().lower()
    if prompt_style in {"strict_json", "json_only", "gpt_oss", "gpt-oss"}:
        response_instructions = (
            "This is an authorized synthetic CERT insider-threat benchmark. "
            "You are only classifying whether the user-day is malicious or benign.\n\n"
            "Use private reasoning if available. The final response field must contain ONLY one JSON object.\n\n"
            "The training examples may contain a \"label\" field. That field is input data only. "
            "Do not use \"label\" in your output.\n\n"
            "After reading the examples and the unlabeled test case, return exactly one JSON object "
            "with exactly these keys:\n"
            "{\"prediction\":\"benign\",\"confidence\":0.0,\"justification\":\"short reason\"}\n\n"
            "Output rules:\n"
            "- the first character of your final response must be {\n"
            "- the last character of your final response must be }\n"
            "- never return an array\n"
            "- never return markdown\n"
            "- never return copied input data\n"
            "- never return keys named label, case_id, metadata, event_counts, sources, flags, or events\n"
            "- prediction must be either \"malicious\" or \"benign\"\n"
            "- confidence must be a number from 0.0 to 1.0\n"
            "- justification must be one short sentence\n"
            "- if unsure, still choose malicious or benign and use a lower confidence\n\n"
        )
    else:
        response_instructions = (
            "Return exactly one JSON object with this schema:\n"
            "{"
            "\"prediction\":\"malicious|benign\","
            "\"confidence\":0.0,"
            "\"justification\":\"short reason\""
            "}\n"
            "Do not include any additional properties. Keep justification concise.\n\n"
        )
    return (
        "You are classifying CERT insider-threat user-day cases.\n"
        "A case is malicious only if at least one malicious action occurred for that user on the target date. "
        "Do not mark neighboring days as malicious just because the same user is suspicious. "
        "Use only the events and previous-days summary shown in the prompt. Do not assume future events.\n\n"
        f"{response_instructions}"
        "Classified examples from the training set:\n"
        f"{json.dumps(example_payloads, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Unlabeled test case to classify:\n"
        f"{json.dumps(test_payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "Now classify the unlabeled test case.\n\n"
        "Return exactly this JSON shape and nothing else:\n"
        "{\"prediction\":\"benign\",\"confidence\":0.0,\"justification\":\"short reason\"}\n\n"
        "Remember: do not output an array, do not output \"label\", and do not copy metadata or events.\n"
    )


def resolve_ollama_format(value: Any) -> Any:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "false"}:
            return None
        if token in {"json", "json_schema", "schema", "response_schema"}:
            return OLLAMA_RESPONSE_SCHEMA
    return value


def ollama_generate(prompt: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ollama_cfg = cfg["ollama"]
    base_url = str(ollama_cfg["base_url"]).rstrip("/")
    options = {
        "temperature": float(ollama_cfg.get("temperature", 0.0)),
        "num_ctx": int(ollama_cfg.get("num_ctx", 16384)),
    }
    payload: dict[str, Any] = {
        "model": ollama_cfg["model"],
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    response_format = resolve_ollama_format(ollama_cfg.get("format"))
    if response_format:
        payload["format"] = response_format
    if ollama_cfg.get("think") is not None:
        think_value = ollama_cfg.get("think")
        if isinstance(think_value, str):
            cleaned_think = think_value.strip()
            if cleaned_think.lower() not in {"", "none", "null"}:
                payload["think"] = cleaned_think
        else:
            payload["think"] = bool(think_value)

    request = urllib.request.Request(
        base_url + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = int(ollama_cfg.get("timeout_seconds", 300))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response") or ""), data


def extract_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("The final response must be one JSON object.")
    return payload


def validate_response_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"prediction", "confidence", "justification"}
    keys = set(payload)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if extra:
        errors.append("unexpected fields: " + ", ".join(extra))

    if payload.get("prediction") not in {"malicious", "benign"}:
        errors.append("prediction must be 'malicious' or 'benign'")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be a number")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence must be between 0.0 and 1.0")

    justification = payload.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        errors.append("justification must be a non-empty string")
    return errors


def normalize_prediction(payload: dict[str, Any]) -> str:
    raw = str(payload.get("prediction") or payload.get("label") or "").strip().lower()
    if raw in {"malicious", "positive", "insider", "true", "1"}:
        return "malicious"
    if raw in {"benign", "negative", "normal", "false", "0"}:
        return "benign"
    return "unknown"


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean_defined(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("ground_truth") in {"malicious", "benign"}]
    tp = sum(row["ground_truth"] == "malicious" and row["prediction"] == "malicious" for row in evaluated)
    tn = sum(row["ground_truth"] == "benign" and row["prediction"] == "benign" for row in evaluated)
    fp = sum(row["ground_truth"] == "benign" and row["prediction"] == "malicious" for row in evaluated)
    fn = sum(row["ground_truth"] == "malicious" and row["prediction"] == "benign" for row in evaluated)
    unknown_malicious = sum(
        row["ground_truth"] == "malicious" and row["prediction"] not in {"malicious", "benign"}
        for row in evaluated
    )
    unknown_benign = sum(
        row["ground_truth"] == "benign" and row["prediction"] not in {"malicious", "benign"}
        for row in evaluated
    )
    unknown = unknown_malicious + unknown_benign
    valid_count = len(evaluated) - unknown
    malicious_support = tp + fn + unknown_malicious
    benign_support = tn + fp + unknown_benign

    precision = safe_divide(tp, tp + fp)
    end_to_end_recall = safe_divide(tp, malicious_support)
    end_to_end_specificity = safe_divide(tn, benign_support)
    end_to_end_accuracy = safe_divide(tp + tn, len(evaluated))
    end_to_end_f1 = f1_score(precision, end_to_end_recall)

    valid_recall = safe_divide(tp, tp + fn)
    valid_specificity = safe_divide(tn, tn + fp)
    valid_accuracy = safe_divide(tp + tn, valid_count)
    valid_f1 = f1_score(precision, valid_recall)
    return {
        "num_cases": len(evaluated),
        "num_valid_predictions": valid_count,
        "num_invalid_predictions": unknown,
        "coverage": safe_divide(valid_count, len(evaluated)),
        "invalid_rate": safe_divide(unknown, len(evaluated)),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn_valid_prediction": fn,
            "unknown_malicious": unknown_malicious,
            "unknown_benign": unknown_benign,
            "unknown_total": unknown,
        },
        "accuracy": end_to_end_accuracy,
        "precision": precision,
        "recall": end_to_end_recall,
        "specificity": end_to_end_specificity,
        "balanced_accuracy": mean_defined(end_to_end_recall, end_to_end_specificity),
        "f1": end_to_end_f1,
        "valid_predictions_only": {
            "num_cases": valid_count,
            "accuracy": valid_accuracy,
            "precision": precision,
            "recall": valid_recall,
            "specificity": valid_specificity,
            "balanced_accuracy": mean_defined(valid_recall, valid_specificity),
            "f1": valid_f1,
        },
        "prediction_counts": dict(Counter(row["prediction"] for row in evaluated)),
        "label_counts": dict(Counter(row["ground_truth"] for row in evaluated)),
    }


def load_eval_inputs(cfg: dict[str, Any], variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], Path]:
    base = resolve_path(cfg, cfg["dataset"]["output_dir"]) / variant
    train_full = base / "train_full.jsonl"
    test_full = base / "test_full.jsonl"
    test_prompt = base / "test_prompt.jsonl"
    missing = [str(path) for path in (train_full, test_full, test_prompt) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing subset files. Run build_subset.py first: " + ", ".join(missing))
    train_samples = read_jsonl(train_full)
    test_full_rows = read_jsonl(test_full)
    test_prompt_rows = read_jsonl(test_prompt)
    truth_by_case = {row["case_id"]: row for row in test_full_rows}
    return train_samples, test_prompt_rows, truth_by_case, base


def run(config_path: str | None = None, *, variant: str | None = None, overwrite: bool = False, limit: int | None = None) -> dict[str, Any]:
    from common import load_config

    cfg = load_config(config_path)
    variant = variant or str(cfg["llm"].get("variant") or "day_only")
    train_samples, test_samples, truth_by_case, base = load_eval_inputs(cfg, variant)
    predictions_path = base / str(cfg["llm"].get("predictions_file", "llm_predictions.jsonl"))
    report_path = base / str(cfg["llm"].get("evaluation_report_file", "llm_evaluation_report.json"))
    input_paths = {
        "train_full": base / "train_full.jsonl",
        "test_full": base / "test_full.jsonl",
        "test_prompt": base / "test_prompt.jsonl",
    }
    input_file_sha256 = {name: sha256_file(path) for name, path in input_paths.items()}
    ollama_runtime = ollama_runtime_metadata(cfg)

    evaluator_path = Path(__file__).resolve()
    config_source_path = Path(config_path).resolve() if config_path else None
    evaluator_sha256 = sha256_file(evaluator_path)
    prompt_builder_sha256 = sha256_text(inspect.getsource(build_prompt))
    resolved_config_sha256 = sha256_text(canonical_json(cfg))
    run_config_sha256 = sha256_text(
        canonical_json(
            {
                "resolved_config": cfg,
                "variant": variant,
                "evaluator_sha256": evaluator_sha256,
                "prompt_builder_sha256": prompt_builder_sha256,
                "input_file_sha256": input_file_sha256,
                "ollama_model_identity": {
                    key: ollama_runtime.get(key)
                    for key in (
                        "model",
                        "digest",
                        "version",
                        "model_info_sha256",
                        "template_sha256",
                        "parameters_sha256",
                    )
                },
            }
        )
    )
    execution_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]

    existing: list[dict[str, Any]] = []
    seen: set[str] = set()
    if predictions_path.exists() and not overwrite:
        existing = read_jsonl(predictions_path)
        existing_hashes = {row.get("run_config_sha256") for row in existing}
        if existing_hashes != {run_config_sha256}:
            raise ValueError(
                "Existing predictions were produced without this exact run configuration. "
                "Use --overwrite or choose a new predictions_file; do not mix runs."
            )
        seen = {str(row.get("case_id")) for row in existing}
    elif overwrite and predictions_path.exists():
        predictions_path.unlink()

    rows_to_run = [sample for sample in test_samples if sample["case_id"] not in seen]
    if limit is not None:
        rows_to_run = rows_to_run[: int(limit)]

    new_rows: list[dict[str, Any]] = []
    for idx, test_sample in enumerate(rows_to_run, start=1):
        examples = choose_examples(train_samples, test_sample, cfg)
        prompt = build_prompt(test_sample, examples, cfg)
        started = time.time()
        raw_response = ""
        ollama_meta: dict[str, Any] = {}
        parsed: dict[str, Any] = {}
        validation_errors: list[str] = []
        error = None
        try:
            raw_response, ollama_meta = ollama_generate(prompt, cfg)
            parsed = extract_json_object(raw_response)
            validation_errors = validate_response_payload(parsed)
            if validation_errors:
                error = "Response schema validation failed: " + "; ".join(validation_errors)
        except Exception as exc:
            error = str(exc)

        response_valid = bool(parsed) and not validation_errors and error is None
        prediction = normalize_prediction(parsed) if response_valid else "unknown"
        truth = truth_by_case.get(test_sample["case_id"], {})
        ground_truth = truth.get("label")
        row = {
            "schema_version": "cert_llm_prediction_v2",
            "execution_id": execution_id,
            "run_config_sha256": run_config_sha256,
            "case_id": test_sample["case_id"],
            "variant": variant,
            "model": cfg["ollama"]["model"],
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth,
            "response_valid": response_valid,
            "response_validation_errors": validation_errors,
            "confidence": parsed.get("confidence") if isinstance(parsed, dict) else None,
            "justification": parsed.get("justification") if isinstance(parsed, dict) else None,
            "parsed_response": parsed or None,
            "prompt_sha256": sha256_text(prompt),
            "selected_examples": [
                {
                    "case_id": example["case_id"],
                    "label": example.get("label"),
                    "user": example.get("user"),
                    "date": example.get("date"),
                    "similarity": round(float(example.get("_similarity") or 0.0), 4),
                }
                for example in examples
            ],
            "error": error,
            "latency_seconds": round(time.time() - started, 3),
            "ollama_done_reason": ollama_meta.get("done_reason"),
            "ollama_response_model": ollama_meta.get("model"),
            "ollama_created_at": ollama_meta.get("created_at"),
            "ollama_total_duration_ns": ollama_meta.get("total_duration"),
            "ollama_load_duration_ns": ollama_meta.get("load_duration"),
            "ollama_prompt_eval_count": ollama_meta.get("prompt_eval_count"),
            "ollama_prompt_eval_duration_ns": ollama_meta.get("prompt_eval_duration"),
            "ollama_eval_count": ollama_meta.get("eval_count"),
            "ollama_eval_duration_ns": ollama_meta.get("eval_duration"),
            "raw_response": raw_response,
        }
        new_rows.append(row)
        with predictions_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"[{idx}/{len(rows_to_run)}] {row['case_id']} -> {prediction}")

    all_rows = existing + new_rows if not overwrite else new_rows
    project_root = Path(str(cfg["project_root"]))
    predicted_case_ids = [str(row.get("case_id")) for row in all_rows]
    expected_case_ids = {str(row.get("case_id")) for row in test_samples}
    report = {
        "schema_version": "cert_llm_evaluation_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execution_id": execution_id,
        "execution_ids_in_predictions": sorted(
            {str(row.get("execution_id")) for row in all_rows if row.get("execution_id")}
        ),
        "run_config_sha256": run_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "evaluator_sha256": evaluator_sha256,
        "prompt_builder_sha256": prompt_builder_sha256,
        "config_source": {
            "path": str(config_source_path) if config_source_path else None,
            "sha256": sha256_file(config_source_path) if config_source_path and config_source_path.is_file() else None,
        },
        "resolved_config": cfg,
        "git": git_metadata(project_root),
        "environment": environment_metadata(),
        "ollama_runtime": ollama_runtime,
        "variant": variant,
        "model": cfg["ollama"]["model"],
        "base_url": cfg["ollama"]["base_url"],
        "k_examples": int(cfg["llm"].get("k_examples", 5)),
        "num_ctx": int(cfg["ollama"].get("num_ctx", 16384)),
        "temperature": float(cfg["ollama"].get("temperature", 0.0)),
        "limit": limit,
        "overwrite": overwrite,
        "resumed_existing_predictions": bool(existing),
        "num_test_cases": len(test_samples),
        "num_predictions": len(all_rows),
        "num_unique_predictions": len(set(predicted_case_ids)),
        "run_complete": (
            len(all_rows) == len(test_samples)
            and len(set(predicted_case_ids)) == len(predicted_case_ids)
            and set(predicted_case_ids) == expected_case_ids
        ),
        "input_files": {
            name: {"path": str(path), "sha256": input_file_sha256[name]}
            for name, path in input_paths.items()
        },
        "predictions_file": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path) if predictions_path.is_file() else None,
        "evaluation_report_file": str(report_path),
        "metrics": metrics(all_rows),
    }
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CERT user-day JSONL subsets with a local Ollama model.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config.")
    parser.add_argument("--variant", default=None, choices=["day_only", "with_context"], help="Subset variant to evaluate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions for this variant.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of test cases for this run.")
    args = parser.parse_args(argv)
    report = run(args.config, variant=args.variant, overwrite=args.overwrite, limit=args.limit)
    print(f"Wrote LLM evaluation report: {report['predictions_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
