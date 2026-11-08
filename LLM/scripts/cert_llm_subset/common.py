from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


CERT_DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

DEFAULT_CONFIG: dict[str, Any] = {
    "project_root": ".",
    "dataset": {
        "name": "cert_r4.2_user_day_v1",
        "raw_dir": "data/raw/CERT/r4.2",
        "answers_glob": "data/raw/CERT/answers/r4.2-*/*.csv",
        "output_dir": "data/llm_subsets/cert_r4.2/user_day_v1",
        "variants": ["day_only", "with_context"],
        "context_days": 7,
        "include_content_preview": True,
        "content_preview_chars": 180,
        "internal_email_domains": ["dtaa.com"],
        "internal_http_domains": ["dtaa.com"],
        "profile_sources": ["ldap", "psychometric"],
        "suspicious_http_keywords": [
            "wikileaks",
            "keylogger",
            "spectorsoft",
            "dailykeylogger",
            "wellresearchedreviews",
            "job",
            "monster",
            "careerbuilder",
            "craigslist",
            "resume",
        ],
    },
    "duckdb": {
        "database": ":memory:",
        "threads": 8,
        "memory_limit": "16GB",
    },
    "sampling": {
        "random_seed": 42,
        "max_main_user_days": 1200,
        "same_user_negatives_per_positive_user": 5,
        "temporal_negatives_per_positive_day": 2,
        "temporal_window_days": 7,
        "similar_user_negatives_per_positive_day": 2,
        "similar_temporal_window_days": 21,
        "hard_negative_limit": 400,
        "many_files_threshold": 25,
        "off_hours_start": 7,
        "off_hours_end": 19,
        "volume_bins": [20, 100, 300],
    },
    "split": {
        "test_size": 0.20,
        "user_disjoint": True,
    },
    "truncation": {
        "max_events_per_user_day": 220,
        "keep_start_events": 12,
        "keep_end_events": 12,
    },
    "llm": {
        "variant": "day_only",
        "k_examples": 5,
        "balance_example_labels": True,
        "prefer_different_users": True,
        "max_events_per_example": 120,
        "max_events_for_test": 180,
        "predictions_file": "llm_predictions.jsonl",
        "evaluation_report_file": "llm_evaluation_report.json",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3:14b",
        "temperature": 0.0,
        "num_ctx": 16384,
        "timeout_seconds": 300,
        "format": {
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
        },
        "think": False,
    },
}


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:  # pragma: no cover - dependency guard
                raise ImportError("YAML config requires pyyaml. Install it or use JSON config.") from exc
            loaded = yaml.safe_load(text) or {}
        else:
            loaded = json.loads(text)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Config must be a mapping: {path}")
        _deep_update(cfg, loaded)

    root_value = os.environ.get("CERT_LLM_PROJECT_ROOT") or cfg.get("project_root") or "."
    root = Path(root_value)
    if not root.is_absolute():
        root = (project_root_from_here() / root).resolve()
    cfg["project_root"] = str(root)
    return cfg


def resolve_path(cfg: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(str(cfg["project_root"])) / path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=json_default, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=json_default, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_cert_datetime(value: str) -> datetime:
    return datetime.strptime(str(value).strip(), CERT_DATETIME_FORMAT)


def stable_int(value: str, seed: int = 0) -> int:
    import hashlib

    token = f"{seed}|{value}".encode("utf-8", errors="replace")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


def sql_literal_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def sql_string(value: str) -> str:
    return str(value).replace("'", "''")


def normalize_domain(domain: str | None) -> str:
    domain = str(domain or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def domain_matches(domain: str, internal_domains: list[str]) -> bool:
    domain = normalize_domain(domain)
    for internal in internal_domains:
        internal_norm = normalize_domain(internal)
        if domain == internal_norm or domain.endswith("." + internal_norm):
            return True
    return False


def email_domains(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    domains = [normalize_domain(match) for match in re.findall(r"@([A-Za-z0-9_.-]+)", text)]
    return [domain for domain in domains if domain]


def url_domain(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else "http://" + text)
    return normalize_domain(parsed.netloc)


def is_external_email(value: Any, internal_domains: list[str]) -> bool:
    domains = email_domains(value)
    return any(not domain_matches(domain, internal_domains) for domain in domains)


def is_external_url(value: Any, internal_domains: list[str]) -> bool:
    domain = url_domain(value)
    if not domain:
        return False
    return not domain_matches(domain, internal_domains)


def compact_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def volume_class(total_events: int, bins: list[int]) -> str:
    if total_events < int(bins[0]):
        return "low"
    if total_events < int(bins[1]):
        return "medium"
    if total_events < int(bins[2]):
        return "high"
    return "very_high"


def label_from_bool(value: bool | int) -> str:
    return "malicious" if bool(value) else "benign"


def sanitize_for_prompt(sample: dict[str, Any], *, keep_label: bool = False) -> dict[str, Any]:
    clean = copy.deepcopy(sample)
    if not keep_label:
        clean.pop("label", None)
    metadata = clean.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("answer_reference", None)
        metadata.pop("answer_reference_omitted", None)
        metadata.pop("scenario", None)
        metadata.pop("malicious_event_count", None)
    return clean

