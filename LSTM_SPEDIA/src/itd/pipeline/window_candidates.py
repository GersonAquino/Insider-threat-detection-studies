

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .window_shards import compute_cuts, write_window_shards_from_events_df


def window_candidate_key(window_size: str, max_events: int) -> str:
    return f"{str(window_size)}|{int(max_events)}"


def window_output_dir(base_dir: Path, window_size: str, max_events: int) -> Path:
    return Path(base_dir) / f"w={str(window_size)}_m={int(max_events)}"


def round_up_to_step(value: float, step: int) -> int:
    step = max(1, int(step))
    return int(math.ceil(float(value) / step) * step)


def count_npz_samples(npz_files: list[Path]) -> int:
    n = 0
    for f in npz_files:
        with np.load(f, allow_pickle=False) as z:
            n += int(z["y"].shape[0])
    return n


def count_npz_samples_in_split(split_dir: Path) -> int:
    return count_npz_samples(sorted(Path(split_dir).glob("shard_*.npz")))


def is_window_dir_compatible(out_dir: Path, window_size: str, max_events: int) -> bool:
    meta_path = Path(out_dir) / "meta.json"
    if not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        int(meta.get("shard_format_version", -1)) == 4
        and
        str(meta.get("window_size")) == str(window_size)
        and int(meta.get("max_events", -1)) == int(max_events)
    )

def reset_window_dir(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    for pattern in ("train/shard_*.npz", "val/shard_*.npz", "test/shard_*.npz"):
        for f in out_dir.glob(pattern):
            f.unlink()
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta_path.unlink()


def _window_length_distribution(events: pd.DataFrame, window_size: str, train_cut: pd.Timestamp) -> pd.Series:
    counts = events.groupby(["user", pd.Grouper(key="ts", freq=window_size)], dropna=False).size()
    if counts.empty:
        return pd.Series(dtype=np.int64)

    window_start = counts.index.get_level_values(1)
    window_end = window_start + pd.to_timedelta(window_size)
    train_mask = window_end <= train_cut
    return counts[train_mask].astype(np.int64)


def derive_spedia_window_candidates(
    events: pd.DataFrame,
    window_sizes: list[str] | None = None,
    quantiles: list[float] | tuple[float, float] | None = None,
    round_step: int | None = None,
    min_gap: int | None = None,
) -> dict:
    if "user" not in events.columns or "ts" not in events.columns:
        raise ValueError("events deve conter as colunas 'user' e 'ts'")

    window_sizes = [str(v) for v in (window_sizes or config.SPEDIA_HPO_WINDOW_SIZES)]
    quantiles = list(quantiles or config.SPEDIA_DYNAMIC_CAP_QUANTILES)
    if len(quantiles) != 2:
        raise ValueError("SPEDIA_DYNAMIC_CAP_QUANTILES deve ter exatamente 2 valores")

    q_low, q_high = [float(v) for v in quantiles]
    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError("Quantis invalidos para derivacao de max_events")

    round_step = int(round_step or config.SPEDIA_DYNAMIC_CAP_ROUND_STEP)
    min_gap = int(min_gap or config.SPEDIA_DYNAMIC_CAP_MIN_GAP)

    min_ts = pd.Timestamp(events["ts"].min())
    max_ts = pd.Timestamp(events["ts"].max())
    cuts = compute_cuts(min_ts, max_ts)

    candidates = []
    by_window = {}
    for window_size in window_sizes:
        counts = _window_length_distribution(events, window_size=window_size, train_cut=cuts.train_cut)
        if counts.empty:
            raise RuntimeError(f"Sem janelas de treino para {window_size}")

        low_raw = float(counts.quantile(q_low))
        high_raw = float(counts.quantile(q_high))
        cap_low = round_up_to_step(low_raw, round_step)
        cap_high = round_up_to_step(max(high_raw, cap_low + min_gap), round_step)
        if cap_high <= cap_low:
            cap_high = cap_low + round_step

        caps = [int(cap_low), int(cap_high)]
        summary = {
            "window_size": window_size,
            "train_windows": int(counts.shape[0]),
            "mean_events": float(counts.mean()),
            "max_events_seen": int(counts.max()),
            "quantiles": {
                str(q_low): low_raw,
                str(q_high): high_raw,
            },
            "caps": caps,
        }

        for cap in caps:
            trunc_rate = float((counts > cap).mean())
            lost_events = float(((counts - cap).clip(lower=0)).sum() / counts.sum())
            candidates.append({
                "key": window_candidate_key(window_size, cap),
                "window_size": window_size,
                "max_events": int(cap),
                "train_truncation_rate": trunc_rate,
                "train_lost_event_rate": lost_events,
            })

        by_window[window_size] = summary

    return {
        "source": "spedia_train_split_percentiles",
        "window_sizes": window_sizes,
        "quantiles": [q_low, q_high],
        "round_step": round_step,
        "min_gap": min_gap,
        "train_cut": str(cuts.train_cut),
        "by_window": by_window,
        "candidates": candidates,
    }


def _candidate_summary(base_dir: Path, window_size: str, max_events: int) -> dict:
    out_dir = window_output_dir(base_dir, window_size, max_events)
    n_train = count_npz_samples_in_split(out_dir / "train")
    n_val = count_npz_samples_in_split(out_dir / "val")
    n_test = count_npz_samples_in_split(out_dir / "test")
    return {
        "key": window_candidate_key(window_size, max_events),
        "window_size": str(window_size),
        "max_events": int(max_events),
        "out_dir": str(out_dir),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "steps_train": int(math.ceil(n_train / config.BATCH_SIZE)) if n_train else 0,
        "steps_val": int(math.ceil(n_val / config.BATCH_SIZE)) if n_val else 0,
        "steps_test": int(math.ceil(n_test / config.BATCH_SIZE)) if n_test else 0,
    }


def build_spedia_windows_manifest(
    events: pd.DataFrame,
    manifest_path: Path | None = None,
    base_dir: Path | None = None,
    shard_size: int = 50000,
    reuse_existing: bool = True,
) -> dict:
    manifest_path = Path(manifest_path or config.SPEDIA_WINDOWS_MANIFEST)
    base_dir = Path(base_dir or config.SPEDIA_WINDOWS_BASE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)

    derivation = derive_spedia_window_candidates(events)
    derivation_candidates = derivation.pop("candidates")

    materialized = []
    for cand in derivation_candidates:
        window_size = cand["window_size"]
        max_events = int(cand["max_events"])
        out_dir = window_output_dir(base_dir, window_size, max_events)

        should_write = True
        if reuse_existing and is_window_dir_compatible(out_dir, window_size=window_size, max_events=max_events):
            should_write = False

        if should_write:
            reset_window_dir(out_dir)
            write_window_shards_from_events_df(
                events=events,
                out_dir=out_dir,
                window_size=window_size,
                max_events=max_events,
                shard_size=shard_size,
            )

        summary = _candidate_summary(base_dir, window_size=window_size, max_events=max_events)
        summary.update({
            "train_truncation_rate": float(cand["train_truncation_rate"]),
            "train_lost_event_rate": float(cand["train_lost_event_rate"]),
        })
        materialized.append(summary)

    manifest = {
        "mode": "dynamic_hpo",
        "base_dir": str(base_dir),
        "manifest_path": str(manifest_path),
        "derivation": derivation,
        "candidates": materialized,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_window_candidates_manifest(manifest_path: Path | None = None) -> dict:
    manifest_path = Path(manifest_path or config.SPEDIA_WINDOWS_MANIFEST)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest nao encontrado: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def candidate_search_space_from_manifest(manifest: dict) -> dict[str, list[int]]:
    search_space: dict[str, list[int]] = {}
    for cand in manifest.get("candidates", []):
        window_size = str(cand["window_size"])
        search_space.setdefault(window_size, []).append(int(cand["max_events"]))

    for window_size, values in search_space.items():
        search_space[window_size] = sorted({int(v) for v in values})
    return search_space


def find_candidate_in_manifest(manifest: dict, window_size: str, max_events: int) -> dict | None:
    target_key = window_candidate_key(window_size, max_events)
    for cand in manifest.get("candidates", []):
        if str(cand.get("key")) == target_key:
            return cand
    return None


def load_spedia_selected_window_config(
    best_hp_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    best_hp_path = Path(best_hp_path or (config.SPEDIA_RUN_DIR / "best_hp.json"))
    report_path = Path(report_path or config.SPEDIA_REPORT_PATH)

    if best_hp_path.exists():
        best_hp = json.loads(best_hp_path.read_text(encoding="utf-8"))
        if "window_size" in best_hp and "max_events" in best_hp:
            return {
                "source": str(best_hp_path),
                "window_size": str(best_hp["window_size"]),
                "max_events": int(best_hp["max_events"]),
            }

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("selected_window_config") or {}
        if "window_size" in selected and "max_events" in selected:
            return {
                "source": str(report_path),
                "window_size": str(selected["window_size"]),
                "max_events": int(selected["max_events"]),
            }

    return {
        "source": "config_fallback",
        "window_size": str(config.WINDOW_SIZE),
        "max_events": int(config.MAX_EVENTS_PER_WINDOW),
    }
