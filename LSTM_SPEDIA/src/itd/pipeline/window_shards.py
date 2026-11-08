

# Implementa geração de janelas temporais e escrita de shards .npz por split (train/val/test)
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config


@dataclass
class Cuts:
    train_cut: pd.Timestamp
    val_cut: pd.Timestamp


_NUM_IDX = {name: i for i, name in enumerate(config.NUM_FEATURE_NAMES)}
_DELTA_T_IDX = _NUM_IDX["delta_t_log1p"]

def compute_cuts(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> Cuts:
    total_seconds = (max_ts - min_ts).total_seconds()
    train_cut = min_ts + pd.to_timedelta(total_seconds * config.TRAIN_FRAC, unit="s")
    val_cut = min_ts + pd.to_timedelta(total_seconds * (config.TRAIN_FRAC + config.VAL_FRAC), unit="s")
    return Cuts(train_cut=train_cut, val_cut=val_cut)

def split_for_end_ts(end_ts: pd.Timestamp, cuts: Cuts) -> str:
    if end_ts <= cuts.train_cut:
        return "train"
    if end_ts <= cuts.val_cut:
        return "val"
    return "test"

def _ensure_dirs(base_dir: Path):
    for s in ("train", "val", "test"):
        (base_dir / s).mkdir(parents=True, exist_ok=True)

def _flush_shard(base_dir: Path, split: str, shard_idx: int, payload: dict) -> None:
    out = base_dir / split / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(out, **payload)

def _window_minutes(window_size: str) -> float:
    minutes = pd.to_timedelta(window_size).total_seconds() / 60.0
    if minutes <= 0:
        raise ValueError(f"window_size inválido: {window_size}")
    return float(minutes)

def _offhours_weekend_ratios(ts_values: List[pd.Timestamp]) -> tuple[float, float]:
    n = len(ts_values)
    if n == 0:
        return 0.0, 0.0

    off_hours = 0
    weekend = 0
    for ts in ts_values:
        t = pd.Timestamp(ts)
        h = int(t.hour)
        if h < 7 or h >= 19:
            off_hours += 1
        if int(t.dayofweek) >= 5:
            weekend += 1
    return float(off_hours / n), float(weekend / n)

def _is_workhours(window_end: pd.Timestamp) -> float:
    ts = pd.Timestamp(window_end)
    h = int(ts.hour)
    return float(0.0 if (h < 7 or h >= 19) else 1.0)

def _base_sample_weight(seq_len: int, window_end: pd.Timestamp) -> float:
    w = 1.0
    if bool(config.USE_HARD_SEGMENT_WEIGHTING):
        w += float(config.HARD_SEGMENT_WORKHOURS_BONUS) * _is_workhours(window_end)
        w += float(config.HARD_SEGMENT_LEN1_BONUS) * float(int(seq_len) <= 1)
    return float(np.clip(w, 1.0, float(config.HARD_SEGMENT_MAX_WEIGHT)))

def _hard_segment_weighting_meta() -> dict:
    return {
        "enabled": bool(config.USE_HARD_SEGMENT_WEIGHTING),
        "workhours_bonus": float(config.HARD_SEGMENT_WORKHOURS_BONUS),
        "len1_bonus": float(config.HARD_SEGMENT_LEN1_BONUS),
        "false_pos_bonus": float(config.HARD_SEGMENT_FALSE_POS_BONUS),
        "max_weight": float(config.HARD_SEGMENT_MAX_WEIGHT),
    }


def _intra_window_delta_t_log1p(ts_values: List[pd.Timestamp] | np.ndarray) -> np.ndarray:
    ts_list = [pd.Timestamp(ts) for ts in ts_values]
    if not ts_list:
        return np.zeros((0,), dtype=np.float32)

    delta = np.zeros((len(ts_list),), dtype=np.float32)
    for i in range(1, len(ts_list)):
        gap = max(0.0, (ts_list[i] - ts_list[i - 1]).total_seconds())
        delta[i] = float(np.log1p(gap))
    return delta


def _init_user_window_history() -> dict:
    return {
        "window_count": 0,
        "sum_events_per_min": 0.0,
        "sum_offhours_ratio": 0.0,
        "sum_weekend_ratio": 0.0,
        "seen_event": set(),
        "seen_from": set(),
        "seen_to": set(),
        "seen_url": set(),
        "seen_ext": set(),
    }


def _new_token_ratio(values: np.ndarray, seen_tokens: set[int]) -> float:
    values = np.asarray(values, dtype=np.int32)
    if values.size == 0:
        return 1.0
    if not seen_tokens:
        return 1.0
    is_new = np.fromiter((int(v) not in seen_tokens for v in values.tolist()), dtype=np.float32, count=values.size)
    return float(np.mean(is_new))


def _portable_user_history_features(
    *,
    event_id: np.ndarray,
    from_id: np.ndarray,
    to_id: np.ndarray,
    url_id: np.ndarray,
    ext_id: np.ndarray,
    events_per_min: float,
    off_hours_ratio: float,
    weekend_ratio: float,
    history: dict | None,
) -> np.ndarray:
    history = history or _init_user_window_history()
    window_count = int(history.get("window_count", 0))
    if window_count <= 0:
        return np.array(
            [
                0.0,  # user_hist_window_count_log1p
                0.0,  # user_events_per_min_ratio_log1p
                0.0,  # user_offhours_ratio_delta
                0.0,  # user_weekend_ratio_delta
                1.0,  # user_new_event_ratio
                1.0,  # user_new_from_ratio
                1.0,  # user_new_to_ratio
                1.0,  # user_new_url_ratio
                1.0,  # user_new_ext_ratio
            ],
            dtype=np.float32,
        )

    mean_events_per_min = float(history["sum_events_per_min"] / window_count)
    mean_offhours_ratio = float(history["sum_offhours_ratio"] / window_count)
    mean_weekend_ratio = float(history["sum_weekend_ratio"] / window_count)
    eps = 1e-6

    return np.array(
        [
            float(np.log1p(window_count)),
            float(np.log1p((events_per_min + eps) / (mean_events_per_min + eps))),
            float(off_hours_ratio - mean_offhours_ratio),
            float(weekend_ratio - mean_weekend_ratio),
            _new_token_ratio(event_id, history["seen_event"]),
            _new_token_ratio(from_id, history["seen_from"]),
            _new_token_ratio(to_id, history["seen_to"]),
            _new_token_ratio(url_id, history["seen_url"]),
            _new_token_ratio(ext_id, history["seen_ext"]),
        ],
        dtype=np.float32,
    )


def _update_user_window_history(
    history: dict,
    *,
    event_id: np.ndarray,
    from_id: np.ndarray,
    to_id: np.ndarray,
    url_id: np.ndarray,
    ext_id: np.ndarray,
    events_per_min: float,
    off_hours_ratio: float,
    weekend_ratio: float,
) -> None:
    history["window_count"] = int(history.get("window_count", 0)) + 1
    history["sum_events_per_min"] = float(history.get("sum_events_per_min", 0.0) + float(events_per_min))
    history["sum_offhours_ratio"] = float(history.get("sum_offhours_ratio", 0.0) + float(off_hours_ratio))
    history["sum_weekend_ratio"] = float(history.get("sum_weekend_ratio", 0.0) + float(weekend_ratio))
    history["seen_event"].update(np.asarray(event_id, dtype=np.int32).tolist())
    history["seen_from"].update(np.asarray(from_id, dtype=np.int32).tolist())
    history["seen_to"].update(np.asarray(to_id, dtype=np.int32).tolist())
    history["seen_url"].update(np.asarray(url_id, dtype=np.int32).tolist())
    history["seen_ext"].update(np.asarray(ext_id, dtype=np.int32).tolist())

def _window_features_from_frame(g: pd.DataFrame, window_minutes: float, history: dict | None = None) -> np.ndarray:
    seq_len = int(len(g))
    if seq_len == 0:
        return np.zeros((config.WIN_FEATURES,), dtype=np.float32)

    num = g[config.NUM_FEATURE_NAMES].astype(np.float32).to_numpy()
    num[:, _DELTA_T_IDX] = _intra_window_delta_t_log1p(list(g["ts"]))
    event_id = g["event_id"].astype(np.int32).to_numpy()
    from_id = g["from_domain_id"].astype(np.int32).to_numpy()
    to_id = g["to_domain_id"].astype(np.int32).to_numpy()
    url_id = g["url_domain_id"].astype(np.int32).to_numpy()
    ext_id = g["file_ext_id"].astype(np.int32).to_numpy()

    off_hours_ratio, weekend_ratio = _offhours_weekend_ratios(list(g["ts"]))
    events_per_min = float(seq_len / window_minutes)
    hist_feat = _portable_user_history_features(
        event_id=event_id,
        from_id=from_id,
        to_id=to_id,
        url_id=url_id,
        ext_id=ext_id,
        events_per_min=events_per_min,
        off_hours_ratio=off_hours_ratio,
        weekend_ratio=weekend_ratio,
        history=history,
    )

    feat = np.array([
        float(np.log1p(seq_len)),
        events_per_min,
        float(np.unique(event_id).size / seq_len),
        float(np.unique(from_id).size / seq_len),
        float(np.unique(to_id).size / seq_len),
        float(np.unique(url_id).size / seq_len),
        float(np.unique(ext_id).size / seq_len),
        float(np.mean(num[:, _NUM_IDX["delta_t_log1p"]])),
        float(np.std(num[:, _NUM_IDX["delta_t_log1p"]])),
        float(np.mean(num[:, _NUM_IDX["attachments_count"]])),
        float(np.mean(num[:, _NUM_IDX["is_missing_attachments"]])),
        float(np.mean(num[:, _NUM_IDX["size_log1p"]])),
        float(np.mean(num[:, _NUM_IDX["is_missing_size"]])),
        float(np.mean(num[:, _NUM_IDX["size_delta_log1p"]])),
        float(off_hours_ratio),
        float(weekend_ratio),
    ], dtype=np.float32)

    feat = np.concatenate([feat, hist_feat.astype(np.float32)], axis=0)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def write_window_shards_from_events_df(
    events: pd.DataFrame,
    out_dir: Path,
    window_size: str,
    max_events: int,
    shard_size: int = 50000,
) -> Path:
    """
    Gera janelas (por utilizador e window_start) a partir de um DataFrame em memória (SPEDIA).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dirs(out_dir)

    events = events.sort_values(["user", "ts"]).copy()
    min_ts = events["ts"].min()
    max_ts = events["ts"].max()
    cuts = compute_cuts(min_ts, max_ts)

    meta = {
        "shard_format_version": 4,
        "window_size": window_size,
        "max_events": max_events,
        "num_features": config.NUM_FEATURES,
        "num_feature_names": config.NUM_FEATURE_NAMES,
        "win_features": config.WIN_FEATURES,
        "win_feature_names": config.WIN_FEATURE_NAMES,
        "cuts": {"train_cut": str(cuts.train_cut), "val_cut": str(cuts.val_cut)},
        "hard_segment_weighting": _hard_segment_weighting_meta(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    events["window_start"] = events["ts"].dt.floor(window_size)
    grouped = events.groupby(["user", "window_start"], sort=True)
    window_minutes = _window_minutes(window_size)
    history_by_user: Dict[str, dict] = {}

    shard_idx = {"train": 0, "val": 0, "test": 0}
    buf = {k: [] for k in ("train", "val", "test")}

    def pack_and_write(split: str):
        nonlocal buf
        if not buf[split]:
            return
        rows = buf[split]
        n = len(rows)

        X_event = np.zeros((n, max_events), dtype=np.int32)
        X_from  = np.zeros((n, max_events), dtype=np.int32)
        X_to    = np.zeros((n, max_events), dtype=np.int32)
        X_url   = np.zeros((n, max_events), dtype=np.int32)
        X_ext   = np.zeros((n, max_events), dtype=np.int32)
        X_num   = np.zeros((n, max_events, config.NUM_FEATURES), dtype=np.float32)
        X_win   = np.zeros((n, config.WIN_FEATURES), dtype=np.float32)
        y       = np.zeros((n,), dtype=np.int8)
        seq_len_out = np.zeros((n,), dtype=np.int32)
        is_workhours_out = np.zeros((n,), dtype=np.float32)
        w_base = np.ones((n,), dtype=np.float32)
        w_end = []
        u_out = []
        sample_key = []

        for i, r in enumerate(rows):
            seq_len = r["len"]
            X_event[i, :seq_len] = r["event_id"]
            X_from[i, :seq_len]  = r["from_domain_id"]
            X_to[i, :seq_len]    = r["to_domain_id"]
            X_url[i, :seq_len]   = r["url_domain_id"]
            X_ext[i, :seq_len]   = r["file_ext_id"]
            X_num[i, :seq_len, :] = r["num"]
            X_win[i, :] = r["win_num"]
            y[i] = r["label"]
            seq_len_out[i] = int(seq_len)
            is_workhours_out[i] = _is_workhours(r["window_end"])
            w_base[i] = _base_sample_weight(seq_len, r["window_end"])

            w_end_str = str(r["window_end"])
            user_str = str(r["user"])
            w_end.append(w_end_str)
            u_out.append(user_str)
            sample_key.append(f"{user_str}||{w_end_str}")

        payload = {
            "X_event": X_event,
            "X_from": X_from,
            "X_to": X_to,
            "X_url": X_url,
            "X_ext": X_ext,
            "X_num": X_num,
            "X_win": X_win,
            "y": y,
            "window_end_ts": np.asarray(w_end, dtype=str),
            "user": np.asarray(u_out, dtype=str),
            "seq_len": seq_len_out,
            "is_workhours": is_workhours_out,
            "W_base": w_base,
            "sample_key": np.asarray(sample_key, dtype=str),
        }
        _flush_shard(out_dir, split, shard_idx[split], payload)
        shard_idx[split] += 1
        buf[split] = []

    for (user, wstart), g in grouped:
        wend = wstart + pd.to_timedelta(window_size)
        split = split_for_end_ts(wend, cuts)

        if len(g) > max_events:
            g = g.iloc[-max_events:]

        y_win = int(g["label_event"].max())
        user_str = str(user)
        history = history_by_user.setdefault(user_str, _init_user_window_history())

        num = g[config.NUM_FEATURE_NAMES].astype(np.float32).to_numpy()
        num[:, _DELTA_T_IDX] = _intra_window_delta_t_log1p(list(g["ts"]))
        win_num = _window_features_from_frame(g, window_minutes, history=history)
        seq_len = len(g)
        event_id = g["event_id"].astype(np.int32).to_numpy()
        from_domain_id = g["from_domain_id"].astype(np.int32).to_numpy()
        to_domain_id = g["to_domain_id"].astype(np.int32).to_numpy()
        url_domain_id = g["url_domain_id"].astype(np.int32).to_numpy()
        file_ext_id = g["file_ext_id"].astype(np.int32).to_numpy()
        off_hours_ratio, weekend_ratio = _offhours_weekend_ratios(list(g["ts"]))
        _update_user_window_history(
            history,
            event_id=event_id,
            from_id=from_domain_id,
            to_id=to_domain_id,
            url_id=url_domain_id,
            ext_id=file_ext_id,
            events_per_min=float(seq_len / window_minutes),
            off_hours_ratio=off_hours_ratio,
            weekend_ratio=weekend_ratio,
        )

        buf[split].append({
            "user": user_str,
            "window_end": wend,
            "len": seq_len,
            "label": y_win,
            "event_id": event_id,
            "from_domain_id": from_domain_id,
            "to_domain_id": to_domain_id,
            "url_domain_id": url_domain_id,
            "file_ext_id": file_ext_id,
            "num": num,
            "win_num": win_num,
        })

        if len(buf[split]) >= shard_size:
            pack_and_write(split)

    for s in ("train", "val", "test"):
        pack_and_write(s)

    return out_dir
