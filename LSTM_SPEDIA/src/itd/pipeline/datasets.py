# Implementa leitura de shards .npz e criação de tf.data.Dataset para treino/val/test
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import config


def list_npz_files(split_dir: Path) -> List[Path]:
    return sorted(split_dir.glob("shard_*.npz"))


def _sample_key(user_value: object, window_end_value: object) -> str:
    return f"{str(user_value)}||{str(window_end_value)}"


def _current_weighting_signature() -> dict:
    return {
        "enabled": bool(config.USE_HARD_SEGMENT_WEIGHTING),
        "workhours_bonus": float(config.HARD_SEGMENT_WORKHOURS_BONUS),
        "len1_bonus": float(config.HARD_SEGMENT_LEN1_BONUS),
        "false_pos_bonus": float(config.HARD_SEGMENT_FALSE_POS_BONUS),
        "max_weight": float(config.HARD_SEGMENT_MAX_WEIGHT),
    }


@lru_cache(maxsize=None)
def _shard_weighting_signature(shard_path: str) -> dict:
    meta_path = Path(shard_path).parent.parent / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(meta.get("hard_segment_weighting", {}))


def _weighting_signature_matches(shard_path: Path) -> bool:
    return _shard_weighting_signature(str(Path(shard_path).resolve())) == _current_weighting_signature()


def _base_weights_from_metadata(seq_len: np.ndarray, is_workhours: np.ndarray) -> np.ndarray:
    seq_len = np.asarray(seq_len, dtype=np.int32)
    is_workhours = np.asarray(is_workhours, dtype=np.float32)
    w = np.ones(shape=(seq_len.shape[0],), dtype=np.float32)
    if bool(config.USE_HARD_SEGMENT_WEIGHTING):
        w += float(config.HARD_SEGMENT_WORKHOURS_BONUS) * is_workhours
        w += float(config.HARD_SEGMENT_LEN1_BONUS) * (seq_len <= 1).astype(np.float32)
    return np.clip(w, 1.0, float(config.HARD_SEGMENT_MAX_WEIGHT)).astype(np.float32)


def _is_workhours_from_window_end(window_end_ts: np.ndarray) -> np.ndarray:
    w_end = pd.to_datetime(pd.Series(window_end_ts.astype(str)), errors="coerce")
    hour = w_end.dt.hour.fillna(-1).astype(int).to_numpy()
    return (1.0 - ((hour < 7) | (hour >= 19)).astype(np.float32)).astype(np.float32)


def _resolve_seq_len(z) -> np.ndarray:
    if "seq_len" in z.files:
        return z["seq_len"].astype(np.int32, copy=False)
    return (z["X_event"] != 0).sum(axis=1).astype(np.int32)


def _resolve_sample_weights(
    z,
    shard_path: Path,
    hard_negative_bonus_by_key: dict[str, float] | None = None,
) -> np.ndarray:
    y = z["y"]
    n = int(y.shape[0])

    if "W_base" in z.files and _weighting_signature_matches(shard_path):
        weights = z["W_base"].astype(np.float32, copy=False)
    else:
        seq_len = _resolve_seq_len(z)
        is_workhours = (
            z["is_workhours"]
            if "is_workhours" in z.files
            else _is_workhours_from_window_end(
                z["window_end_ts"] if "window_end_ts" in z.files else np.array([""] * n, dtype=str)
            )
        )
        weights = _base_weights_from_metadata(seq_len=seq_len, is_workhours=is_workhours)

    if hard_negative_bonus_by_key:
        if "sample_key" in z.files:
            sample_keys = z["sample_key"].astype(str)
        else:
            users = z["user"].astype(str) if "user" in z.files else np.array([""] * n, dtype=str)
            w_end = z["window_end_ts"].astype(str) if "window_end_ts" in z.files else np.array([""] * n, dtype=str)
            sample_keys = np.array([_sample_key(users[i], w_end[i]) for i in range(n)], dtype=str)

        bonus = np.fromiter(
            (float(hard_negative_bonus_by_key.get(k, 0.0)) for k in sample_keys.tolist()),
            dtype=np.float32,
            count=n,
        )
        weights = weights + bonus
        weights = np.clip(weights, 1.0, float(config.HARD_SEGMENT_MAX_WEIGHT)).astype(np.float32)

    return weights.astype(np.float32, copy=False)


def _load_shard_arrays(
    shard_path: Path,
    include_sample_weights: bool = False,
    hard_negative_bonus_by_key: dict[str, float] | None = None,
    return_seq_len: bool = False,
):
    shard_path = Path(shard_path)
    with np.load(shard_path, allow_pickle=True) as z:
        seq_len = _resolve_seq_len(z)
        x = {
            "event_id": z["X_event"].astype(np.int32, copy=False),
            "from_domain_id": z["X_from"].astype(np.int32, copy=False),
            "to_domain_id": z["X_to"].astype(np.int32, copy=False),
            "url_domain_id": z["X_url"].astype(np.int32, copy=False),
            "file_ext_id": z["X_ext"].astype(np.int32, copy=False),
            "num": z["X_num"].astype(np.float32, copy=False),
            "win_num": z["X_win"].astype(np.float32, copy=False)
            if "X_win" in z.files else np.zeros((z["X_event"].shape[0], config.WIN_FEATURES), dtype=np.float32),
        }
        y = z["y"].astype(np.float32, copy=False)

        if include_sample_weights:
            w = _resolve_sample_weights(
                z,
                shard_path=shard_path,
                hard_negative_bonus_by_key=hard_negative_bonus_by_key,
            )
            if return_seq_len:
                return x, y, w, seq_len
            return x, y, w
        if return_seq_len:
            return x, y, seq_len
        return x, y


def _concat_feature_batches(feature_batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not feature_batches:
        raise ValueError("feature_batches nao pode estar vazio")
    return {
        key: np.concatenate([batch[key] for batch in feature_batches], axis=0)
        for key in feature_batches[0]
    }


def _load_dataset_arrays(
    files: List[Path],
    include_sample_weights: bool = False,
    hard_negative_bonus_by_key: dict[str, float] | None = None,
    return_seq_len: bool = False,
):
    if not files:
        raise ValueError("Nenhum shard .npz fornecido para criar o dataset")

    x_parts: list[dict[str, np.ndarray]] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    seq_len_parts: list[np.ndarray] = []

    for shard_path in files:
        loaded = _load_shard_arrays(
            shard_path=shard_path,
            include_sample_weights=include_sample_weights,
            hard_negative_bonus_by_key=hard_negative_bonus_by_key,
            return_seq_len=return_seq_len,
        )
        if include_sample_weights:
            if return_seq_len:
                x_part, y_part, w_part, seq_len_part = loaded
                seq_len_parts.append(seq_len_part)
            else:
                x_part, y_part, w_part = loaded
            w_parts.append(w_part)
        else:
            if return_seq_len:
                x_part, y_part, seq_len_part = loaded
                seq_len_parts.append(seq_len_part)
            else:
                x_part, y_part = loaded
        x_parts.append(x_part)
        y_parts.append(y_part)

    x = _concat_feature_batches(x_parts)
    y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False)
    seq_len = np.concatenate(seq_len_parts, axis=0).astype(np.int32, copy=False) if return_seq_len else None

    if include_sample_weights:
        w = np.concatenate(w_parts, axis=0).astype(np.float32, copy=False)
        if return_seq_len:
            return x, y, w, seq_len
        return x, y, w
    if return_seq_len:
        return x, y, seq_len
    return x, y


def _bucket_boundaries(max_events: int) -> list[int]:
    boundaries = sorted({
        int(v) for v in getattr(config, "SEQUENCE_LENGTH_BUCKET_BOUNDARIES", [])
        if 0 < int(v) < int(max_events)
    })
    return boundaries


def _trim_features_to_batch_max_len(x: dict[str, tf.Tensor], seq_len: tf.Tensor) -> dict[str, tf.Tensor]:
    max_len = tf.maximum(1, tf.reduce_max(tf.cast(seq_len, tf.int32)))
    return {
        "event_id": x["event_id"][:, :max_len],
        "from_domain_id": x["from_domain_id"][:, :max_len],
        "to_domain_id": x["to_domain_id"][:, :max_len],
        "url_domain_id": x["url_domain_id"][:, :max_len],
        "file_ext_id": x["file_ext_id"][:, :max_len],
        "num": x["num"][:, :max_len, :],
        "win_num": x["win_num"],
    }


def _trim_batch_with_weights(x, y, w, seq_len):
    return _trim_features_to_batch_max_len(x, seq_len), y, w


def _trim_batch_without_weights(x, y, seq_len):
    return _trim_features_to_batch_max_len(x, seq_len), y


def make_tf_dataset(
    files: List[Path],
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_events: int | None = None,
    include_sample_weights: bool = False,
    hard_negative_bonus_by_key: dict[str, float] | None = None,
):
    max_events = int(max_events or config.MAX_EVENTS_PER_WINDOW)
    loaded = _load_dataset_arrays(
        files,
        include_sample_weights=include_sample_weights,
        hard_negative_bonus_by_key=hard_negative_bonus_by_key,
        return_seq_len=True,
    )

    if include_sample_weights:
        x, y, w, seq_len = loaded
        ds = tf.data.Dataset.from_tensor_slices((x, y, w, seq_len))
    else:
        x, y, seq_len = loaded
        ds = tf.data.Dataset.from_tensor_slices((x, y, seq_len))

    if shuffle and bool(config.USE_SEQUENCE_LENGTH_BUCKETING):
        bucket_boundaries = _bucket_boundaries(max_events=max_events)
        ds = ds.shuffle(
            buffer_size=min(10000, int(seq_len.shape[0])),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.bucket_by_sequence_length(
            element_length_func=(lambda x, y, w, seq_len: tf.cast(seq_len, tf.int32))
            if include_sample_weights else
            (lambda x, y, seq_len: tf.cast(seq_len, tf.int32)),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=[int(batch_size)] * (len(bucket_boundaries) + 1),
            pad_to_bucket_boundary=False,
            drop_remainder=False,
        )
    else:
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(10000, int(seq_len.shape[0])),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(batch_size)

    ds = ds.map(
        _trim_batch_with_weights if include_sample_weights else _trim_batch_without_weights,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
