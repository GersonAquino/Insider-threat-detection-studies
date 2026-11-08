


# Implementa adaptador SPEDIA: lê CSV, gera features canónicas por evento e devolve DataFrame canónico
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    extract_email_domain,
    extract_file_ext,
    extract_url_domain,
    log1p_nonneg,
    safe_float,
    signed_log1p,
    stable_hash_bucket,
    to_str,
)
from .. import config

CANONICAL_COLS = [
    "user",
    "ts",
    "label_event",
    "event_id",
    "from_domain_id",
    "to_domain_id",
    "url_domain_id",
    "file_ext_id",
] + config.NUM_FEATURE_NAMES

def load_spedia_events() -> pd.DataFrame:
    df = pd.read_csv(config.SPEDIA_CSV_PATH, low_memory=False)

    # User e Timestamp
    df["User"] = df["User"].astype("string")
    df = df[df["User"].notna() & (df["User"].str.strip() != "")].copy()

    # O CSV equilibrado mistura timestamps com e sem fracções de segundo.
    # format="mixed" evita perdas indevidas no parse vetorizado.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    df = df[df["Timestamp"].notna()].copy()

    df.sort_values(["User", "Timestamp"], inplace=True)

    # Token do evento (Activity|Action|Decoder_name|Command)
    for c in ["Activity", "Action", "Decoder_name", "Command"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].apply(to_str)

    event_token = (
        df["Activity"].astype(str) + "|" +
        df["Action"].astype(str) + "|" +
        df["Decoder_name"].astype(str) + "|" +
        df["Command"].astype(str)
    )

    # Gap global desde o evento anterior do mesmo utilizador.
    # O delta intra-janela é recalculado mais tarde na geração de shards.
    prev_user_gap = df.groupby("User")["Timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)

    # anexos
    attachments = safe_float(df.get("Attachments", pd.Series([np.nan] * len(df))))
    is_missing_attachments = attachments.isna().astype(np.float32)
    attachments_count = attachments.fillna(0.0).astype(np.float32)

    # sizes
    size = safe_float(df.get("Size", pd.Series([np.nan] * len(df))))
    size_before = safe_float(df.get("Size_before", pd.Series([np.nan] * len(df))))
    size_after = safe_float(df.get("Size_after", pd.Series([np.nan] * len(df))))

    is_missing_size = size.isna().astype(np.float32)
    is_missing_size_before = size_before.isna().astype(np.float32)
    is_missing_size_after = size_after.isna().astype(np.float32)

    size = size.fillna(0.0).astype(np.float32)
    size_before = size_before.fillna(0.0).astype(np.float32)
    size_after = size_after.fillna(0.0).astype(np.float32)

    size_before_raw = safe_float(df.get("Size_before", pd.Series([np.nan] * len(df))))
    size_after_raw = safe_float(df.get("Size_after", pd.Series([np.nan] * len(df))))
    size_delta = (size_after_raw - size_before_raw)
    is_missing_size_delta = (size_before_raw.isna() | size_after_raw.isna()).astype(np.float32)
    size_delta = size_delta.fillna(0.0).astype(np.float32)

    # extrações de alta cardinalidade
    from_domain = df.get("From", pd.Series([None] * len(df))).apply(extract_email_domain)
    to_domain = df.get("To", pd.Series([None] * len(df))).apply(extract_email_domain)
    url_domain = df.get("Url", pd.Series([None] * len(df))).apply(extract_url_domain)
    file_ext = df.get("Filename", pd.Series([None] * len(df))).apply(extract_file_ext)

    # hashing estável
    event_id = event_token.map(lambda s: stable_hash_bucket(s, config.HASH_BINS_EVENT)).astype(np.int32)
    from_domain_id = from_domain.map(lambda s: stable_hash_bucket(s, config.HASH_BINS_DOMAIN)).astype(np.int32)
    to_domain_id = to_domain.map(lambda s: stable_hash_bucket(s, config.HASH_BINS_DOMAIN)).astype(np.int32)
    url_domain_id = url_domain.map(lambda s: stable_hash_bucket(s, config.HASH_BINS_URL)).astype(np.int32)
    file_ext_id = file_ext.map(lambda s: stable_hash_bucket(s, config.HASH_BINS_EXT)).astype(np.int32)

    # label binário por evento
    label_event = pd.to_numeric(df["Anomaly"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    out = pd.DataFrame({
        "user": df["User"].astype(str).str.strip(),
        "ts": df["Timestamp"],
        "label_event": label_event.astype(np.int8),
        "event_id": event_id,
        "from_domain_id": from_domain_id,
        "to_domain_id": to_domain_id,
        "url_domain_id": url_domain_id,
        "file_ext_id": file_ext_id,
        "delta_t_log1p": log1p_nonneg(prev_user_gap).astype(np.float32),
        "prev_user_gap_log1p": log1p_nonneg(prev_user_gap).astype(np.float32),

        "attachments_count": attachments_count,
        "is_missing_attachments": is_missing_attachments,

        "size_log1p": log1p_nonneg(size).astype(np.float32),
        "is_missing_size": is_missing_size,

        "size_before_log1p": log1p_nonneg(size_before).astype(np.float32),
        "is_missing_size_before": is_missing_size_before,

        "size_after_log1p": log1p_nonneg(size_after).astype(np.float32),
        "is_missing_size_after": is_missing_size_after,

        "size_delta_log1p": signed_log1p(size_delta).astype(np.float32),
        "is_missing_size_delta": is_missing_size_delta,
    })

    return out[CANONICAL_COLS].copy()
