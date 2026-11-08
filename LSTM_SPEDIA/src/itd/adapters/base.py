

# Implementa utilitários comuns: hashing estável, extractores (domínio/extensão), parsing numérico e log1p
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

NA_STR = "__NA__"

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

def to_str(x) -> str:
    if x is None:
        return NA_STR
    if isinstance(x, float) and np.isnan(x):
        return NA_STR
    s = str(x).strip()
    return s if s else NA_STR

def stable_hash_bucket(text: str, mod: int) -> int:
    """
    Hash estável (md5) para obter bucket em [1, mod-1].
    0 fica reservado para padding.
    """
    if mod <= 2:
        raise ValueError("mod deve ser > 2")
    s = to_str(text).lower()
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h, 16) % (mod - 1)) + 1

def extract_email_domain(value: object) -> str:
    s = to_str(value)
    if s == NA_STR:
        return NA_STR
    m = EMAIL_RE.search(s)
    if not m:
        return NA_STR
    return m.group(2).lower()

def extract_url_domain(value: object) -> str:
    s = to_str(value)
    if s == NA_STR:
        return NA_STR
    try:
        u = urlparse(s if "://" in s else ("http://" + s))
        host = (u.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host if host else NA_STR
    except Exception:
        return NA_STR

def extract_file_ext(value: object) -> str:
    s = to_str(value).lower()
    if s == NA_STR:
        return NA_STR
    s = s.split("/")[-1].split("\\")[-1]
    if "." not in s:
        return NA_STR
    ext = s.rsplit(".", 1)[-1].strip()
    return ext if ext else NA_STR

def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def log1p_nonneg(series: pd.Series) -> pd.Series:
    x = safe_float(series).fillna(0.0)
    x = x.clip(lower=0.0)
    return np.log1p(x)


def signed_log1p(series: pd.Series) -> pd.Series:
    x = safe_float(series).fillna(0.0)
    return np.sign(x) * np.log1p(np.abs(x))
