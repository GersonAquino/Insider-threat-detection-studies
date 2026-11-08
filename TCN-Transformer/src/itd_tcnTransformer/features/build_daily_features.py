# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportReturnType=false
import re

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.itd_tcnTransformer.config.paths import (
    DAILY_FEATURES_DIR,
    INTERIM_DIR,
    REPORTS_DIR,
)
from src.itd_tcnTransformer.utils.io import save_json


INTERNAL_EMAIL_DOMAIN = "@dtaa.com"
RECIPIENT_COLUMNS = ("to", "cc", "bcc")
RELEVANT_FILE_EXTENSIONS = ("doc", "pdf", "txt", "jpg", "zip", "exe")
ARCHIVE_OR_EXECUTABLE_FILE_EXTENSIONS = ("zip", "exe")
HTTP_JOB_SEARCH_DOMAINS = (
    "linkedin.com",
    "careerbuilder.com",
    "monster.com",
    "indeed.com",
    "job-hunt.org",
    "jobhuntersbible.com",
    "simplyhired.com",
)
HTTP_FILE_SHARING_DOMAINS = (
    "mediafire.com",
    "dropbox.com",
    "megaupload.com",
    "4shared.com",
)
HTTP_PIRACY_DOMAINS = ("thepiratebay.org", "torrentz.eu")
HTTP_LEAK_DOMAINS = ("wikileaks.org",)
HTTP_KEYLOGGER_DOMAINS = (
    "www.refog.com",
    "www.dailykeylogger.com",
    "www.actualkeylogger.com",
    "www.keylogpc.com",
    "www.spectorsoft.com",
    "www.webwatchernow.com",
    "www.best-spy-soft.com",
)
HTTP_KEYLOGGER_URL_TERMS = ("keylogger",)
HTTP_CAREER_URL_TERMS = ("jobs", "hotjobs")
PROFILE_CATEGORY_COLUMNS = ("functional_unit", "department", "role", "team")
PROFILE_RARE_CATEGORY_MIN_USERS = 10
EVENT_DAILY_FEATURE_INPUT_FILES = (
    "logon_daily_features.parquet",
    "device_daily_features.parquet",
    "file_daily_features.parquet",
    "email_daily_features.parquet",
    "http_daily_features.parquet",
)
OPTIONAL_DAILY_FEATURE_INPUT_FILES = ("profile_daily_features.parquet",)
FINAL_DAILY_FEATURE_FILE = "daily_features.parquet"
DAILY_FEATURES_REPORT_FILE = "daily_features_report.json"
HOUR_MISSING_SENTINEL = -1
HOUR_FEATURE_COLUMNS = ("logon_first_hour", "logon_last_hour")
EVENT_FEATURE_PREFIXES = ("logon_", "device_", "file_", "email_", "http_")
PROFILE_PSYCHOMETRIC_PREFIX = "profile_psychometric_"
PSYCHOMETRIC_FEATURE_COLUMNS = ("o", "c", "e", "a", "n")
BEHAVIORAL_DERIVED_FEATURE_COLUMNS = (
    "logon_active_span_hours",
    "logon_after_hours_ratio",
    "logon_first_hour_vs_user_30d_median",
    "logon_last_hour_vs_user_30d_median",
    "logon_new_pc_count",
    "logon_new_pc_flag",
    "device_connect_ratio",
    "device_disconnect_ratio",
    "device_after_hours_ratio",
    "device_unique_pc_count",
    "device_after_hours_unique_pc_count",
    "device_connect_count_rolling_7d_mean",
    "device_connect_count_rolling_30d_mean",
    "device_connect_count_vs_rolling_30d_delta",
    "device_connect_count_vs_rolling_30d_ratio",
    "device_connect_count_vs_user_30d_median_delta",
    "device_unique_pc_count_rolling_7d_mean",
    "device_unique_pc_count_rolling_30d_mean",
    "device_unique_pc_count_vs_rolling_30d_delta",
    "device_unique_pc_count_vs_rolling_30d_ratio",
    "device_connect_rolling_7d_sum",
    "device_connect_rolling_30d_sum",
    "device_new_pc_count",
    "device_new_pc_flag",
    "device_connect_without_same_day_logon_flag",
    "file_sensitive_extension_count",
    "file_archive_or_executable_count",
    "file_sensitive_extension_ratio",
    "file_archive_or_executable_ratio",
    "file_after_hours_ratio",
    "file_total_events_rolling_7d_mean",
    "file_total_events_rolling_30d_mean",
    "file_total_events_vs_rolling_30d_delta",
    "file_total_events_vs_rolling_30d_ratio",
    "file_sensitive_extension_count_rolling_7d_mean",
    "file_sensitive_extension_count_rolling_30d_mean",
    "file_sensitive_extension_count_vs_rolling_30d_delta",
    "file_sensitive_extension_count_vs_rolling_30d_ratio",
    "file_sensitive_rolling_7d_sum",
    "file_sensitive_rolling_30d_sum",
    "file_new_pc_count",
    "file_new_pc_flag",
    "email_total_recipient_count",
    "email_external_recipient_ratio",
    "email_external_email_ratio",
    "email_external_recipients_count_rolling_7d_mean",
    "email_external_recipients_count_rolling_30d_mean",
    "email_external_recipients_count_vs_rolling_30d_delta",
    "email_external_recipients_count_vs_rolling_30d_ratio",
    "email_new_external_recipient_count",
    "email_new_external_recipient_flag",
    "email_new_external_domain_count",
    "email_new_external_domain_flag",
    "email_external_recipient_rolling_7d_sum",
    "email_external_recipient_rolling_30d_sum",
    "http_job_search_ratio",
    "http_file_sharing_ratio",
    "http_piracy_ratio",
    "http_leak_ratio",
    "http_keylogger_ratio",
    "http_career_url_ratio",
    "http_after_hours_ratio",
    "http_unique_domains_total_rolling_30d_mean",
    "http_unique_domains_total_vs_rolling_30d_delta",
    "http_unique_domains_total_vs_rolling_30d_ratio",
    "http_new_domain_count",
    "http_new_domain_flag",
    "http_new_suspicious_domain_count",
    "http_new_suspicious_domain_flag",
    "http_suspicious_activity_rolling_7d_sum",
    "http_suspicious_activity_rolling_30d_sum",
    "daily_total_event_count",
    "daily_after_hours_event_count",
    "daily_after_hours_event_ratio",
    "daily_active_modality_count",
    "daily_high_risk_event_count",
    "daily_high_risk_event_ratio",
    "daily_high_risk_event_count_vs_user_30d_median_delta",
    "daily_high_risk_event_count_vs_user_30d_iqr_score",
    "daily_high_risk_event_count_vs_user_7d_mean_delta",
    "daily_total_event_count_rolling_30d_mean",
    "daily_total_event_count_vs_rolling_30d_delta",
    "daily_total_event_count_vs_rolling_30d_ratio",
    "daily_high_risk_rolling_7d_sum",
    "daily_high_risk_rolling_30d_sum",
    "daily_multi_domain_high_risk_flag",
)

__all__ = ["main"]


@dataclass
class DailyFeaturesReport:
    created_at: datetime
    total_rows: int
    total_users: int
    min_day: object | None
    max_day: object | None
    feature_group_counts: dict[str, int] = field(default_factory=dict)
    behavioral_derived_features: list[str] = field(default_factory=list)
    profile_feature_columns: list[str] = field(default_factory=list)
    profile_psychometric_columns: list[str] = field(default_factory=list)
    profile_static_columns: list[str] = field(default_factory=list)
    missing_value_columns: list[str] = field(default_factory=list)
    infinite_value_columns: list[str] = field(default_factory=list)
    zero_variance_columns_before_sequence_filter: list[str] = field(default_factory=list)


def main() -> None:

    _build_logon_features()
    _build_device_features()
    _build_file_features()
    _build_email_features()
    _build_http_features()
    _build_profile_features()
    _build_combined_daily_features()


def _get_interim_source_dir(source_name: str) -> Path:
    source_dir = INTERIM_DIR / source_name

    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Path '{source_dir}' is not a valid directory")

    return source_dir


def _iter_parquet_files(source_dir: Path):
    parquet_files = source_dir.glob("*.parquet")

    for parquet_file in sorted(parquet_files):
        if parquet_file.is_file():
            yield parquet_file


def _save_daily_features(df: pd.DataFrame, output_file_name: str) -> None:
    output_path = DAILY_FEATURES_DIR / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")


def _concat_and_aggregate_counts(
    partials: list[pd.DataFrame],
    source_dir: Path,
    aggregations: dict[str, pd.NamedAgg]
) -> pd.DataFrame:
    if not partials:
        raise ValueError(f"No parquet files found in {source_dir}")

    return (
        pd.concat(partials, ignore_index=True)
        .groupby(["user", "day"], as_index=False)
        .agg(**aggregations)
    )


def _count_unique_user_day_values(
    partials: list[pd.DataFrame],
    value_column: str,
    output_column: str
) -> pd.DataFrame:
    return (
        pd.concat(partials, ignore_index=True)
        .drop_duplicates(subset=["user", "day", value_column])
        .groupby(["user", "day"], as_index=False)
        .size()
        .rename(columns={"size": output_column})
    )


def _add_after_hours_event_column(df: pd.DataFrame) -> None:
    df["is_after_hours_event"] = df["is_after_hours"].astype("int64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator_values = pd.to_numeric(denominator, errors="coerce").fillna(0)

    ratio = numerator_values.divide(
        denominator_values.where(denominator_values.ne(0)),
    )

    return ratio.replace([np.inf, -np.inf], 0).fillna(0)


def _add_user_rolling_features(
    df: pd.DataFrame,
    user_column: str,
    day_column: str,
    source_columns: list[str],
    windows: tuple[int, ...],
) -> pd.DataFrame:
    result = df.sort_values([user_column, day_column]).copy()

    for source_column in source_columns:
        shifted_values = result.groupby(user_column)[source_column].shift(1)

        for window in windows:
            rolling_mean = (
                shifted_values
                .groupby(result[user_column])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            result[f"{source_column}_rolling_{window}d_mean"] = (
                rolling_mean.fillna(0)
            )

        rolling_30d_column = f"{source_column}_rolling_30d_mean"
        if rolling_30d_column in result.columns:
            result[f"{source_column}_vs_rolling_30d_delta"] = (
                result[source_column] - result[rolling_30d_column]
            )
            result[f"{source_column}_vs_rolling_30d_ratio"] = _safe_ratio(
                result[source_column],
                result[rolling_30d_column],
            )

    return result


def _add_user_shifted_rolling_sum_feature(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    window: int,
    user_column: str = "user",
    day_column: str = "day",
) -> pd.DataFrame:
    result = df.sort_values([user_column, day_column]).copy()
    shifted_values = result.groupby(user_column)[source_column].shift(1)
    rolling_sum = (
        shifted_values
        .groupby(result[user_column])
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    result[output_column] = rolling_sum.fillna(0)

    return result


def _add_user_rolling_median_delta(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    window: int = 30,
    user_column: str = "user",
    day_column: str = "day",
) -> pd.DataFrame:
    result = df.sort_values([user_column, day_column]).copy()
    shifted_values = result.groupby(user_column)[source_column].shift(1)
    rolling_median = (
        shifted_values
        .groupby(result[user_column])
        .rolling(window=window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    result[output_column] = (
        result[source_column] - rolling_median
    ).replace([np.inf, -np.inf], 0).fillna(0)

    return result


def _add_user_rolling_mean_delta(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    window: int,
    user_column: str = "user",
    day_column: str = "day",
) -> pd.DataFrame:
    result = df.sort_values([user_column, day_column]).copy()
    shifted_values = result.groupby(user_column)[source_column].shift(1)
    rolling_mean = (
        shifted_values
        .groupby(result[user_column])
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    result[output_column] = (
        result[source_column] - rolling_mean
    ).replace([np.inf, -np.inf], 0).fillna(0)

    return result


def _add_user_rolling_iqr_score(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    window: int = 30,
    user_column: str = "user",
    day_column: str = "day",
) -> pd.DataFrame:
    result = df.sort_values([user_column, day_column]).copy()
    shifted_values = result.groupby(user_column)[source_column].shift(1)
    grouped_values = shifted_values.groupby(result[user_column])
    rolling_median = (
        grouped_values
        .rolling(window=window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    rolling_q1 = (
        grouped_values
        .rolling(window=window, min_periods=1)
        .quantile(0.25)
        .reset_index(level=0, drop=True)
    )
    rolling_q3 = (
        grouped_values
        .rolling(window=window, min_periods=1)
        .quantile(0.75)
        .reset_index(level=0, drop=True)
    )
    rolling_iqr = (rolling_q3 - rolling_q1).replace(0, np.nan)
    score = (result[source_column] - rolling_median).divide(rolling_iqr)
    result[output_column] = score.replace([np.inf, -np.inf], 0).fillna(0)

    return result


def _add_user_cumulative_new_count_feature(
    df: pd.DataFrame,
    source_events: pd.DataFrame,
    entity_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = df.copy()

    if source_events.empty:
        result[output_column] = 0
        return result

    required_columns = {"user", "day", entity_column}
    missing_columns = required_columns - set(source_events.columns)
    if missing_columns:
        raise ValueError(
            "source_events is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    events = source_events[["user", "day", entity_column]].copy()
    events["user"] = events["user"].astype("string").str.strip()
    events["day"] = pd.to_datetime(events["day"]).dt.floor("D")
    events[entity_column] = events[entity_column].astype("string").str.strip().str.lower()
    events = (
        events
        .dropna(subset=["user", "day", entity_column])
        .loc[lambda frame: frame[entity_column].ne("")]
        .drop_duplicates()
        .sort_values(["user", "day", entity_column])
    )

    new_count_rows: list[dict[str, object]] = []

    for user, user_events in events.groupby("user", sort=True):
        seen_entities: set[str] = set()

        for day, day_events in user_events.groupby("day", sort=True):
            day_entities = set(day_events[entity_column].astype(str))
            new_entities = day_entities - seen_entities
            new_count_rows.append(
                {
                    "user": str(user),
                    "day": day,
                    output_column: int(len(new_entities)),
                }
            )
            seen_entities.update(day_entities)

    if not new_count_rows:
        result[output_column] = 0
        return result

    new_counts = pd.DataFrame(new_count_rows)
    result = result.merge(new_counts, on=["user", "day"], how="left")
    result[output_column] = result[output_column].fillna(0).astype("int64")

    return result


def _normalize_http_text_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _add_http_category_flags(
    df: pd.DataFrame,
    normalized_domains: pd.Series,
    normalized_urls: pd.Series
) -> None:
    df["is_http_job_search"] = normalized_domains.isin(HTTP_JOB_SEARCH_DOMAINS)
    df["is_http_career_url"] = df["is_http_job_search"] | _contains_any_literal_term(
        normalized_urls,
        HTTP_CAREER_URL_TERMS,
    )
    df["is_http_file_sharing"] = normalized_domains.isin(HTTP_FILE_SHARING_DOMAINS)
    df["is_http_piracy"] = normalized_domains.isin(HTTP_PIRACY_DOMAINS)
    df["is_http_leak"] = normalized_domains.isin(HTTP_LEAK_DOMAINS)

    df["is_http_keylogger"] = (
        normalized_domains.isin(HTTP_KEYLOGGER_DOMAINS) |
        normalized_urls.str.contains(
            HTTP_KEYLOGGER_URL_TERMS[0],
            regex=False,
            na=False
        )
    )

    df["is_http_suspicious"] = (
        df["is_http_file_sharing"] |
        df["is_http_piracy"] |
        df["is_http_leak"] |
        df["is_http_keylogger"]
    )

    is_after_hours = df["is_after_hours_event"].astype("bool")
    df["is_http_job_search_after_hours"] = df["is_http_job_search"] & is_after_hours
    df["is_http_career_url_after_hours"] = df["is_http_career_url"] & is_after_hours
    df["is_http_suspicious_after_hours"] = df["is_http_suspicious"] & is_after_hours


def _contains_any_literal_term(
    series: pd.Series,
    terms: tuple[str, ...],
) -> pd.Series:
    escaped_terms = "|".join(re.escape(term) for term in terms)
    pattern = rf"(?:^|[^a-z0-9])(?:{escaped_terms})(?:$|[^a-z0-9])"

    return series.str.contains(pattern, regex=True, na=False)


def _build_user_day_index_from_daily_features() -> pd.DataFrame:
    user_day_parts: list[pd.DataFrame] = []

    for file_name in EVENT_DAILY_FEATURE_INPUT_FILES:
        daily_feature_path = DAILY_FEATURES_DIR / file_name

        if not daily_feature_path.exists() or not daily_feature_path.is_file():
            raise FileNotFoundError(f"Daily feature file not found: {daily_feature_path}")

        user_day_parts.append(pd.read_parquet(daily_feature_path, columns=["user", "day"]))

    return (
        pd.concat(user_day_parts, ignore_index=True)
        .dropna(subset=["user", "day"])
        .assign(day=lambda df: pd.to_datetime(df["day"]).dt.floor("D"))
        .drop_duplicates()
        .sort_values(["user", "day"])
    )


def _load_daily_feature_files(
    required_files: tuple[str, ...],
    optional_files: tuple[str, ...]
) -> list[pd.DataFrame]:
    feature_frames: list[pd.DataFrame] = []

    for file_name in required_files:
        daily_feature_path = DAILY_FEATURES_DIR / file_name

        if not daily_feature_path.exists() or not daily_feature_path.is_file():
            raise FileNotFoundError(f"Daily feature file not found: {daily_feature_path}")

        feature_frames.append(pd.read_parquet(daily_feature_path))

    for file_name in optional_files:
        daily_feature_path = DAILY_FEATURES_DIR / file_name

        if not daily_feature_path.exists():
            continue

        if not daily_feature_path.is_file():
            raise FileNotFoundError(f"Daily feature path is not a file: {daily_feature_path}")

        feature_frames.append(pd.read_parquet(daily_feature_path))

    return feature_frames


def _build_user_day_index_from_feature_frames(
    feature_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    if not feature_frames:
        raise ValueError("At least one feature frame is required")

    user_day_parts = []

    for feature_frame in feature_frames:
        user_day_parts.append(feature_frame[["user", "day"]])

    return (
        pd.concat(user_day_parts, ignore_index=True)
        .dropna(subset=["user", "day"])
        .assign(day=lambda df: pd.to_datetime(df["day"]).dt.floor("D"))
        .drop_duplicates()
        .sort_values(["user", "day"])
    )


def _fill_missing_combined_daily_features(df: pd.DataFrame) -> None:
    for column in HOUR_FEATURE_COLUMNS:
        if column not in df.columns:
            continue

        missing_column = f"{column}_missing"
        df[missing_column] = df[column].isna()
        df[column] = df[column].fillna(HOUR_MISSING_SENTINEL).astype("int64")

    for column in df.columns:
        if column in ("user", "day") or column in HOUR_FEATURE_COLUMNS:
            continue

        if column.endswith("_missing"):
            continue

        if column.startswith(EVENT_FEATURE_PREFIXES):
            non_missing_values = df[column].dropna()
            is_boolean_feature = (
                pd.api.types.is_bool_dtype(df[column]) or
                (
                    pd.api.types.is_object_dtype(df[column]) and
                    not non_missing_values.empty and
                    non_missing_values.map(lambda value: isinstance(value, bool)).all()
                )
            )

            if is_boolean_feature:
                df[column] = df[column].fillna(False).astype("bool")
            else:
                df[column] = df[column].fillna(0)

            continue

        if column.startswith(PROFILE_PSYCHOMETRIC_PREFIX):
            df[column] = pd.to_numeric(df[column], errors="raise")
            continue

        if (
            column.startswith("profile_") and
            not column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)
        ):
            df[column] = df[column].fillna(0).astype("int64")


def _validate_no_unencoded_text_columns(df: pd.DataFrame) -> None:
    text_columns = []

    for column in df.columns:
        if column == "user":
            continue

        if (
            pd.api.types.is_object_dtype(df[column]) or
            pd.api.types.is_string_dtype(df[column]) or
            isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            text_columns.append(column)

    if text_columns:
        raise ValueError(
            "Daily features contain unencoded text columns: "
            f"{', '.join(text_columns)}"
        )


def _load_ldap_profile_snapshots() -> pd.DataFrame:
    ldap_dir = _get_interim_source_dir("LDAP")
    ldap_parts: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(ldap_dir):
        ldap_month = pd.to_datetime(
            f"{parquet_file.stem.split('_', maxsplit=1)[0]}-01",
            format="%Y-%m-%d"
        )

        df = pd.read_parquet(
            parquet_file,
            columns=["user", *PROFILE_CATEGORY_COLUMNS]
        )
        df["user"] = df["user"].astype("string").str.strip()
        df["ldap_month"] = ldap_month

        for column in PROFILE_CATEGORY_COLUMNS:
            df[column] = df[column].astype("string").str.strip()

        ldap_parts.append(df)

    if not ldap_parts:
        raise ValueError(f"No LDAP parquet files found in {ldap_dir}")

    return (
        pd.concat(ldap_parts, ignore_index=True)
        .drop_duplicates(subset=["user", "ldap_month"], keep="last")
    )


def _load_psychometric_profiles() -> pd.DataFrame:
    psychometric_dir = _get_interim_source_dir("psychometric")
    psychometric_parts: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(psychometric_dir):
        psychometric_parts.append(
            pd.read_parquet(
                parquet_file,
                columns=["user", *PSYCHOMETRIC_FEATURE_COLUMNS]
            )
        )

    if not psychometric_parts:
        raise ValueError(f"No psychometric parquet files found in {psychometric_dir}")

    psychometric_profiles = pd.concat(psychometric_parts, ignore_index=True)
    psychometric_profiles["user"] = psychometric_profiles["user"].astype("string").str.strip()

    return (
        psychometric_profiles
        .drop_duplicates(subset=["user"], keep="last")
        .rename(
            columns={
                column: f"profile_psychometric_{column}"
                for column in PSYCHOMETRIC_FEATURE_COLUMNS
            }
        )
    )


def _encode_rare_categories_as_other(
    series: pd.Series,
    frequent_values: set[str]
) -> pd.Series:
    encoded = series.astype("string").str.strip()
    missing_values = encoded.isna() | encoded.eq("")

    encoded = encoded.where(missing_values | encoded.isin(frequent_values), "other")
    encoded = encoded.mask(missing_values, "unknown")

    return encoded.astype("string")


def _slugify_category_value(value: object) -> str:
    slug = re.sub(r"[^0-9a-z]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")

    return slug or "unknown"


def _build_logon_features() -> None:
    logon_dir = _get_interim_source_dir("logon")

    print(f"Building logon features ...")

    partial_counts: list[pd.DataFrame] = []
    partial_unique_pc_pairs: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(logon_dir):
        df = pd.read_parquet(
            parquet_file, columns=["user", "day", "activity", "pc", "hour", "is_after_hours"]
        )

        df["is_logon"] = df["activity"].eq("logon").astype("int64")

        # Tecnicamente não é necessário criar esta coluna, já que pode ser calculada a partir da coluna "is_logon" (1 - is_logon), mas pode ser útil para diagnosticar problemas de dados
        df["is_logoff"] = df["activity"].eq("logoff").astype("int64")

        _add_after_hours_event_column(df)

        counts = (
            df.groupby(["user", "day"], as_index=False)
            .agg(
                logon_total_events=pd.NamedAgg("activity", "size"),
                logon_logon_count=pd.NamedAgg("is_logon", "sum"),
                logon_logoff_count=pd.NamedAgg("is_logoff", "sum"),
                logon_after_hours_count=pd.NamedAgg("is_after_hours_event", "sum"),
                logon_first_hour=pd.NamedAgg("hour", "min"),
                logon_last_hour=pd.NamedAgg("hour", "max"),
            )
        )

        partial_counts.append(counts)

        unique_pc_pairs = (
            df[["user", "day", "pc"]]
            .dropna(subset=["user", "day", "pc"])
            .drop_duplicates()
        )

        partial_unique_pc_pairs.append(unique_pc_pairs)

    final_counts = _concat_and_aggregate_counts(
        partial_counts,
        logon_dir,
        {
            "logon_total_events": pd.NamedAgg("logon_total_events", "sum"),
            "logon_logon_count": pd.NamedAgg("logon_logon_count", "sum"),
            "logon_logoff_count": pd.NamedAgg("logon_logoff_count", "sum"),
            "logon_after_hours_count": pd.NamedAgg("logon_after_hours_count", "sum"),
            "logon_first_hour": pd.NamedAgg("logon_first_hour", "min"),
            "logon_last_hour": pd.NamedAgg("logon_last_hour", "max"),
        }
    )

    final_unique_pc_pairs = _count_unique_user_day_values(
        partial_unique_pc_pairs,
        "pc",
        "logon_unique_pc_count"
    )

    final_logon_features = final_counts.merge(
        final_unique_pc_pairs,
        on=["user", "day"],
        how="left"
    )

    final_logon_features["logon_unique_pc_count"] = (
        final_logon_features["logon_unique_pc_count"]
        .fillna(0)
        .astype("int64")
    )

    final_logon_features["logon_has_after_hours_activity"] = (
        final_logon_features["logon_after_hours_count"] > 0
    )
    final_logon_features = _add_logon_behavioral_features(
        final_logon_features,
        pd.concat(partial_unique_pc_pairs, ignore_index=True),
    )

    # .reset_index(drop=True) # Isto não é necessário porque se vai usar Index=False no to_parquet
    final_logon_features = final_logon_features.sort_values(["user", "day"])

    _save_daily_features(final_logon_features, "logon_daily_features.parquet")


def _build_device_features() -> None:
    device_dir = _get_interim_source_dir("device")

    print(f"Building device features ...")

    partial_counts: list[pd.DataFrame] = []
    partial_unique_pc_pairs: list[pd.DataFrame] = []
    partial_after_hours_unique_pc_pairs: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(device_dir):
        df = pd.read_parquet(
            parquet_file, columns=["user", "day", "activity", "pc", "is_after_hours"]
        )

        df["is_connect"] = df["activity"].eq("connect").astype("int64")

        # Isto será feito calculando total_events - total_connects
        # df["is_disconnect"] = df["activity"].eq("disconnect").astype("int64")

        _add_after_hours_event_column(df)

        counts = (
            df.groupby(["user", "day"], as_index=False)
            .agg(
                device_total_events=pd.NamedAgg("activity", "size"),
                device_connect_count=pd.NamedAgg("is_connect", "sum"),
                device_after_hours_count=pd.NamedAgg("is_after_hours_event", "sum"),
            )
        )

        counts["device_disconnect_count"] = (counts["device_total_events"] -
                                             counts["device_connect_count"])

        partial_counts.append(counts)
        partial_unique_pc_pairs.append(
            df[["user", "day", "pc"]]
            .dropna(subset=["user", "day", "pc"])
            .drop_duplicates()
        )
        partial_after_hours_unique_pc_pairs.append(
            df.loc[
                df["is_after_hours_event"].astype("bool"),
                ["user", "day", "pc"]
            ]
            .dropna(subset=["user", "day", "pc"])
            .drop_duplicates()
        )

    final_counts = _concat_and_aggregate_counts(
        partial_counts,
        device_dir,
        {
            "device_total_events": pd.NamedAgg("device_total_events", "sum"),
            "device_connect_count": pd.NamedAgg("device_connect_count", "sum"),
            "device_disconnect_count": pd.NamedAgg("device_disconnect_count", "sum"),
            "device_after_hours_count": pd.NamedAgg("device_after_hours_count", "sum"),
        }
    )

    final_unique_pc_counts = _count_unique_user_day_values(
        partial_unique_pc_pairs,
        "pc",
        "device_unique_pc_count",
    )
    final_after_hours_unique_pc_counts = _count_unique_user_day_values(
        partial_after_hours_unique_pc_pairs,
        "pc",
        "device_after_hours_unique_pc_count",
    )

    final_counts = (
        final_counts
        .merge(final_unique_pc_counts, on=["user", "day"], how="left")
        .merge(final_after_hours_unique_pc_counts, on=["user", "day"], how="left")
    )
    for column in ("device_unique_pc_count", "device_after_hours_unique_pc_count"):
        final_counts[column] = final_counts[column].fillna(0).astype("int64")

    final_device_features = _add_device_behavioral_features(
        final_counts,
        pd.concat(partial_unique_pc_pairs, ignore_index=True),
    )

    final_device_features = final_device_features.sort_values(["user", "day"])
    _save_daily_features(final_device_features, "device_daily_features.parquet")


def _build_file_features() -> None:
    file_dir = _get_interim_source_dir("file")

    print(f"Building file features ...")

    partial_counts: list[pd.DataFrame] = []
    partial_unique_extension_pairs: list[pd.DataFrame] = []
    partial_unique_filename_pairs: list[pd.DataFrame] = []
    partial_unique_pc_pairs: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(file_dir):
        df = pd.read_parquet(
            parquet_file,
            columns=["user", "day", "pc", "filename", "extension", "is_after_hours"]
        )

        _add_after_hours_event_column(df)

        for extension in RELEVANT_FILE_EXTENSIONS:
            df[f"is_ext_{extension}"] = df["extension"].eq(extension).astype("int64")

        counts = (
            df.groupby(["user", "day"], as_index=False)
            .agg(
                file_total_events=pd.NamedAgg("filename", "size"),
                file_after_hours_count=pd.NamedAgg("is_after_hours_event", "sum"),
                file_doc_count=pd.NamedAgg("is_ext_doc", "sum"),
                file_pdf_count=pd.NamedAgg("is_ext_pdf", "sum"),
                file_txt_count=pd.NamedAgg("is_ext_txt", "sum"),
                file_jpg_count=pd.NamedAgg("is_ext_jpg", "sum"),
                file_zip_count=pd.NamedAgg("is_ext_zip", "sum"),
                file_exe_count=pd.NamedAgg("is_ext_exe", "sum"),
            )
        )

        partial_counts.append(counts)

        unique_extension_pairs = (
            df[["user", "day", "extension"]]
            .dropna(subset=["user", "day", "extension"])
            .drop_duplicates()
        )

        unique_filename_pairs = (
            df[["user", "day", "filename"]]
            .dropna(subset=["user", "day", "filename"])
            .drop_duplicates()
        )

        partial_unique_extension_pairs.append(unique_extension_pairs)
        partial_unique_filename_pairs.append(unique_filename_pairs)
        partial_unique_pc_pairs.append(
            df[["user", "day", "pc"]]
            .dropna(subset=["user", "day", "pc"])
            .drop_duplicates()
        )

    final_counts = _concat_and_aggregate_counts(
        partial_counts,
        file_dir,
        {
            "file_total_events": pd.NamedAgg("file_total_events", "sum"),
            "file_after_hours_count": pd.NamedAgg("file_after_hours_count", "sum"),
            "file_doc_count": pd.NamedAgg("file_doc_count", "sum"),
            "file_pdf_count": pd.NamedAgg("file_pdf_count", "sum"),
            "file_txt_count": pd.NamedAgg("file_txt_count", "sum"),
            "file_jpg_count": pd.NamedAgg("file_jpg_count", "sum"),
            "file_zip_count": pd.NamedAgg("file_zip_count", "sum"),
            "file_exe_count": pd.NamedAgg("file_exe_count", "sum"),
        }
    )

    final_unique_extension_pairs = _count_unique_user_day_values(
        partial_unique_extension_pairs,
        "extension",
        "file_unique_extension_count"
    )

    final_unique_filename_pairs = _count_unique_user_day_values(
        partial_unique_filename_pairs,
        "filename",
        "file_unique_filename_count"
    )

    final_file_features = (
        final_counts
        .merge(final_unique_extension_pairs, on=["user", "day"], how="left")
        .merge(final_unique_filename_pairs, on=["user", "day"], how="left")
        .sort_values(["user", "day"])
    )

    final_file_features["file_unique_extension_count"] = (
        final_file_features["file_unique_extension_count"]
        .fillna(0)
        .astype("int64")
    )

    final_file_features["file_unique_filename_count"] = (
        final_file_features["file_unique_filename_count"]
        .fillna(0)
        .astype("int64")
    )
    final_file_features = _add_file_behavioral_features(
        final_file_features,
        pd.concat(partial_unique_pc_pairs, ignore_index=True),
    )

    _save_daily_features(final_file_features, "file_daily_features.parquet")


def _build_email_features() -> None:
    email_dir = _get_interim_source_dir("email")

    print(f"Building email features ...")

    partial_counts: list[pd.DataFrame] = []
    partial_unique_external_recipient_pairs: list[pd.DataFrame] = []
    partial_unique_external_domain_pairs: list[pd.DataFrame] = []

    for parquet_email in _iter_parquet_files(email_dir):
        df = pd.read_parquet(
            parquet_email,
            columns=["user", "day", "to", "cc", "bcc", "size", "attachments", "is_after_hours"]
        )

        df = df.reset_index(drop=True)
        _add_after_hours_event_column(df)
        df["has_attachments"] = df["attachments"] > 0

        (
            recipient_metrics,
            unique_external_recipient_pairs,
            unique_external_domain_pairs,
        ) = _build_email_recipient_metrics(df)
        df = pd.concat([df, recipient_metrics], axis=1)

        counts = (
            df.groupby(["user", "day"], as_index=False)
            .agg(
                email_total_events=pd.NamedAgg("size", "size"),
                email_after_hours_count=pd.NamedAgg("is_after_hours_event", "sum"),
                email_with_attachments_count=pd.NamedAgg("has_attachments", "sum"),
                email_max_attachments=pd.NamedAgg("attachments", "max"),
                email_external_recipients_count=pd.NamedAgg("external_recipients_count", "sum"),
                email_with_external_recipients_count=pd.NamedAgg("has_external_recipients", "sum"),
                email_total_recipient_count=pd.NamedAgg("unique_recipients_count", "sum"),

                # Estas contagens servem apenas para poder calcular médias no fim, não são features em si
                email_unique_recipients_sum=pd.NamedAgg("unique_recipients_count", "sum"),
                email_total_size_sum=pd.NamedAgg("size", "sum"),
            )
        )

        partial_counts.append(counts)
        partial_unique_external_recipient_pairs.append(unique_external_recipient_pairs)
        partial_unique_external_domain_pairs.append(unique_external_domain_pairs)

    final_counts = _concat_and_aggregate_counts(
        partial_counts,
        email_dir,
        {
            "email_total_events": pd.NamedAgg("email_total_events", "sum"),
            "email_after_hours_count": pd.NamedAgg("email_after_hours_count", "sum"),
            "email_with_attachments_count": pd.NamedAgg("email_with_attachments_count", "sum"),
            "email_max_attachments": pd.NamedAgg("email_max_attachments", "max"),
            "email_external_recipients_count": pd.NamedAgg("email_external_recipients_count", "sum"),
            "email_with_external_recipients_count": pd.NamedAgg(
                "email_with_external_recipients_count",
                "sum"
            ),
            "email_total_recipient_count": pd.NamedAgg("email_total_recipient_count", "sum"),

            # Estas contagens servem apenas para poder calcular médias, não são features em si
            "email_unique_recipients_sum": pd.NamedAgg("email_unique_recipients_sum", "sum"),
            "email_total_size_sum": pd.NamedAgg("email_total_size_sum", "sum"),
        }
    )

    final_counts["email_unique_recipients_avg"] = (
        final_counts["email_unique_recipients_sum"] / final_counts["email_total_events"]
    )

    final_counts["email_total_size_avg"] = (
        final_counts["email_total_size_sum"] / final_counts["email_total_events"]
    )

    final_counts = final_counts.drop(
        columns=["email_unique_recipients_sum", "email_total_size_sum"])

    final_unique_external_recipients = _count_unique_user_day_values(
        partial_unique_external_recipient_pairs,
        "recipient",
        "email_unique_external_recipients_count"
    )
    final_unique_external_domains = _count_unique_user_day_values(
        partial_unique_external_domain_pairs,
        "domain",
        "email_unique_external_domains_count"
    )

    final_email_features = (
        final_counts
        .merge(final_unique_external_recipients, on=["user", "day"], how="left")
        .merge(final_unique_external_domains, on=["user", "day"], how="left")
        .sort_values(["user", "day"])
    )

    for column in (
        "email_unique_external_recipients_count",
        "email_unique_external_domains_count",
    ):
        final_email_features[column] = (
            final_email_features[column]
            .fillna(0)
            .astype("int64")
        )

    final_email_features = _add_email_behavioral_features(
        final_email_features,
        pd.concat(partial_unique_external_recipient_pairs, ignore_index=True),
        pd.concat(partial_unique_external_domain_pairs, ignore_index=True),
    )

    _save_daily_features(final_email_features, "email_daily_features.parquet")


def _build_http_features() -> None:
    http_dir = _get_interim_source_dir("http")

    print(f"Building http features ...")

    partial_counts: list[pd.DataFrame] = []
    partial_unique_url_pairs: list[pd.DataFrame] = []
    partial_unique_domain_pairs: list[pd.DataFrame] = []
    partial_unique_job_search_domain_pairs: list[pd.DataFrame] = []
    partial_unique_career_url_domain_pairs: list[pd.DataFrame] = []
    partial_unique_suspicious_domain_pairs: list[pd.DataFrame] = []

    for parquet_file in _iter_parquet_files(http_dir):
        df = pd.read_parquet(
            parquet_file,
            columns=["user", "day", "url", "domain", "is_after_hours"]
        )

        _add_after_hours_event_column(df)
        normalized_domains = _normalize_http_text_column(df["domain"])
        normalized_urls = _normalize_http_text_column(df["url"])
        _add_http_category_flags(df, normalized_domains, normalized_urls)

        counts = (
            df.groupby(["user", "day"], as_index=False)
            .agg(
                http_total_events=pd.NamedAgg("url", "size"),
                http_after_hours_count=pd.NamedAgg("is_after_hours_event", "sum"),
                http_job_search_events_count=pd.NamedAgg("is_http_job_search", "sum"),
                http_job_search_after_hours_count=pd.NamedAgg(
                    "is_http_job_search_after_hours",
                    "sum"
                ),
                http_career_url_events_count=pd.NamedAgg(
                    "is_http_career_url",
                    "sum"
                ),
                http_career_url_after_hours_count=pd.NamedAgg(
                    "is_http_career_url_after_hours",
                    "sum"
                ),
                http_suspicious_events_count=pd.NamedAgg("is_http_suspicious", "sum"),
                http_suspicious_after_hours_count=pd.NamedAgg(
                    "is_http_suspicious_after_hours",
                    "sum"
                ),
                http_file_sharing_events_count=pd.NamedAgg(
                    "is_http_file_sharing",
                    "sum"
                ),
                http_piracy_events_count=pd.NamedAgg("is_http_piracy", "sum"),
                http_leak_events_count=pd.NamedAgg("is_http_leak", "sum"),
                http_keylogger_events_count=pd.NamedAgg("is_http_keylogger", "sum"),
            )
        )

        partial_counts.append(counts)

        unique_url_pairs = df[["user", "day"]].copy()
        unique_url_pairs["url"] = normalized_urls
        unique_url_pairs = (
            unique_url_pairs
            .dropna(subset=["user", "day", "url"])
            .drop_duplicates()
        )

        unique_domain_pairs = df[["user", "day"]].copy()
        unique_domain_pairs["domain"] = normalized_domains
        unique_domain_pairs = (
            unique_domain_pairs
            .dropna(subset=["user", "day", "domain"])
            .drop_duplicates()
        )

        partial_unique_url_pairs.append(unique_url_pairs)
        partial_unique_domain_pairs.append(unique_domain_pairs)

        unique_job_search_domain_pairs = df.loc[
            df["is_http_job_search"],
            ["user", "day"]
        ].copy()
        unique_job_search_domain_pairs["domain"] = normalized_domains[
            df["is_http_job_search"]
        ]
        unique_job_search_domain_pairs = (
            unique_job_search_domain_pairs
            .dropna(subset=["user", "day", "domain"])
            .drop_duplicates()
        )

        unique_career_url_domain_pairs = df.loc[
            df["is_http_career_url"],
            ["user", "day"]
        ].copy()
        unique_career_url_domain_pairs["domain"] = normalized_domains[
            df["is_http_career_url"]
        ]
        unique_career_url_domain_pairs = (
            unique_career_url_domain_pairs
            .dropna(subset=["user", "day", "domain"])
            .drop_duplicates()
        )

        unique_suspicious_domain_pairs = df.loc[
            df["is_http_suspicious"],
            ["user", "day"]
        ].copy()
        unique_suspicious_domain_pairs["domain"] = normalized_domains[
            df["is_http_suspicious"]
        ]
        unique_suspicious_domain_pairs = (
            unique_suspicious_domain_pairs
            .dropna(subset=["user", "day", "domain"])
            .drop_duplicates()
        )

        partial_unique_job_search_domain_pairs.append(unique_job_search_domain_pairs)
        partial_unique_career_url_domain_pairs.append(unique_career_url_domain_pairs)
        partial_unique_suspicious_domain_pairs.append(unique_suspicious_domain_pairs)

    final_counts = _concat_and_aggregate_counts(
        partial_counts,
        http_dir,
        {
            "http_total_events": pd.NamedAgg("http_total_events", "sum"),
            "http_after_hours_count": pd.NamedAgg("http_after_hours_count", "sum"),
            "http_job_search_events_count": pd.NamedAgg(
                "http_job_search_events_count",
                "sum"
            ),
            "http_job_search_after_hours_count": pd.NamedAgg(
                "http_job_search_after_hours_count",
                "sum"
            ),
            "http_career_url_events_count": pd.NamedAgg(
                "http_career_url_events_count",
                "sum"
            ),
            "http_career_url_after_hours_count": pd.NamedAgg(
                "http_career_url_after_hours_count",
                "sum"
            ),
            "http_suspicious_events_count": pd.NamedAgg(
                "http_suspicious_events_count",
                "sum"
            ),
            "http_suspicious_after_hours_count": pd.NamedAgg(
                "http_suspicious_after_hours_count",
                "sum"
            ),
            "http_file_sharing_events_count": pd.NamedAgg(
                "http_file_sharing_events_count",
                "sum"
            ),
            "http_piracy_events_count": pd.NamedAgg("http_piracy_events_count", "sum"),
            "http_leak_events_count": pd.NamedAgg("http_leak_events_count", "sum"),
            "http_keylogger_events_count": pd.NamedAgg(
                "http_keylogger_events_count",
                "sum"
            ),
        }
    )

    final_unique_urls = _count_unique_user_day_values(
        partial_unique_url_pairs,
        "url",
        "http_unique_urls_total"
    )

    final_unique_domains = _count_unique_user_day_values(
        partial_unique_domain_pairs,
        "domain",
        "http_unique_domains_total"
    )

    final_unique_job_search_domains = _count_unique_user_day_values(
        partial_unique_job_search_domain_pairs,
        "domain",
        "http_unique_job_search_domains_total"
    )

    final_unique_career_url_domains = _count_unique_user_day_values(
        partial_unique_career_url_domain_pairs,
        "domain",
        "http_unique_career_url_domains_total"
    )

    final_unique_suspicious_domains = _count_unique_user_day_values(
        partial_unique_suspicious_domain_pairs,
        "domain",
        "http_unique_suspicious_domains_total"
    )

    final_http_features = (
        final_counts
        .merge(final_unique_urls, on=["user", "day"], how="left")
        .merge(final_unique_domains, on=["user", "day"], how="left")
        .merge(final_unique_job_search_domains, on=["user", "day"], how="left")
        .merge(final_unique_career_url_domains, on=["user", "day"], how="left")
        .merge(final_unique_suspicious_domains, on=["user", "day"], how="left")
        .sort_values(["user", "day"])
    )

    for column in (
        "http_unique_urls_total",
        "http_unique_domains_total",
        "http_unique_job_search_domains_total",
        "http_unique_career_url_domains_total",
        "http_unique_suspicious_domains_total",
    ):
        final_http_features[column] = (
            final_http_features[column]
            .fillna(0)
            .astype("int64")
        )
    final_http_features = _add_http_behavioral_features(
        final_http_features,
        pd.concat(partial_unique_domain_pairs, ignore_index=True),
        pd.concat(
            partial_unique_suspicious_domain_pairs +
            partial_unique_job_search_domain_pairs,
            ignore_index=True,
        ),
    )

    _save_daily_features(final_http_features, "http_daily_features.parquet")


def _build_profile_features() -> None:
    print(f"Building profile features ...")

    user_day_index = _build_user_day_index_from_daily_features()
    user_day_index["ldap_month"] = user_day_index["day"].dt.to_period("M").dt.to_timestamp()

    ldap_profiles = _load_ldap_profile_snapshots()
    psychometric_profiles = _load_psychometric_profiles()

    profile_rows = (
        user_day_index
        .merge(ldap_profiles, on=["user", "ldap_month"], how="left")
        .merge(psychometric_profiles, on="user", how="left")
    )

    category_feature_frames: list[pd.DataFrame] = []

    for column in PROFILE_CATEGORY_COLUMNS:
        user_category_pairs = (
            ldap_profiles[["user", column]]
            .dropna(subset=["user", column])
            .drop_duplicates()
        )
        category_user_counts = user_category_pairs[column].value_counts()
        frequent_values = set(
            category_user_counts[
                category_user_counts >= PROFILE_RARE_CATEGORY_MIN_USERS
            ].index
        )

        encoded_categories = _encode_rare_categories_as_other(
            profile_rows[column],
            frequent_values
        )
        slugged_categories = encoded_categories.map(_slugify_category_value)

        dummy_features = pd.get_dummies(
            slugged_categories,
            prefix=f"profile_{column}",
            dtype="int64"
        )

        expected_category_values = sorted(
            frequent_values,
            key=_slugify_category_value
        ) + ["other", "unknown"]
        expected_dummy_columns = [
            f"profile_{column}_{_slugify_category_value(value)}"
            for value in expected_category_values
        ]

        dummy_features = dummy_features.reindex(
            columns=expected_dummy_columns,
            fill_value=0
        )

        category_feature_frames.append(dummy_features)

    psychometric_feature_columns = [
        f"profile_psychometric_{column}"
        for column in PSYCHOMETRIC_FEATURE_COLUMNS
    ]

    final_profile_features = pd.concat(
        [
            profile_rows[["user", "day"]],
            *category_feature_frames,
            profile_rows[psychometric_feature_columns],
        ],
        axis=1
    ).sort_values(["user", "day"])

    _save_daily_features(final_profile_features, "profile_daily_features.parquet")


def _build_combined_daily_features() -> None:
    print(f"Building combined daily features ...")

    feature_frames = _load_daily_feature_files(
        EVENT_DAILY_FEATURE_INPUT_FILES,
        OPTIONAL_DAILY_FEATURE_INPUT_FILES
    )

    final_daily_features = _build_user_day_index_from_feature_frames(feature_frames)

    for feature_frame in feature_frames:
        normalized_feature_frame = feature_frame.copy()
        normalized_feature_frame["day"] = (
            pd.to_datetime(normalized_feature_frame["day"]).dt.floor("D")
        )

        final_daily_features = final_daily_features.merge(
            normalized_feature_frame,
            on=["user", "day"],
            how="left",
            validate="one_to_one"
        )

    _fill_missing_combined_daily_features(final_daily_features)
    final_daily_features = _add_cross_domain_behavioral_features(final_daily_features)
    _validate_no_unencoded_text_columns(final_daily_features)

    final_daily_features = final_daily_features.sort_values(["user", "day"])

    _save_daily_features(final_daily_features, FINAL_DAILY_FEATURE_FILE)
    _save_daily_features_report(_build_daily_features_report(final_daily_features))


def _add_logon_behavioral_features(
    df: pd.DataFrame,
    unique_pc_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["logon_active_span_hours"] = (
        result["logon_last_hour"] - result["logon_first_hour"]
    ).clip(lower=0).fillna(0)
    result["logon_after_hours_ratio"] = _safe_ratio(
        result["logon_after_hours_count"],
        result["logon_total_events"],
    )
    result = _add_user_rolling_median_delta(
        result,
        "logon_first_hour",
        "logon_first_hour_vs_user_30d_median",
    )
    result = _add_user_rolling_median_delta(
        result,
        "logon_last_hour",
        "logon_last_hour_vs_user_30d_median",
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["logon_unique_pc_count"],
        (30,),
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_pc_pairs,
        "pc",
        "logon_new_pc_count",
    )
    result["logon_new_pc_flag"] = result["logon_new_pc_count"].gt(0).astype("int64")

    return result


def _add_device_behavioral_features(
    df: pd.DataFrame,
    unique_pc_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["device_connect_ratio"] = _safe_ratio(
        result["device_connect_count"],
        result["device_total_events"],
    )
    result["device_disconnect_ratio"] = _safe_ratio(
        result["device_disconnect_count"],
        result["device_total_events"],
    )
    result["device_after_hours_ratio"] = _safe_ratio(
        result["device_after_hours_count"],
        result["device_total_events"],
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["device_connect_count", "device_unique_pc_count"],
        (7, 30),
    )
    result = _add_user_rolling_median_delta(
        result,
        "device_connect_count",
        "device_connect_count_vs_user_30d_median_delta",
        window=30,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "device_connect_count",
        "device_connect_rolling_7d_sum",
        7,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "device_connect_count",
        "device_connect_rolling_30d_sum",
        30,
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_pc_pairs,
        "pc",
        "device_new_pc_count",
    )
    result["device_new_pc_flag"] = result["device_new_pc_count"].gt(0).astype("int64")

    return result


def _add_file_behavioral_features(
    df: pd.DataFrame,
    unique_pc_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["file_sensitive_extension_count"] = result[
        [f"file_{extension}_count" for extension in RELEVANT_FILE_EXTENSIONS]
    ].sum(axis=1)
    result["file_archive_or_executable_count"] = result[
        [
            f"file_{extension}_count"
            for extension in ARCHIVE_OR_EXECUTABLE_FILE_EXTENSIONS
        ]
    ].sum(axis=1)
    result["file_sensitive_extension_ratio"] = _safe_ratio(
        result["file_sensitive_extension_count"],
        result["file_total_events"],
    )
    result["file_archive_or_executable_ratio"] = _safe_ratio(
        result["file_archive_or_executable_count"],
        result["file_total_events"],
    )
    result["file_after_hours_ratio"] = _safe_ratio(
        result["file_after_hours_count"],
        result["file_total_events"],
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["file_total_events", "file_sensitive_extension_count"],
        (7, 30),
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "file_sensitive_extension_count",
        "file_sensitive_rolling_7d_sum",
        7,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "file_sensitive_extension_count",
        "file_sensitive_rolling_30d_sum",
        30,
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_pc_pairs,
        "pc",
        "file_new_pc_count",
    )
    result["file_new_pc_flag"] = result["file_new_pc_count"].gt(0).astype("int64")

    return result


def _add_email_behavioral_features(
    df: pd.DataFrame,
    unique_external_recipient_pairs: pd.DataFrame,
    unique_external_domain_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["email_external_recipient_ratio"] = _safe_ratio(
        result["email_external_recipients_count"],
        result["email_total_recipient_count"],
    )
    result["email_external_email_ratio"] = _safe_ratio(
        result["email_with_external_recipients_count"],
        result["email_total_events"],
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["email_external_recipients_count"],
        (7, 30),
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "email_external_recipients_count",
        "email_external_recipient_rolling_7d_sum",
        7,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "email_external_recipients_count",
        "email_external_recipient_rolling_30d_sum",
        30,
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_external_recipient_pairs,
        "recipient",
        "email_new_external_recipient_count",
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_external_domain_pairs,
        "domain",
        "email_new_external_domain_count",
    )
    result["email_new_external_recipient_flag"] = (
        result["email_new_external_recipient_count"].gt(0).astype("int64")
    )
    result["email_new_external_domain_flag"] = (
        result["email_new_external_domain_count"].gt(0).astype("int64")
    )

    return result


def _add_http_behavioral_features(
    df: pd.DataFrame,
    unique_domain_pairs: pd.DataFrame,
    unique_suspicious_domain_pairs: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["http_job_search_ratio"] = _safe_ratio(
        result["http_job_search_events_count"],
        result["http_total_events"],
    )
    result["http_file_sharing_ratio"] = _safe_ratio(
        result["http_file_sharing_events_count"],
        result["http_total_events"],
    )
    result["http_piracy_ratio"] = _safe_ratio(
        result["http_piracy_events_count"],
        result["http_total_events"],
    )
    result["http_leak_ratio"] = _safe_ratio(
        result["http_leak_events_count"],
        result["http_total_events"],
    )
    result["http_keylogger_ratio"] = _safe_ratio(
        result["http_keylogger_events_count"],
        result["http_total_events"],
    )
    result["http_career_url_ratio"] = _safe_ratio(
        result["http_career_url_events_count"],
        result["http_total_events"],
    )
    result["http_after_hours_ratio"] = _safe_ratio(
        result["http_after_hours_count"],
        result["http_total_events"],
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["http_unique_domains_total"],
        (30,),
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "http_suspicious_events_count",
        "http_suspicious_activity_rolling_7d_sum",
        7,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "http_suspicious_events_count",
        "http_suspicious_activity_rolling_30d_sum",
        30,
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_domain_pairs,
        "domain",
        "http_new_domain_count",
    )
    result = _add_user_cumulative_new_count_feature(
        result,
        unique_suspicious_domain_pairs,
        "domain",
        "http_new_suspicious_domain_count",
    )
    result["http_new_domain_flag"] = result["http_new_domain_count"].gt(0).astype("int64")
    result["http_new_suspicious_domain_flag"] = (
        result["http_new_suspicious_domain_count"].gt(0).astype("int64")
    )

    return result


def _add_cross_domain_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    _ensure_numeric_columns(
        result,
        [
            "logon_total_events",
            "device_total_events",
            "file_total_events",
            "email_total_events",
            "http_total_events",
            "logon_after_hours_count",
            "device_after_hours_count",
            "file_after_hours_count",
            "email_after_hours_count",
            "http_after_hours_count",
            "email_external_recipients_count",
            "file_sensitive_extension_count",
            "device_connect_count",
            "http_file_sharing_events_count",
            "http_piracy_events_count",
            "http_leak_events_count",
            "http_keylogger_events_count",
            "http_job_search_events_count",
        ],
    )
    event_count_columns = [
        "logon_total_events",
        "device_total_events",
        "file_total_events",
        "email_total_events",
        "http_total_events",
    ]
    after_hours_columns = [
        "logon_after_hours_count",
        "device_after_hours_count",
        "file_after_hours_count",
        "email_after_hours_count",
        "http_after_hours_count",
    ]
    high_risk_columns = [
        "email_external_recipients_count",
        "file_sensitive_extension_count",
        "device_connect_count",
        "http_file_sharing_events_count",
        "http_piracy_events_count",
        "http_leak_events_count",
        "http_keylogger_events_count",
        "http_job_search_events_count",
    ]

    result["daily_total_event_count"] = result[event_count_columns].sum(axis=1)
    result["daily_after_hours_event_count"] = result[after_hours_columns].sum(axis=1)
    result["daily_after_hours_event_ratio"] = _safe_ratio(
        result["daily_after_hours_event_count"],
        result["daily_total_event_count"],
    )
    result["daily_active_modality_count"] = result[event_count_columns].gt(0).sum(axis=1)
    result["daily_high_risk_event_count"] = result[high_risk_columns].sum(axis=1)
    result["daily_high_risk_event_ratio"] = _safe_ratio(
        result["daily_high_risk_event_count"],
        result["daily_total_event_count"],
    )
    result = _add_user_rolling_median_delta(
        result,
        "daily_high_risk_event_count",
        "daily_high_risk_event_count_vs_user_30d_median_delta",
        window=30,
    )
    result = _add_user_rolling_iqr_score(
        result,
        "daily_high_risk_event_count",
        "daily_high_risk_event_count_vs_user_30d_iqr_score",
        window=30,
    )
    result = _add_user_rolling_mean_delta(
        result,
        "daily_high_risk_event_count",
        "daily_high_risk_event_count_vs_user_7d_mean_delta",
        window=7,
    )
    result = _add_user_rolling_features(
        result,
        "user",
        "day",
        ["daily_total_event_count"],
        (30,),
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "daily_high_risk_event_count",
        "daily_high_risk_rolling_7d_sum",
        7,
    )
    result = _add_user_shifted_rolling_sum_feature(
        result,
        "daily_high_risk_event_count",
        "daily_high_risk_rolling_30d_sum",
        30,
    )
    risk_by_domain = pd.DataFrame(
        {
            "email": result["email_external_recipients_count"].gt(0),
            "file": result["file_sensitive_extension_count"].gt(0),
            "device": result["device_connect_count"].gt(0),
            "http": result[
                [
                    "http_file_sharing_events_count",
                    "http_piracy_events_count",
                    "http_leak_events_count",
                    "http_keylogger_events_count",
                    "http_job_search_events_count",
                ]
            ].sum(axis=1).gt(0),
        },
        index=result.index,
    )
    result["daily_multi_domain_high_risk_flag"] = (
        risk_by_domain.sum(axis=1).ge(2).astype("int64")
    )
    result["device_connect_without_same_day_logon_flag"] = (
        result["device_connect_count"].gt(0)
        & result["logon_total_events"].eq(0)
    ).astype("int64")

    return result


def _ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)


def _extract_recipient_domain(recipient: object) -> object:
    recipient_text = str(recipient).strip().lower()
    if "@" not in recipient_text:
        return pd.NA

    domain = recipient_text.rsplit("@", maxsplit=1)[-1].strip()
    return domain or pd.NA


def _build_email_recipient_metrics(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recipient_parts: list[pd.DataFrame] = []
    email_row_ids = pd.Series(df.index, index=df.index, name="_email_row_id")

    for column in RECIPIENT_COLUMNS:
        part = df[["user", "day"]].copy()
        part["_email_row_id"] = email_row_ids
        part["recipient"] = df[column].astype("string").str.split(";")

        # "Explodir" a coluna de destinatários para ter uma linha por destinatário.
        part = part.explode("recipient")

        part["recipient"] = part["recipient"].astype("string").str.strip().str.lower()

        part = part.dropna(subset=["user", "day", "recipient"])
        part = part[part["recipient"] != ""]

        recipient_parts.append(part)

    recipients = pd.concat(recipient_parts, ignore_index=True)

    unique_recipients = recipients.drop_duplicates(subset=["_email_row_id", "recipient"]).copy()
    unique_recipients["is_external"] = ~unique_recipients["recipient"].str.endswith(
        INTERNAL_EMAIL_DOMAIN,
        na=False
    )

    metric_counts = (
        unique_recipients
        .groupby("_email_row_id", as_index=False)
        .agg(
            unique_recipients_count=pd.NamedAgg("recipient", "size"),
            external_recipients_count=pd.NamedAgg("is_external", "sum"),
        )
    )

    metrics = pd.DataFrame({"_email_row_id": df.index})
    metrics = metrics.merge(metric_counts, on="_email_row_id", how="left")
    metrics["unique_recipients_count"] = (
        metrics["unique_recipients_count"]
        .fillna(0)
        .astype("int64")
    )
    metrics["external_recipients_count"] = (
        metrics["external_recipients_count"]
        .fillna(0)
        .astype("int64")
    )
    metrics["has_external_recipients"] = metrics["external_recipients_count"] > 0
    metrics = metrics.drop(columns=["_email_row_id"])

    external_recipients = unique_recipients[unique_recipients["is_external"]].copy()
    external_recipients["domain"] = external_recipients["recipient"].map(
        _extract_recipient_domain
    )

    unique_external_recipient_pairs = (
        external_recipients[["user", "day", "recipient"]]
        .drop_duplicates()
    )
    unique_external_domain_pairs = (
        external_recipients[["user", "day", "domain"]]
        .dropna(subset=["user", "day", "domain"])
        .drop_duplicates()
    )

    return metrics, unique_external_recipient_pairs, unique_external_domain_pairs


def _build_daily_features_report(df: pd.DataFrame) -> DailyFeaturesReport:
    feature_columns = [column for column in df.columns if column not in ("user", "day")]
    profile_feature_columns = [
        column
        for column in feature_columns
        if column.startswith("profile_")
    ]
    profile_psychometric_columns = [
        column
        for column in profile_feature_columns
        if column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)
    ]
    profile_static_columns = [
        column
        for column in profile_feature_columns
        if not column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)
    ]

    return DailyFeaturesReport(
        created_at=datetime.now(),
        total_rows=int(len(df.index)),
        total_users=int(df["user"].nunique()),
        min_day=df["day"].min() if not df.empty else None,
        max_day=df["day"].max() if not df.empty else None,
        feature_group_counts=_get_feature_group_counts(feature_columns),
        behavioral_derived_features=[
            column
            for column in BEHAVIORAL_DERIVED_FEATURE_COLUMNS
            if column in df.columns
        ],
        profile_feature_columns=profile_feature_columns,
        profile_psychometric_columns=profile_psychometric_columns,
        profile_static_columns=profile_static_columns,
        missing_value_columns=[
            column
            for column in feature_columns
            if df[column].isna().any()
        ],
        infinite_value_columns=_get_infinite_value_columns(df, feature_columns),
        zero_variance_columns_before_sequence_filter=_get_zero_variance_columns(
            df,
            feature_columns,
        ),
    )


def _get_feature_group_counts(feature_columns: list[str]) -> dict[str, int]:
    group_counts = {
        "logon": 0,
        "device": 0,
        "file": 0,
        "email": 0,
        "http": 0,
        "profile_psychometric": 0,
        "profile_static": 0,
        "calendar": 0,
        "behavioral_derived": 0,
        "other": 0,
    }
    behavioral_columns = set(BEHAVIORAL_DERIVED_FEATURE_COLUMNS)

    for column in feature_columns:
        if column in behavioral_columns:
            group_counts["behavioral_derived"] += 1
        elif column.startswith("profile_psychometric_"):
            group_counts["profile_psychometric"] += 1
        elif column.startswith("profile_"):
            group_counts["profile_static"] += 1
        elif column == "calendar_missing_day":
            group_counts["calendar"] += 1
        elif column.startswith("logon_"):
            group_counts["logon"] += 1
        elif column.startswith("device_"):
            group_counts["device"] += 1
        elif column.startswith("file_"):
            group_counts["file"] += 1
        elif column.startswith("email_"):
            group_counts["email"] += 1
        elif column.startswith("http_"):
            group_counts["http"] += 1
        else:
            group_counts["other"] += 1

    return group_counts


def _get_infinite_value_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    infinite_columns: list[str] = []

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype="float64")
        if np.isinf(values).any():
            infinite_columns.append(column)

    return infinite_columns


def _get_zero_variance_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    zero_variance_columns: list[str] = []

    for column in feature_columns:
        if df[column].nunique(dropna=False) <= 1:
            zero_variance_columns.append(column)

    return zero_variance_columns


def _save_daily_features_report(report: DailyFeaturesReport) -> None:
    output_path = REPORTS_DIR / DAILY_FEATURES_REPORT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output_path)


if __name__ == "__main__":
    main()
