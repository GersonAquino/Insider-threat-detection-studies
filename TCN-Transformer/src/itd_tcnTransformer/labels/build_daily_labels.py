from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from src.itd_tcnTransformer.config.paths import (
    DAILY_FEATURES_DIR,
    DAILY_LABELS_DIR,
    INTERIM_DIR,
    REPORTS_DIR,
)
from src.itd_tcnTransformer.config.settings import (
    ALLOW_LABEL_DAYS_WITHOUT_FEATURES,
    ANSWER_LABEL_USER_SOURCE,
    EXPECTED_CERT_R42_MALICIOUS_USERS,
    EXPECTED_CERT_R42_SCENARIO_IDS,
    EXPECTED_CERT_R42_TOTAL_USERS,
    LABEL_ANOMALOUS,
    LABEL_MODE,
    LABEL_NORMAL,
    LABEL_SOURCE,
)
from src.itd_tcnTransformer.utils.io import save_json


DAILY_FEATURES_FILE = "daily_features.parquet"
DAILY_LABELS_FILE = "daily_labels.parquet"
DAILY_LABEL_METADATA_FILE = "daily_label_metadata.parquet"
DAILY_LABELS_REPORT_FILE = "daily_labels_report.json"
REPORT_SAMPLE_SIZE = 10
USER_DAY_COLUMNS = ["user", "day"]

__all__ = ["main"]


@dataclass
class LabelsOutsideLogRangeReport:
    count: int = 0
    min_log_day: object | None = None
    max_log_day: object | None = None
    sample: list[dict[str, object]] = field(default_factory=list)


@dataclass
class MaliciousUsersWithoutDailyEventsReport:
    count: int = 0
    sample: list[str] = field(default_factory=list)


@dataclass
class PositiveDaysOutsideDailyFeaturesReport:
    count: int = 0
    sample: list[dict[str, object]] = field(default_factory=list)


@dataclass
class PositiveDaysPerUserSummary:
    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0


@dataclass
class UserDayMultiScenarioConflictsReport:
    count: int = 0
    sample: list[dict[str, object]] = field(default_factory=list)


@dataclass
class AnswerEventUserMismatchReport:
    count: int = 0
    sample: list[dict[str, object]] = field(default_factory=list)


@dataclass
class DailyLabelsValidationReport:
    created_at: datetime | None = None
    label_source: str = ""
    label_mode: str = ""
    total_users: int = 0
    expected_total_users: int = 0
    positive_users: int = 0
    expected_malicious_users: int = 0
    negative_users: int = 0
    total_days: int = 0
    positive_days: int = 0
    negative_days: int = 0
    positive_day_percentage: float = 0.0
    expanded_attack_interval_days: int = 0
    observed_positive_days: int = 0
    attack_days_without_features: int = 0
    attack_days_without_features_percentage: float = 0.0
    attack_days_without_features_by_scenario: dict[str, int] = field(default_factory=dict)
    attack_days_without_features_by_user_summary: PositiveDaysPerUserSummary = field(
        default_factory=PositiveDaysPerUserSummary
    )
    attack_days_without_features_sample: list[dict[str, object]] = field(
        default_factory=list
    )
    observed_positive_days_by_scenario: dict[str, int] = field(default_factory=dict)
    observed_positive_users_by_scenario: dict[str, int] = field(default_factory=dict)
    positive_users_by_scenario: dict[str, int] = field(default_factory=dict)
    positive_days_by_scenario: dict[str, int] = field(default_factory=dict)
    positive_days_per_user_summary: PositiveDaysPerUserSummary = field(
        default_factory=PositiveDaysPerUserSummary
    )
    insiders_parquet_files: list[str] = field(default_factory=list)
    insiders_rows: int = 0
    insiders_unique_users: int = 0
    insiders_unique_scenarios: int = 0
    user_day_multi_scenario_conflicts: UserDayMultiScenarioConflictsReport = field(
        default_factory=UserDayMultiScenarioConflictsReport
    )
    answer_event_users_different_from_filename_count: int = 0
    answer_event_users_different_from_filename_sample: list[dict[str, object]] = field(
        default_factory=list
    )
    malicious_users_without_daily_events: MaliciousUsersWithoutDailyEventsReport = field(
        default_factory=MaliciousUsersWithoutDailyEventsReport
    )
    positive_days_outside_daily_features: PositiveDaysOutsideDailyFeaturesReport = field(
        default_factory=PositiveDaysOutsideDailyFeaturesReport
    )
    labels_outside_log_time_range: LabelsOutsideLogRangeReport = field(
        default_factory=LabelsOutsideLogRangeReport
    )


def main() -> None:
    print("Building daily labels ...")

    user_days = _load_daily_feature_user_days()
    positive_user_days_for_join, positive_label_metadata = _load_malicious_user_days(
        user_days
    )

    print("Creating daily labels ...")
    daily_labels = _build_daily_labels(user_days, positive_user_days_for_join)

    print("Validating daily labels ...")
    _validate_daily_labels(
        daily_labels,
        user_days,
        positive_user_days_for_join,
        positive_label_metadata
    )

    print("Building daily labels validation report ...")
    daily_labels_report = _build_daily_labels_report(
        daily_labels,
        user_days,
        positive_user_days_for_join,
        positive_label_metadata
    )

    _save_daily_labels(daily_labels)
    _save_daily_label_metadata(positive_label_metadata)
    _save_daily_labels_report(daily_labels_report)


def _build_daily_label_metadata_from_positive_days(
    positive_days: pd.DataFrame
) -> pd.DataFrame:
    metadata = positive_days[
        [
            "user",
            "day",
            "label",
            "scenario_id",
            "attack_start_day",
            "attack_end_day",
            "label_source",
        ]
    ].copy()

    return metadata.sort_values(["user", "day", "scenario_id"])


def _build_daily_labels(
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame
) -> pd.DataFrame:
    user_days = _normalize_user_day_frame(user_days)
    positive_user_days_for_join = _normalize_user_day_frame(positive_user_days_for_join)

    _validate_unique_user_day(user_days, "daily feature user/day pairs")
    _validate_unique_user_day(positive_user_days_for_join, "positive user/day pairs")

    daily_labels = user_days.merge(
        positive_user_days_for_join[USER_DAY_COLUMNS + ["label"]],
        on=USER_DAY_COLUMNS,
        how="left"
    )

    daily_labels["label"] = daily_labels["label"].fillna(LABEL_NORMAL).astype("int64")

    return daily_labels[USER_DAY_COLUMNS + ["label"]].sort_values(USER_DAY_COLUMNS)


def _build_daily_labels_report(
    labels: pd.DataFrame,
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame,
    positive_label_metadata: pd.DataFrame
) -> DailyLabelsValidationReport:
    total_days = len(labels.index)
    positive_days = int(labels["label"].eq(LABEL_ANOMALOUS).sum())
    negative_days = int(labels["label"].eq(LABEL_NORMAL).sum())

    positive_users = set(labels.loc[labels["label"].eq(LABEL_ANOMALOUS), "user"])
    all_users = set(labels["user"])
    negative_users = all_users - positive_users

    report = DailyLabelsValidationReport(
        created_at=datetime.now(),
        label_source=LABEL_SOURCE,
        label_mode=LABEL_MODE,
        total_users=len(all_users),
        expected_total_users=EXPECTED_CERT_R42_TOTAL_USERS,
        positive_users=len(positive_users),
        expected_malicious_users=EXPECTED_CERT_R42_MALICIOUS_USERS,
        negative_users=len(negative_users),
        total_days=total_days,
        positive_days=positive_days,
        negative_days=negative_days,
        positive_day_percentage=(
            positive_days / total_days * 100 if total_days > 0 else 0.0
        )
    )

    report.positive_users_by_scenario = _build_positive_users_by_scenario(
        positive_label_metadata
    )
    report.positive_days_by_scenario = _build_positive_days_by_scenario(
        positive_label_metadata
    )
    report.positive_days_per_user_summary = _build_positive_days_per_user_summary(
        labels
    )
    report.expanded_attack_interval_days = _get_report_metadata(
        "expanded_attack_interval_days",
        positive_days
    )
    report.observed_positive_days = _get_report_metadata(
        "observed_positive_days",
        positive_days
    )
    report.attack_days_without_features = _get_report_metadata(
        "attack_days_without_features",
        0
    )
    report.attack_days_without_features_percentage = _get_report_metadata(
        "attack_days_without_features_percentage",
        0.0
    )
    report.attack_days_without_features_by_scenario = _get_report_metadata(
        "attack_days_without_features_by_scenario",
        {}
    )
    report.attack_days_without_features_by_user_summary = _get_report_metadata(
        "attack_days_without_features_by_user_summary",
        PositiveDaysPerUserSummary()
    )
    report.attack_days_without_features_sample = _get_report_metadata(
        "attack_days_without_features_sample",
        []
    )
    report.observed_positive_days_by_scenario = _build_positive_days_by_scenario(
        positive_label_metadata
    )
    report.observed_positive_users_by_scenario = _build_positive_users_by_scenario(
        positive_label_metadata
    )
    report.insiders_parquet_files = _get_report_metadata("insiders_parquet_files", [])
    report.insiders_rows = _get_report_metadata("insiders_rows", 0)
    report.insiders_unique_users = _get_report_metadata("insiders_unique_users", 0)
    report.insiders_unique_scenarios = _get_report_metadata(
        "insiders_unique_scenarios",
        0
    )
    report.user_day_multi_scenario_conflicts = UserDayMultiScenarioConflictsReport(
        count=_get_report_metadata("multi_scenario_conflict_count", 0),
        sample=_get_report_metadata("multi_scenario_conflict_sample", [])
    )
    report.answer_event_users_different_from_filename_count = _get_report_metadata(
        "answer_event_users_different_from_filename_count",
        0
    )
    report.answer_event_users_different_from_filename_sample = _get_report_metadata(
        "answer_event_users_different_from_filename_sample",
        []
    )
    report.malicious_users_without_daily_events = (
        _build_malicious_users_without_daily_events_report(
            user_days,
            positive_user_days_for_join
        )
    )
    report.positive_days_outside_daily_features = (
        _build_positive_days_outside_daily_features_report(
            user_days,
            positive_user_days_for_join
        )
    )
    report.labels_outside_log_time_range = _build_labels_outside_log_range_report(
        labels,
        user_days
    )

    return report


def _build_labels_outside_log_range_report(
    labels: pd.DataFrame,
    user_days: pd.DataFrame
) -> LabelsOutsideLogRangeReport:
    min_log_day = user_days["day"].min()
    max_log_day = user_days["day"].max()
    labels_outside_range = labels[
        labels["day"].lt(min_log_day) | labels["day"].gt(max_log_day)
    ]

    return LabelsOutsideLogRangeReport(
        count=len(labels_outside_range.index),
        min_log_day=_to_python_datetime(min_log_day),
        max_log_day=_to_python_datetime(max_log_day),
        sample=_build_user_day_label_sample(labels_outside_range)
    )


def _build_malicious_users_without_daily_events_report(
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame
) -> MaliciousUsersWithoutDailyEventsReport:
    malicious_users = {str(user) for user in positive_user_days_for_join["user"].unique()}
    users_with_daily_events = {str(user) for user in user_days["user"].unique()}
    missing_users = sorted(malicious_users - users_with_daily_events)

    return MaliciousUsersWithoutDailyEventsReport(
        count=len(missing_users),
        sample=missing_users[:REPORT_SAMPLE_SIZE]
    )


def _build_positive_days_outside_daily_features_report(
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame
) -> PositiveDaysOutsideDailyFeaturesReport:
    missing_positive_days = _get_positive_days_outside_daily_features(
        user_days,
        positive_user_days_for_join
    )

    return PositiveDaysOutsideDailyFeaturesReport(
        count=len(missing_positive_days.index),
        sample=_build_user_day_label_sample(missing_positive_days)
    )


def _build_positive_days_per_user_summary(
    labels: pd.DataFrame
) -> PositiveDaysPerUserSummary:
    positive_days_per_user = labels.loc[
        labels["label"].eq(LABEL_ANOMALOUS)
    ].groupby("user").size()

    if positive_days_per_user.empty:
        return PositiveDaysPerUserSummary()

    return PositiveDaysPerUserSummary(
        min=int(positive_days_per_user.min()),
        max=int(positive_days_per_user.max()),
        mean=float(positive_days_per_user.mean()),
        median=float(positive_days_per_user.median())
    )


def _build_positive_days_by_scenario(
    positive_label_metadata: pd.DataFrame
) -> dict[str, int]:
    if positive_label_metadata.empty:
        return {}

    counts = (
        positive_label_metadata
        .groupby("scenario_id")[USER_DAY_COLUMNS]
        .size()
        .sort_index()
    )

    return _build_int_key_count_dict(counts)


def _build_positive_users_by_scenario(
    positive_label_metadata: pd.DataFrame
) -> dict[str, int]:
    if positive_label_metadata.empty:
        return {}

    counts = (
        positive_label_metadata
        .groupby("scenario_id")["user"]
        .nunique()
        .sort_index()
    )

    return _build_int_key_count_dict(counts)


def _build_user_day_label_sample(df: pd.DataFrame) -> list[dict[str, object]]:
    sample: list[dict[str, object]] = []
    columns = [column for column in [*USER_DAY_COLUMNS, "label"] if column in df.columns]

    for row in df[columns].head(REPORT_SAMPLE_SIZE).itertuples(index=False, name=None):
        item: dict[str, object] = {}
        for column, value in zip(columns, row):
            if column == "day":
                item[column] = _to_python_datetime(value)
            elif column == "label":
                item[column] = int(value)
            else:
                item[column] = str(value)
        sample.append(item)

    return sample


def _build_int_key_count_dict(counts: pd.Series) -> dict[str, int]:
    return {
        str(_to_int(key)): _to_int(value)
        for key, value in counts.items()
    }


def _expand_insider_attack_intervals(insiders: pd.DataFrame) -> pd.DataFrame:
    positive_parts: list[pd.DataFrame] = []

    for row in insiders.itertuples(index=False):
        attack_start_day = _to_day_datetime(row.start_day)
        attack_end_day = _to_day_datetime(row.end_day)
        days = pd.date_range(
            start=attack_start_day,
            end=attack_end_day,
            freq="D"
        )
        part = pd.DataFrame(
            {
                "user": row.user,
                "day": days,
                "label": LABEL_ANOMALOUS,
                "scenario_id": _to_int(row.scenario_id),
                "attack_start_day": attack_start_day,
                "attack_end_day": attack_end_day,
                "label_source": LABEL_SOURCE,
            }
        )
        positive_parts.append(part)

    if not positive_parts:
        return _empty_positive_label_metadata()

    return pd.concat(positive_parts, ignore_index=True)


def _filter_positive_days_to_observed_feature_days(
    positive_days: pd.DataFrame,
    feature_user_days: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_days = _normalize_user_day_frame(positive_days)
    feature_user_days = _normalize_user_day_frame(feature_user_days)
    feature_user_days = feature_user_days[USER_DAY_COLUMNS].drop_duplicates()

    merged_positive_days = positive_days.merge(
        feature_user_days,
        on=USER_DAY_COLUMNS,
        how="left",
        indicator=True
    )
    observed_positive_days = merged_positive_days[
        merged_positive_days["_merge"].eq("both")
    ].drop(columns=["_merge"])
    attack_days_without_features = merged_positive_days[
        merged_positive_days["_merge"].eq("left_only")
    ].drop(columns=["_merge"])

    return observed_positive_days, attack_days_without_features


def _build_attack_days_without_features_sample(
    attack_days_without_features: pd.DataFrame
) -> list[dict[str, object]]:
    sample: list[dict[str, object]] = []
    columns = [
        column
        for column in [*USER_DAY_COLUMNS, "label", "scenario_id"]
        if column in attack_days_without_features.columns
    ]

    for row in attack_days_without_features[columns].head(REPORT_SAMPLE_SIZE).itertuples(
        index=False,
        name=None
    ):
        item: dict[str, object] = {}
        for column, value in zip(columns, row):
            if column == "day":
                item[column] = _to_python_datetime(value)
            elif column in {"label", "scenario_id"}:
                item[column] = _to_int(value)
            else:
                item[column] = str(value)
        sample.append(item)

    return sample


def _build_attack_days_without_features_by_scenario(
    attack_days_without_features: pd.DataFrame
) -> dict[str, int]:
    if attack_days_without_features.empty:
        return {}

    counts = (
        attack_days_without_features
        .groupby("scenario_id")
        .size()
        .sort_index()
    )

    return _build_int_key_count_dict(counts)


def _build_attack_days_without_features_by_user_summary(
    attack_days_without_features: pd.DataFrame
) -> PositiveDaysPerUserSummary:
    if attack_days_without_features.empty:
        return PositiveDaysPerUserSummary()

    counts = attack_days_without_features.groupby("user").size()

    return PositiveDaysPerUserSummary(
        min=int(counts.min()),
        max=int(counts.max()),
        mean=float(counts.mean()),
        median=float(counts.median())
    )


def _extract_answer_file_metadata(parquet_file: Path) -> tuple[int, str]:
    match = re.match(r"r4\.2-(\d+)-([A-Z]{3}\d{4})_", parquet_file.name)

    if match is None:
        raise ValueError(
            "Could not extract scenario/user from answer file name: "
            f"{parquet_file.name}"
        )

    scenario_id = int(match.group(1))
    malicious_user = match.group(2)

    return scenario_id, malicious_user


def _get_answers_dir() -> Path:
    answers_dir = INTERIM_DIR / "answers"

    if not answers_dir.exists() or not answers_dir.is_dir():
        raise NotADirectoryError(f"Path '{answers_dir}' is not a valid directory")

    return answers_dir


def _get_positive_days_outside_daily_features(
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame
) -> pd.DataFrame:
    missing_positive_days = positive_user_days_for_join.merge(
        user_days[USER_DAY_COLUMNS],
        on=USER_DAY_COLUMNS,
        how="left",
        indicator=True
    )

    return missing_positive_days[missing_positive_days["_merge"].eq("left_only")]


def _get_report_metadata(key: str, default):
    return getattr(_load_malicious_user_days, key, default)


def _iter_answer_event_parquet_files(source_dir: Path):
    for parquet_file in _iter_parquet_files(source_dir):
        if parquet_file.stem.startswith("insiders"):
            continue

        yield parquet_file


def _iter_insiders_parquet_files(source_dir: Path):
    for parquet_file in _iter_parquet_files(source_dir):
        if parquet_file.stem.startswith("insiders"):
            yield parquet_file


def _iter_parquet_files(source_dir: Path):
    parquet_files = source_dir.glob("*.parquet")

    for parquet_file in sorted(parquet_files):
        if parquet_file.is_file():
            yield parquet_file


def _load_daily_feature_user_days() -> pd.DataFrame:
    daily_features_path = DAILY_FEATURES_DIR / DAILY_FEATURES_FILE

    if not daily_features_path.exists() or not daily_features_path.is_file():
        raise FileNotFoundError(f"Daily features file not found: {daily_features_path}")

    print(f"Loading daily feature user/day pairs from {daily_features_path} ...")

    return _normalize_user_day_frame(
        pd.read_parquet(daily_features_path, columns=USER_DAY_COLUMNS)
    ).sort_values(USER_DAY_COLUMNS)


def _load_malicious_user_days(
    user_days: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LABEL_SOURCE == "insiders":
        return _load_malicious_user_days_from_insiders(user_days)

    if LABEL_SOURCE == "answers_events":
        return _load_malicious_user_days_from_answer_events()

    raise ValueError(f"Unsupported LABEL_SOURCE: {LABEL_SOURCE}")


def _load_malicious_user_days_from_answer_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    answers_dir = _get_answers_dir()

    print(f"Loading malicious user/day pairs from answer events in {answers_dir} ...")

    positive_parts: list[pd.DataFrame] = []
    different_event_users_count = 0
    different_event_users_sample: list[dict[str, object]] = []

    for parquet_file in _iter_answer_event_parquet_files(answers_dir):
        scenario_id, malicious_user = _extract_answer_file_metadata(parquet_file)
        part = pd.read_parquet(parquet_file, columns=["user", "day"])
        part = _normalize_user_day_frame(part)
        part = part.rename(columns={"user": "event_user"})

        if ANSWER_LABEL_USER_SOURCE == "filename":
            different_users = part.loc[
                part["event_user"].ne(malicious_user),
                "event_user"
            ].dropna().unique()
            different_event_users_count += len(different_users)
            remaining_sample_size = REPORT_SAMPLE_SIZE - len(different_event_users_sample)
            for event_user in different_users[:remaining_sample_size]:
                different_event_users_sample.append(
                    {
                        "file": parquet_file.name,
                        "malicious_user": malicious_user,
                        "event_user": str(event_user),
                    }
                )
            part["user"] = malicious_user
        elif ANSWER_LABEL_USER_SOURCE == "event_user":
            part["user"] = part["event_user"]
        else:
            raise ValueError(
                f"Unsupported ANSWER_LABEL_USER_SOURCE: {ANSWER_LABEL_USER_SOURCE}"
            )

        part["label"] = LABEL_ANOMALOUS
        part["scenario_id"] = scenario_id
        part["attack_start_day"] = part["day"]
        part["attack_end_day"] = part["day"]
        part["label_source"] = LABEL_SOURCE
        positive_parts.append(part)

    if not positive_parts:
        raise ValueError(f"No answer event parquet files found in {answers_dir}")

    positive_days = pd.concat(positive_parts, ignore_index=True)
    positive_days = _resolve_positive_metadata_conflicts(positive_days)
    positive_metadata = _build_daily_label_metadata_from_positive_days(positive_days)
    positive_for_join = positive_metadata[USER_DAY_COLUMNS + ["label"]].drop_duplicates()

    _set_report_metadata("insiders_parquet_files", [])
    _set_report_metadata("insiders_rows", 0)
    _set_report_metadata("insiders_unique_users", 0)
    _set_report_metadata("insiders_unique_scenarios", 0)
    _set_report_metadata(
        "answer_event_users_different_from_filename_count",
        different_event_users_count
    )
    _set_report_metadata(
        "answer_event_users_different_from_filename_sample",
        different_event_users_sample
    )

    return (
        positive_for_join.sort_values(USER_DAY_COLUMNS),
        positive_metadata.sort_values(USER_DAY_COLUMNS)
    )


def _load_malicious_user_days_from_insiders(
    user_days: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    answers_dir = _get_answers_dir()
    insiders_files = list(_iter_insiders_parquet_files(answers_dir))

    if not insiders_files:
        raise FileNotFoundError(
            f"No insiders parquet files found in {answers_dir}. "
            "Expected files matching insiders_*.parquet."
        )

    print(f"Loading malicious user/day pairs from insiders metadata in {answers_dir} ...")

    insiders_parts = [pd.read_parquet(parquet_file) for parquet_file in insiders_files]
    insiders = pd.concat(insiders_parts, ignore_index=True)
    insiders = _prepare_insiders_frame(insiders)

    if LABEL_MODE not in {"attack_interval_day", "attack_interval_observed_day"}:
        raise ValueError(
            'LABEL_MODE must be "attack_interval_day" or '
            '"attack_interval_observed_day" when LABEL_SOURCE="insiders".'
        )

    positive_days = _expand_insider_attack_intervals(insiders)
    positive_days = _resolve_positive_metadata_conflicts(positive_days)

    _set_attack_interval_report_metadata(positive_days, _empty_positive_label_metadata())

    if LABEL_MODE == "attack_interval_observed_day":
        positive_days, attack_days_without_features = (
            _filter_positive_days_to_observed_feature_days(
                positive_days,
                user_days
            )
        )
        _set_attack_interval_report_metadata(
            positive_days,
            attack_days_without_features
        )

        if positive_days.empty:
            raise ValueError(
                "LABEL_MODE='attack_interval_observed_day' produced zero observed "
                "positive days."
            )

    positive_metadata = _build_daily_label_metadata_from_positive_days(positive_days)
    positive_for_join = positive_metadata[USER_DAY_COLUMNS + ["label"]].drop_duplicates()

    _set_report_metadata("answer_event_users_different_from_filename_count", 0)
    _set_report_metadata("answer_event_users_different_from_filename_sample", [])
    _set_report_metadata("insiders_parquet_files", [file.name for file in insiders_files])
    _set_report_metadata("insiders_rows", len(insiders.index))
    _set_report_metadata("insiders_unique_users", int(insiders["user"].nunique()))
    _set_report_metadata(
        "insiders_unique_scenarios",
        int(insiders["scenario_id"].nunique())
    )

    return (
        positive_for_join.sort_values(USER_DAY_COLUMNS),
        positive_metadata.sort_values(USER_DAY_COLUMNS)
    )


def _normalize_user_day_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["user"] = normalized["user"].astype("string").str.strip()
    normalized["day"] = pd.to_datetime(normalized["day"]).dt.floor("D")

    return normalized


def _prepare_insiders_frame(insiders: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "dataset",
        "scenario",
        "user",
        "start",
        "end",
        "start_day",
        "end_day",
    }
    missing_columns = sorted(required_columns - set(insiders.columns))

    if missing_columns:
        raise ValueError(
            "Insiders parquet files are missing required columns: "
            f"{missing_columns}"
        )

    insiders = insiders.copy()
    insiders = insiders[
        insiders["dataset"].astype("string").str.strip().eq("4.2")
    ].copy()

    if insiders.empty:
        raise ValueError("Insiders metadata does not contain CERT r4.2 rows.")

    insiders["user"] = insiders["user"].astype("string").str.strip()
    insiders["scenario_id"] = insiders["scenario"].astype("int64")
    insiders["start_day"] = pd.to_datetime(insiders["start_day"]).dt.floor("D")
    insiders["end_day"] = pd.to_datetime(insiders["end_day"]).dt.floor("D")

    if insiders["user"].isna().any() or insiders["user"].eq("").any():
        raise ValueError("Insiders metadata contains missing malicious users.")

    invalid_intervals = insiders[insiders["start_day"].gt(insiders["end_day"])]
    if not invalid_intervals.empty:
        sample = invalid_intervals[
            ["user", "scenario_id", "start_day", "end_day"]
        ].head(REPORT_SAMPLE_SIZE).to_dict(orient="records")
        raise ValueError(f"Insiders metadata contains invalid intervals: {sample}")

    invalid_scenarios = insiders[
        ~insiders["scenario_id"].isin(EXPECTED_CERT_R42_SCENARIO_IDS)
    ]
    if not invalid_scenarios.empty:
        sample = invalid_scenarios[
            ["user", "scenario_id"]
        ].head(REPORT_SAMPLE_SIZE).to_dict(orient="records")
        raise ValueError(f"Unexpected CERT r4.2 scenario ids in insiders metadata: {sample}")

    unique_malicious_users = int(insiders["user"].nunique())
    if unique_malicious_users != EXPECTED_CERT_R42_MALICIOUS_USERS:
        raise ValueError(
            "Unexpected number of malicious users in insiders metadata. "
            f"Expected {EXPECTED_CERT_R42_MALICIOUS_USERS}, "
            f"found {unique_malicious_users}."
        )

    return insiders


def _resolve_positive_metadata_conflicts(positive_days: pd.DataFrame) -> pd.DataFrame:
    positive_days = positive_days.drop_duplicates(
        subset=[*USER_DAY_COLUMNS, "scenario_id"]
    )
    conflict_rows = positive_days[
        positive_days.duplicated(subset=USER_DAY_COLUMNS, keep=False)
    ].sort_values([*USER_DAY_COLUMNS, "scenario_id"])

    _set_report_metadata("multi_scenario_conflict_count", 0)
    _set_report_metadata("multi_scenario_conflict_sample", [])

    if not conflict_rows.empty:
        conflict_groups = conflict_rows.groupby(USER_DAY_COLUMNS)["scenario_id"].apply(
            lambda values: sorted({_to_int(value) for value in values})
        )
        conflict_sample: list[dict[str, object]] = []
        for index_value, scenario_ids in conflict_groups.head(
            REPORT_SAMPLE_SIZE
        ).items():
            if not isinstance(index_value, tuple) or len(index_value) != 2:
                raise ValueError(
                    "Unexpected index while building multi-scenario conflict sample."
                )

            user, day = index_value
            conflict_sample.append(
                {
                    "user": str(user),
                    "day": _to_python_datetime(day),
                    "scenario_ids": _to_int_list(scenario_ids),
                }
            )

        _set_report_metadata("multi_scenario_conflict_count", len(conflict_groups.index))
        _set_report_metadata("multi_scenario_conflict_sample", conflict_sample)

    return (
        positive_days
        .sort_values([*USER_DAY_COLUMNS, "scenario_id"])
        .drop_duplicates(subset=USER_DAY_COLUMNS, keep="first")
    )


def _save_daily_label_metadata(metadata: pd.DataFrame) -> None:
    output_path = DAILY_LABELS_DIR / DAILY_LABEL_METADATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving daily label metadata to {output_path} ...")
    metadata.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )


def _save_daily_labels(labels: pd.DataFrame) -> None:
    output_path = DAILY_LABELS_DIR / DAILY_LABELS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving daily labels to {output_path} ...")
    labels.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")


def _save_daily_labels_report(report: DailyLabelsValidationReport) -> None:
    output_path = REPORTS_DIR / DAILY_LABELS_REPORT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving daily labels validation report to {output_path} ...")
    save_json(report, output_path)


def _set_report_metadata(key: str, value) -> None:
    setattr(_load_malicious_user_days, key, value)


def _set_attack_interval_report_metadata(
    observed_positive_days: pd.DataFrame,
    attack_days_without_features: pd.DataFrame
) -> None:
    observed_count = int(len(observed_positive_days.index))
    missing_count = int(len(attack_days_without_features.index))
    expanded_count = observed_count + missing_count

    _set_report_metadata("expanded_attack_interval_days", expanded_count)
    _set_report_metadata("observed_positive_days", observed_count)
    _set_report_metadata("attack_days_without_features", missing_count)
    _set_report_metadata(
        "attack_days_without_features_percentage",
        (
            missing_count / expanded_count
            if expanded_count > 0
            else 0.0
        )
    )
    _set_report_metadata(
        "attack_days_without_features_by_scenario",
        _build_attack_days_without_features_by_scenario(
            attack_days_without_features
        )
    )
    _set_report_metadata(
        "attack_days_without_features_by_user_summary",
        _build_attack_days_without_features_by_user_summary(
            attack_days_without_features
        )
    )
    _set_report_metadata(
        "attack_days_without_features_sample",
        _build_attack_days_without_features_sample(attack_days_without_features)
    )


def _to_python_datetime(value: object) -> object:
    to_pydatetime = getattr(value, "to_pydatetime", None)

    if callable(to_pydatetime):
        return to_pydatetime()

    return value


def _to_day_datetime(value: object) -> datetime:
    return pd.Timestamp(str(value)).floor("D").to_pydatetime()


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    return int(str(value))


def _to_int_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        return [_to_int(item) for item in value]

    return [_to_int(value)]


def _validate_daily_labels(
    labels: pd.DataFrame,
    user_days: pd.DataFrame,
    positive_user_days_for_join: pd.DataFrame,
    positive_label_metadata: pd.DataFrame
) -> None:
    _validate_unique_user_day(labels, "daily labels")
    _validate_no_missing_values(labels, USER_DAY_COLUMNS + ["label"], "daily labels")

    invalid_labels = labels[~labels["label"].isin([LABEL_NORMAL, LABEL_ANOMALOUS])]
    if not invalid_labels.empty:
        sample = _build_user_day_label_sample(invalid_labels)
        raise ValueError(f"Daily labels contain invalid label values. Sample: {sample}")

    total_users = labels["user"].nunique()
    if total_users > EXPECTED_CERT_R42_TOTAL_USERS:
        raise ValueError(
            "Daily labels contain more users than expected for CERT r4.2. "
            f"Expected at most {EXPECTED_CERT_R42_TOTAL_USERS}, found {total_users}."
        )

    positive_users = labels.loc[
        labels["label"].eq(LABEL_ANOMALOUS),
        "user"
    ].nunique()
    if LABEL_SOURCE == "insiders" and positive_users != EXPECTED_CERT_R42_MALICIOUS_USERS:
        raise ValueError(
            "Unexpected number of positive users for LABEL_SOURCE='insiders'. "
            f"Expected {EXPECTED_CERT_R42_MALICIOUS_USERS}, found {positive_users}."
        )

    missing_users_report = _build_malicious_users_without_daily_events_report(
        user_days,
        positive_user_days_for_join
    )
    if missing_users_report.count > 0:
        raise ValueError(
            "Some malicious users do not exist in daily features. "
            f"Missing users: {missing_users_report.count}. "
            f"Sample: {missing_users_report.sample}"
        )

    missing_positive_days = _get_positive_days_outside_daily_features(
        user_days,
        positive_user_days_for_join
    )
    if not missing_positive_days.empty and not ALLOW_LABEL_DAYS_WITHOUT_FEATURES:
        sample = _build_user_day_label_sample(missing_positive_days)
        raise ValueError(
            "Some positive user/day labels do not exist in daily features. "
            f"Missing positive days: {len(missing_positive_days.index)}. "
            f"Sample: {sample}"
        )

    invalid_scenarios = positive_label_metadata[
        ~positive_label_metadata["scenario_id"].isin(EXPECTED_CERT_R42_SCENARIO_IDS)
    ]
    if not invalid_scenarios.empty:
        sample = invalid_scenarios[
            [*USER_DAY_COLUMNS, "scenario_id"]
        ].head(REPORT_SAMPLE_SIZE).to_dict(orient="records")
        raise ValueError(f"Invalid scenario ids in daily label metadata: {sample}")

    _validate_metadata_consistency(labels, positive_label_metadata)
    _validate_unique_user_day(positive_label_metadata, "daily label metadata")


def _validate_metadata_consistency(
    labels: pd.DataFrame,
    positive_label_metadata: pd.DataFrame
) -> None:
    label_lookup = labels[USER_DAY_COLUMNS + ["label"]]
    metadata_labels = positive_label_metadata.merge(
        label_lookup,
        on=USER_DAY_COLUMNS,
        how="left",
        suffixes=("_metadata", "_daily")
    )

    metadata_without_positive_label = metadata_labels[
        ~metadata_labels["label_daily"].eq(LABEL_ANOMALOUS)
    ]
    if not metadata_without_positive_label.empty:
        sample = _build_user_day_label_sample(
            metadata_without_positive_label.rename(columns={"label_metadata": "label"})
        )
        raise ValueError(
            "Some positive metadata user/day pairs do not have label=1 in daily labels. "
            f"Sample: {sample}"
        )

    negative_labels_in_metadata = labels.loc[
        labels["label"].eq(LABEL_NORMAL),
        USER_DAY_COLUMNS
    ].merge(
        positive_label_metadata[USER_DAY_COLUMNS],
        on=USER_DAY_COLUMNS,
        how="inner"
    )
    if not negative_labels_in_metadata.empty:
        sample = _build_user_day_label_sample(negative_labels_in_metadata)
        raise ValueError(
            "Some label=0 user/day pairs appear in positive metadata. "
            f"Sample: {sample}"
        )


def _validate_no_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    source_name: str
) -> None:
    for column in columns:
        if df[column].isna().any():
            raise ValueError(f"{source_name} contains missing values in column {column}.")


def _validate_unique_user_day(df: pd.DataFrame, source_name: str) -> None:
    duplicated_rows = df.duplicated(subset=USER_DAY_COLUMNS, keep=False)

    if not duplicated_rows.any():
        return

    duplicated_count = int(duplicated_rows.sum())
    sample = (
        df.loc[duplicated_rows, USER_DAY_COLUMNS]
        .head(REPORT_SAMPLE_SIZE)
        .to_dict(orient="records")
    )

    raise ValueError(
        f"{source_name} contains duplicated user/day pairs. "
        f"Duplicated rows: {duplicated_count}. "
        f"Sample: {sample}"
    )


def _empty_positive_label_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user",
            "day",
            "label",
            "scenario_id",
            "attack_start_day",
            "attack_end_day",
            "label_source",
        ]
    )


if __name__ == "__main__":
    main()
