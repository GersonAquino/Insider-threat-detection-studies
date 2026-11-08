import hashlib

from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from src.itd_tcnTransformer.features.build_daily_features import (
    EVENT_FEATURE_PREFIXES,
    HOUR_FEATURE_COLUMNS,
    HOUR_MISSING_SENTINEL,
    PROFILE_PSYCHOMETRIC_PREFIX,
)
from src.itd_tcnTransformer.config.paths import (
    DAILY_LABELS_DIR,
    FINAL_DAILY_DATASET_DIR,
    SCALERS_DIR,
    SEQUENCES_DIR,
    SPLITS_DIR,
)
from src.itd_tcnTransformer.config.settings import (
    ARTICLE_SPLIT_STRATIFY_BY_SCENARIO,
    ARTICLE_STAGE2_MALICIOUS_TEST_USERS,
    ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS,
    ARTICLE_STAGE2_NORMAL_TEST_USERS,
    ARTICLE_STAGE2_NORMAL_TRAIN_USERS,
    EXACT_DUPLICATE_FEATURE_KEEP_POLICY,
    EXACT_DUPLICATE_FEATURE_SELECTION_ENABLED,
    EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE,
    EXPECTED_CERT_R42_MALICIOUS_USERS,
    EXPECTED_CERT_R42_SCENARIO_IDS,
    EXPECTED_CERT_R42_TOTAL_USERS,
    FINETUNE_VALIDATION_RATIO,
    MAX_CONSECUTIVE_MISSING_DAYS,
    MAX_MISSING_DAY_RATIO,
    NORMAL_PRETRAIN_USER_COUNT,
    PREDICT_ALL_DAYS,
    PROFILE_FEATURE_MODE,
    RANDOM_SEED,
    SPLIT_STRATEGY,
    TEST_RATIO,
    TRAINING_MODE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    WINDOW_SIZE,
    WINDOW_STEP,
)
from src.itd_tcnTransformer.utils.io import save_json


FINAL_DAILY_DATASET_FILE = "final_daily_dataset.parquet"
DAILY_LABEL_METADATA_FILE = "daily_label_metadata.parquet"
SPLIT_METADATA_FILE = "split_metadata.json"
FEATURE_COLUMNS_FILE = "feature_columns.json"
SEQUENCES_METADATA_FILE = "sequences_metadata.json"
STANDARD_SCALER_FILE = "standard_scaler.joblib"
SPLIT_DAILY_DATASET_FILE_SUFFIX = "_daily.parquet"
SEQUENCE_X_FILE_SUFFIX = "_X.pt"
SEQUENCE_Y_FILE_SUFFIX = "_y.pt"
SEQUENCE_METADATA_FILE_SUFFIX = "_metadata.parquet"
METADATA_COLUMNS = ("user", "day")
TARGET_COLUMN = "label"
IDENTIFIER_COLUMNS = (*METADATA_COLUMNS, TARGET_COLUMN)
REQUIRED_COLUMNS: set[str] = set(IDENTIFIER_COLUMNS)
SPLIT_NAMES = ("train", "validation", "test")
ARTICLE_SPLIT_NAMES = (
    "normal_pretrain",
    "finetune_train",
    "finetune_validation",
    "test",
)
ARTICLE_SCALER_FIT_SPLIT_NAME = "normal_pretrain+finetune_train"
ARTICLE_FEATURE_VARIANCE_SCOPE = "normal_pretrain+finetune_train"
RATIO_SUM_TOLERANCE = 1e-9
FEATURE_VARIANCE_SCOPE = "train"
SCALER_FIT_SPLIT_NAME = "train"
FEATURE_VALUE_TRANSFORM = "standard_scaler"
CALENDAR_MISSING_DAY_COLUMN = "calendar_missing_day"
UNSCALED_FEATURE_COLUMNS = (CALENDAR_MISSING_DAY_COLUMN,)
HOUR_MISSING_FEATURE_COLUMNS = tuple(
    f"{column}_missing"
    for column in HOUR_FEATURE_COLUMNS
)
PROFILE_FEATURE_PREFIX = "profile_"
PROFILE_UNKNOWN_SUFFIX = "_unknown"
LABEL_METADATA_COLUMNS = (
    "scenario_id",
    "malicious_user",
    "attack_start",
    "attack_end",
    "attack_start_day",
    "attack_end_day",
    "label_source",
)
EXACT_DUPLICATE_CANONICAL_FEATURE_PRIORITY = (
    "daily_total_event_count",
    "logon_total_events",
    "device_total_events",
    "file_total_events",
    "email_total_events",
    "http_total_events",
    "logon_unique_pc_count",
    "device_unique_pc_count",
    "file_unique_pc_count",
    "email_total_recipient_count",
    "email_external_recipients_count",
    "email_with_external_recipients_count",
    "http_unique_domains_total",
)
EXACT_DUPLICATE_BASE_NAME_PRIORITY = (
    "_total_events",
    "_total_event_count",
    "_count",
    "_total",
)
DERIVED_FEATURE_NAME_MARKERS = (
    "_ratio",
    "_rolling_",
    "_vs_rolling_",
    "_vs_user_",
    "_new_",
    "_flag",
    "_high_risk_",
    "_active_modality_",
    "_behavioral_intensity_",
    "_multi_domain_",
)

__all__ = ["main"]


@dataclass
class SplitRatios:
    train: float
    validation: float
    test: float


@dataclass
class WindowPolicy:
    window_size: int
    window_step: int
    policy: str
    eligible_window_day_range_meaning: str


@dataclass
class DatasetSummary:
    total_rows: int
    observed_rows: int
    calendar_missing_rows: int
    total_users: int
    total_calendar_days: int
    total_observed_days: int
    min_day: pd.Timestamp
    max_day: pd.Timestamp


@dataclass
class SplitSummary:
    start_day: pd.Timestamp
    end_day: pd.Timestamp
    days: int
    rows: int
    observed_rows: int
    calendar_missing_rows: int
    users: int
    positive_rows: int
    negative_rows: int
    eligible_window_start_day: pd.Timestamp
    eligible_window_end_day: pd.Timestamp
    eligible_window_count: int


@dataclass
class SplitsSummary:
    train: SplitSummary
    validation: SplitSummary
    test: SplitSummary


@dataclass
class SplitMetadata:
    source_file: str
    strategy: str
    split_ratios: SplitRatios
    window: WindowPolicy
    dataset: DatasetSummary
    splits: SplitsSummary


@dataclass
class ArticleFeatureScalerPolicy:
    scaler_fit_groups: list[str]
    feature_variance_scope: str


@dataclass
class ArticleWindowPolicy:
    synthetic_calendar_days_enabled: bool
    window_unit: str
    require_consecutive_calendar_days: bool


@dataclass
class ArticleSplitMetadata:
    source_file: str
    label_metadata_file: str
    strategy: str
    training_mode: str
    random_seed: int
    user_counts: dict[str, int]
    malicious_users_by_scenario: dict[str, dict[str, int]]
    sequence_counts: dict[str, "SequenceSplitSummary"]
    sequence_split_names: list[str]
    feature_scaler_policy: ArticleFeatureScalerPolicy
    window_policy: ArticleWindowPolicy
    warnings: list[str]


@dataclass
class ArticleUserGroups:
    normal_users: list[str]
    malicious_users: list[str]
    malicious_user_scenarios: dict[str, int]
    multi_scenario_users: dict[str, list[int]]


@dataclass
class ArticleUserSplit:
    normal_pretrain_users: list[str]
    finetune_train_normal_users: list[str]
    finetune_validation_normal_users: list[str]
    test_normal_users: list[str]
    finetune_train_malicious_users: list[str]
    finetune_validation_malicious_users: list[str]
    test_malicious_users: list[str]


@dataclass
class FeatureColumnCounts:
    candidate_feature_columns: int
    removed_by_profile_mode_columns: int
    removed_zero_variance_columns: int
    removed_exact_duplicate_columns: int
    scaled_feature_columns: int
    unscaled_feature_columns: int
    feature_columns: int


@dataclass
class ExactDuplicateFeatureGroup:
    kept_column: str
    removed_columns: list[str]
    duplicate_columns: list[str]
    selection_scope: str
    keep_reason: str


@dataclass
class FeatureColumnsReport:
    source_file: str
    variance_scope: str
    scaler_file: str
    scaler_fit_split: str
    feature_value_transform: str
    profile_feature_mode: str
    exact_duplicate_feature_selection_enabled: bool
    exact_duplicate_feature_selection_scope: str
    exact_duplicate_feature_keep_policy: str
    metadata_columns: list[str]
    target_column: str
    excluded_identifier_columns: list[str]
    candidate_feature_columns: list[str]
    removed_by_profile_mode_columns: list[str]
    removed_zero_variance_columns: list[str]
    removed_exact_duplicate_columns: list[str]
    exact_duplicate_feature_groups: list[ExactDuplicateFeatureGroup]
    scaled_feature_columns: list[str]
    unscaled_feature_columns: list[str]
    feature_columns: list[str]
    feature_group_counts: dict[str, int]
    behavioral_derived_feature_columns: list[str]
    counts: FeatureColumnCounts


@dataclass
class SequenceSplit:
    X: torch.Tensor
    y: torch.Tensor
    metadata: pd.DataFrame
    candidate_windows: int
    rejected_windows: int
    rejection_reasons: dict[str, int]


@dataclass
class SequenceSplitSummary:
    candidate_windows: int
    created_windows: int
    rejected_windows: int
    y_positive_count: int
    y_negative_count: int
    X_shape: list[int]
    y_shape: list[int]
    n_features: int
    rejection_reasons: dict[str, int]


@dataclass
class SequencesMetadata:
    window_size: int
    window_step: int
    predict_all_days: bool
    synthetic_calendar_days_enabled: bool
    window_unit: str
    require_consecutive_calendar_days: bool
    max_missing_day_ratio: float
    max_consecutive_missing_days: int
    total_candidate_windows: int
    total_created_windows: int
    total_rejected_windows: int
    splits: dict[str, SequenceSplitSummary]


def main() -> None:
    if SPLIT_STRATEGY == "temporal":
        _build_temporal_sequences()
        return

    if SPLIT_STRATEGY == "user_article_two_phase":
        _build_user_article_two_phase_sequences()
        return

    raise ValueError(f"Unsupported SPLIT_STRATEGY: {SPLIT_STRATEGY}")


def _build_temporal_sequences() -> None:
    print("Building temporal sequence split metadata ...")
    _validate_split_ratios()

    final_daily_dataset_path = FINAL_DAILY_DATASET_DIR / FINAL_DAILY_DATASET_FILE
    final_daily_dataset = _load_final_daily_dataset(final_daily_dataset_path)
    _validate_required_columns(final_daily_dataset)
    final_daily_dataset = final_daily_dataset.copy()
    final_daily_dataset["day"] = pd.to_datetime(
        final_daily_dataset["day"]
    ).dt.floor("D")

    calendar_days = _build_complete_calendar_days(final_daily_dataset)
    split_days = _build_split_day_ranges(calendar_days)
    _validate_split_days(split_days)
    split_datasets = _build_split_datasets(final_daily_dataset, split_days)
    split_datasets = _fill_missing_calendar_days_for_splits(split_datasets)
    final_daily_dataset = pd.concat(
        [split_datasets[split_name] for split_name in SPLIT_NAMES],
        ignore_index=True
    )

    split_metadata = _build_temporal_split_metadata(
        final_daily_dataset,
        str(final_daily_dataset_path),
        calendar_days,
        split_days
    )

    feature_columns_report = _build_feature_columns_report(
        final_daily_dataset,
        split_datasets["train"],
        str(final_daily_dataset_path),
    )
    standard_scaler = _fit_standard_scaler_on_train_features(
        split_datasets["train"],
        feature_columns_report.scaled_feature_columns,
    )
    split_datasets = _normalize_split_datasets(
        split_datasets,
        feature_columns_report.scaled_feature_columns,
        standard_scaler,
    )
    _validate_normalized_feature_values(
        split_datasets,
        feature_columns_report.feature_columns
    )
    sequence_splits = _build_all_sequence_splits(
        split_datasets,
        feature_columns_report.feature_columns,
    )
    sequences_metadata = _build_sequences_metadata(
        sequence_splits,
        required_positive_splits=set(SPLIT_NAMES),
        synthetic_calendar_days_enabled=True,
        window_unit="calendar_days",
        require_consecutive_calendar_days=True,
    )

    _save_split_daily_datasets(split_datasets)
    _save_split_metadata(split_metadata)
    _save_feature_columns_report(feature_columns_report)
    _save_standard_scaler(standard_scaler)
    _save_sequence_splits(sequence_splits)
    _save_sequences_metadata(sequences_metadata)


def _build_user_article_two_phase_sequences() -> None:
    print("Building user article two-phase sequence split metadata ...")

    final_daily_dataset_path = FINAL_DAILY_DATASET_DIR / FINAL_DAILY_DATASET_FILE
    final_daily_dataset = _load_final_daily_dataset(final_daily_dataset_path)
    _validate_required_columns(final_daily_dataset)
    final_daily_dataset = final_daily_dataset.copy()
    final_daily_dataset["day"] = pd.to_datetime(
        final_daily_dataset["day"]
    ).dt.floor("D")

    daily_label_metadata_path = DAILY_LABELS_DIR / DAILY_LABEL_METADATA_FILE
    daily_label_metadata = _load_daily_label_metadata(daily_label_metadata_path)
    article_user_groups = _load_article_user_groups(
        final_daily_dataset,
        daily_label_metadata,
    )
    article_user_split = _build_article_user_split(article_user_groups)
    split_datasets, split_user_group_maps = _build_user_group_datasets(
        final_daily_dataset,
        article_user_split,
    )
    split_datasets = _add_observed_only_calendar_missing_column(split_datasets)

    scaler_fit_dataset = pd.concat(
        [
            split_datasets["normal_pretrain"],
            split_datasets["finetune_train"],
        ],
        ignore_index=True,
    )
    feature_columns_report = _build_feature_columns_report(
        scaler_fit_dataset,
        scaler_fit_dataset,
        str(final_daily_dataset_path),
        variance_scope=ARTICLE_FEATURE_VARIANCE_SCOPE,
        scaler_fit_split=ARTICLE_SCALER_FIT_SPLIT_NAME,
    )
    standard_scaler = _fit_standard_scaler_on_train_features(
        scaler_fit_dataset,
        feature_columns_report.scaled_feature_columns,
    )
    split_datasets = _normalize_split_datasets(
        split_datasets,
        feature_columns_report.scaled_feature_columns,
        standard_scaler,
    )
    _validate_normalized_feature_values(
        split_datasets,
        feature_columns_report.feature_columns
    )
    sequence_splits = _build_all_sequence_splits(
        split_datasets,
        feature_columns_report.feature_columns,
        split_user_group_maps=split_user_group_maps,
        require_consecutive_calendar_days=False,
    )
    required_positive_splits = {
        "finetune_train",
        "finetune_validation",
        "test",
    }
    sequences_metadata = _build_sequences_metadata(
        sequence_splits,
        required_positive_splits=required_positive_splits,
        synthetic_calendar_days_enabled=False,
        window_unit="consecutive_observed_user_day_rows",
        require_consecutive_calendar_days=False,
    )
    split_metadata = _build_article_split_metadata(
        source_file=str(final_daily_dataset_path),
        label_metadata_file=str(daily_label_metadata_path),
        user_split=article_user_split,
        malicious_user_scenarios=article_user_groups.malicious_user_scenarios,
        multi_scenario_users=article_user_groups.multi_scenario_users,
        sequence_splits=sequence_splits,
    )

    _save_split_daily_datasets(split_datasets)
    _save_split_metadata(split_metadata)
    _save_feature_columns_report(feature_columns_report)
    _save_standard_scaler(standard_scaler)
    _save_sequence_splits(sequence_splits)
    _save_sequences_metadata(sequences_metadata)


def _load_daily_label_metadata(file_path: Path) -> pd.DataFrame:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Daily label metadata file not found: {file_path}")

    print(f"Loading daily label metadata from {file_path} ...")

    metadata = pd.read_parquet(file_path)
    required_columns = {
        "user",
        "day",
        "label",
        "scenario_id",
        "attack_start_day",
        "attack_end_day",
        "label_source",
    }
    missing_columns = sorted(required_columns - set(metadata.columns))

    if missing_columns:
        raise ValueError(
            "Daily label metadata is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    metadata = metadata.copy()
    metadata["user"] = metadata["user"].astype("string").str.strip()
    metadata["day"] = pd.to_datetime(metadata["day"]).dt.floor("D")
    metadata["scenario_id"] = metadata["scenario_id"].astype("int64")

    return metadata


def _load_article_user_groups(
    final_daily_dataset: pd.DataFrame,
    daily_label_metadata: pd.DataFrame,
) -> ArticleUserGroups:
    all_users = sorted(str(user) for user in final_daily_dataset["user"].unique())
    all_user_set = set(all_users)
    malicious_users = sorted(str(user) for user in daily_label_metadata["user"].unique())
    malicious_user_set = set(malicious_users)
    missing_malicious_users = sorted(malicious_user_set - all_user_set)

    if missing_malicious_users:
        raise ValueError(
            "Some malicious users from daily_label_metadata do not exist in "
            "final_daily_dataset. "
            f"Missing users: {missing_malicious_users[:10]}"
        )

    normal_users = sorted(all_user_set - malicious_user_set)
    scenario_counts = set(daily_label_metadata["scenario_id"].unique().tolist())

    if len(malicious_users) != EXPECTED_CERT_R42_MALICIOUS_USERS:
        raise ValueError(
            "Unexpected malicious user count for article split. "
            f"Expected {EXPECTED_CERT_R42_MALICIOUS_USERS}, "
            f"found {len(malicious_users)}."
        )

    expected_normal_users = (
        EXPECTED_CERT_R42_TOTAL_USERS -
        EXPECTED_CERT_R42_MALICIOUS_USERS
    )
    if len(normal_users) != expected_normal_users:
        raise ValueError(
            "Unexpected normal user count for article split. "
            f"Expected {expected_normal_users}, found {len(normal_users)}."
        )

    if scenario_counts != set(EXPECTED_CERT_R42_SCENARIO_IDS):
        raise ValueError(
            "Unexpected scenario ids in daily_label_metadata. "
            f"Expected {sorted(EXPECTED_CERT_R42_SCENARIO_IDS)}, "
            f"found {sorted(scenario_counts)}."
        )

    scenarios_by_user = (
        daily_label_metadata
        .groupby("user")["scenario_id"]
        .apply(lambda values: sorted({int(value) for value in values}))
        .to_dict()
    )
    multi_scenario_users = {
        str(user): scenario_ids
        for user, scenario_ids in scenarios_by_user.items()
        if len(scenario_ids) > 1
    }
    scenario_by_user = (
        daily_label_metadata
        .sort_values(["user", "scenario_id"])
        .drop_duplicates("user", keep="first")
        .set_index("user")["scenario_id"]
        .astype("int64")
        .to_dict()
    )

    return ArticleUserGroups(
        normal_users=normal_users,
        malicious_users=malicious_users,
        malicious_user_scenarios={str(user): int(scenario) for user, scenario in scenario_by_user.items()},
        multi_scenario_users=multi_scenario_users,
    )


def _build_article_user_split(
    article_user_groups: ArticleUserGroups,
) -> ArticleUserSplit:
    rng = np.random.default_rng(RANDOM_SEED)
    normal_users = _shuffle_users(article_user_groups.normal_users, rng)

    normal_pretrain_users = normal_users[:NORMAL_PRETRAIN_USER_COUNT]
    stage2_normal_users = normal_users[NORMAL_PRETRAIN_USER_COUNT:]

    if len(stage2_normal_users) != (
        ARTICLE_STAGE2_NORMAL_TRAIN_USERS +
        ARTICLE_STAGE2_NORMAL_TEST_USERS
    ):
        raise ValueError(
            "Unexpected number of stage-2 normal users after pretrain selection. "
            f"Found {len(stage2_normal_users)}."
        )

    test_normal_users = stage2_normal_users[:ARTICLE_STAGE2_NORMAL_TEST_USERS]
    stage2_train_normal_users = stage2_normal_users[ARTICLE_STAGE2_NORMAL_TEST_USERS:]
    (
        finetune_train_normal_users,
        finetune_validation_normal_users,
    ) = _split_users_for_validation(stage2_train_normal_users, rng)

    if ARTICLE_SPLIT_STRATIFY_BY_SCENARIO:
        (
            stage2_train_malicious_users,
            test_malicious_users,
        ) = _split_malicious_users_by_scenario(
            article_user_groups.malicious_user_scenarios,
            ARTICLE_STAGE2_MALICIOUS_TEST_USERS,
            rng,
        )
        (
            finetune_train_malicious_users,
            finetune_validation_malicious_users,
        ) = _split_malicious_users_by_scenario(
            {
                user: article_user_groups.malicious_user_scenarios[user]
                for user in stage2_train_malicious_users
            },
            _calculate_validation_user_count(stage2_train_malicious_users),
            rng,
        )
    else:
        malicious_users = _shuffle_users(article_user_groups.malicious_users, rng)
        test_malicious_users = malicious_users[:ARTICLE_STAGE2_MALICIOUS_TEST_USERS]
        stage2_train_malicious_users = malicious_users[
            ARTICLE_STAGE2_MALICIOUS_TEST_USERS:
        ]
        (
            finetune_train_malicious_users,
            finetune_validation_malicious_users,
        ) = _split_users_for_validation(stage2_train_malicious_users, rng)

    user_split = ArticleUserSplit(
        normal_pretrain_users=sorted(normal_pretrain_users),
        finetune_train_normal_users=sorted(finetune_train_normal_users),
        finetune_validation_normal_users=sorted(finetune_validation_normal_users),
        test_normal_users=sorted(test_normal_users),
        finetune_train_malicious_users=sorted(finetune_train_malicious_users),
        finetune_validation_malicious_users=sorted(finetune_validation_malicious_users),
        test_malicious_users=sorted(test_malicious_users),
    )
    _validate_article_user_split(user_split)

    return user_split


def _build_user_group_datasets(
    final_daily_dataset: pd.DataFrame,
    user_split: ArticleUserSplit,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    split_users = {
        "normal_pretrain": user_split.normal_pretrain_users,
        "finetune_train": (
            user_split.finetune_train_normal_users +
            user_split.finetune_train_malicious_users
        ),
        "finetune_validation": (
            user_split.finetune_validation_normal_users +
            user_split.finetune_validation_malicious_users
        ),
        "test": user_split.test_normal_users + user_split.test_malicious_users,
    }
    split_user_group_maps = {
        "normal_pretrain": {
            user: "normal_pretrain"
            for user in user_split.normal_pretrain_users
        },
        "finetune_train": {
            **{
                user: "normal_finetune_train"
                for user in user_split.finetune_train_normal_users
            },
            **{
                user: "malicious_finetune_train"
                for user in user_split.finetune_train_malicious_users
            },
        },
        "finetune_validation": {
            **{
                user: "normal_finetune_validation"
                for user in user_split.finetune_validation_normal_users
            },
            **{
                user: "malicious_finetune_validation"
                for user in user_split.finetune_validation_malicious_users
            },
        },
        "test": {
            **{
                user: "normal_test"
                for user in user_split.test_normal_users
            },
            **{
                user: "malicious_test"
                for user in user_split.test_malicious_users
            },
        },
    }
    split_datasets: dict[str, pd.DataFrame] = {}

    for split_name, users in split_users.items():
        split_dataset = final_daily_dataset[
            final_daily_dataset["user"].isin(users)
        ].copy()

        if split_dataset.empty:
            raise ValueError(f"Article split '{split_name}' has no rows.")

        split_datasets[split_name] = split_dataset.sort_values(["user", "day"])

    return split_datasets, split_user_group_maps


def _build_article_split_metadata(
    source_file: str,
    label_metadata_file: str,
    user_split: ArticleUserSplit,
    malicious_user_scenarios: dict[str, int],
    multi_scenario_users: dict[str, list[int]],
    sequence_splits: dict[str, SequenceSplit],
) -> ArticleSplitMetadata:
    return ArticleSplitMetadata(
        source_file=source_file,
        label_metadata_file=label_metadata_file,
        strategy=SPLIT_STRATEGY,
        training_mode=TRAINING_MODE,
        random_seed=int(RANDOM_SEED),
        user_counts=_build_article_user_counts(user_split),
        malicious_users_by_scenario=_build_article_scenario_counts(
            user_split,
            malicious_user_scenarios,
        ),
        sequence_counts=_build_sequence_split_summaries(sequence_splits),
        sequence_split_names=list(sequence_splits.keys()),
        feature_scaler_policy=ArticleFeatureScalerPolicy(
            scaler_fit_groups=["normal_pretrain", "finetune_train"],
            feature_variance_scope=ARTICLE_FEATURE_VARIANCE_SCOPE,
        ),
        window_policy=ArticleWindowPolicy(
            synthetic_calendar_days_enabled=False,
            window_unit="consecutive_observed_user_day_rows",
            require_consecutive_calendar_days=False,
        ),
        warnings=_build_article_split_warnings(
            user_split,
            malicious_user_scenarios,
            multi_scenario_users,
        ),
    )


def _allocate_stratified_counts(
    scenario_counts: dict[int, int],
    target_count: int,
) -> dict[int, int]:
    total_users = sum(scenario_counts.values())
    if target_count <= 0 or target_count >= total_users:
        raise ValueError(
            "target_count must be positive and lower than total users. "
            f"target_count={target_count}, total_users={total_users}."
        )

    raw_allocations = {
        scenario_id: scenario_count / total_users * target_count
        for scenario_id, scenario_count in scenario_counts.items()
    }
    allocations = {
        scenario_id: int(np.floor(raw_count))
        for scenario_id, raw_count in raw_allocations.items()
    }

    if target_count >= len(scenario_counts):
        for scenario_id in allocations:
            if allocations[scenario_id] == 0:
                allocations[scenario_id] = 1

    while sum(allocations.values()) > target_count:
        scenario_id = max(
            allocations,
            key=lambda key: (allocations[key] - raw_allocations[key], allocations[key])
        )
        allocations[scenario_id] -= 1

    while sum(allocations.values()) < target_count:
        scenario_id = max(
            allocations,
            key=lambda key: raw_allocations[key] - allocations[key]
        )
        allocations[scenario_id] += 1

    for scenario_id, allocation in allocations.items():
        if allocation >= scenario_counts[scenario_id]:
            raise ValueError(
                "Stratified allocation would consume an entire scenario group. "
                f"scenario_id={scenario_id}, allocation={allocation}, "
                f"available={scenario_counts[scenario_id]}."
            )

    return allocations


def _build_article_scenario_counts(
    user_split: ArticleUserSplit,
    malicious_user_scenarios: dict[str, int],
) -> dict[str, dict[str, int]]:
    scenario_groups = {
        "total": (
            user_split.finetune_train_malicious_users +
            user_split.finetune_validation_malicious_users +
            user_split.test_malicious_users
        ),
        "finetune_train": user_split.finetune_train_malicious_users,
        "finetune_validation": user_split.finetune_validation_malicious_users,
        "test": user_split.test_malicious_users,
    }

    return {
        group_name: _count_users_by_scenario(users, malicious_user_scenarios)
        for group_name, users in scenario_groups.items()
    }


def _build_article_split_warnings(
    user_split: ArticleUserSplit,
    malicious_user_scenarios: dict[str, int],
    multi_scenario_users: dict[str, list[int]],
) -> list[str]:
    warnings: list[str] = []

    if multi_scenario_users:
        sample = dict(list(sorted(multi_scenario_users.items()))[:10])
        warnings.append(
            "Some malicious users appear in more than one scenario. The lowest "
            f"scenario_id is used for stratification. Sample: {sample}."
        )

    for group_name, users in {
        "finetune_train_malicious": user_split.finetune_train_malicious_users,
        "finetune_validation_malicious": user_split.finetune_validation_malicious_users,
        "test_malicious": user_split.test_malicious_users,
    }.items():
        scenarios = {
            malicious_user_scenarios[user]
            for user in users
        }
        if scenarios != set(EXPECTED_CERT_R42_SCENARIO_IDS):
            warnings.append(
                f"{group_name} does not contain users from all expected scenarios. "
                f"Found scenarios: {sorted(scenarios)}."
            )

    return warnings


def _build_article_user_counts(user_split: ArticleUserSplit) -> dict[str, int]:
    return {
        "normal_pretrain_users": len(user_split.normal_pretrain_users),
        "finetune_train_normal_users": len(user_split.finetune_train_normal_users),
        "finetune_validation_normal_users": len(user_split.finetune_validation_normal_users),
        "test_normal_users": len(user_split.test_normal_users),
        "finetune_train_malicious_users": len(user_split.finetune_train_malicious_users),
        "finetune_validation_malicious_users": len(user_split.finetune_validation_malicious_users),
        "test_malicious_users": len(user_split.test_malicious_users),
    }


def _calculate_validation_user_count(users: list[str]) -> int:
    validation_user_count = int(round(len(users) * FINETUNE_VALIDATION_RATIO))

    return max(1, min(validation_user_count, len(users) - 1))


def _count_users_by_scenario(
    users: list[str],
    malicious_user_scenarios: dict[str, int],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for user in users:
        scenario_id = str(int(malicious_user_scenarios[user]))
        counts[scenario_id] = counts.get(scenario_id, 0) + 1

    return dict(sorted(counts.items()))


def _shuffle_users(users: list[str], rng: np.random.Generator) -> list[str]:
    shuffled_users = list(users)
    rng.shuffle(shuffled_users)

    return shuffled_users


def _split_malicious_users_by_scenario(
    malicious_user_scenarios: dict[str, int],
    target_second_group_count: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    scenario_to_users: dict[int, list[str]] = {}
    for user, scenario_id in malicious_user_scenarios.items():
        scenario_to_users.setdefault(int(scenario_id), []).append(str(user))

    scenario_counts = {
        scenario_id: len(users)
        for scenario_id, users in scenario_to_users.items()
    }
    second_group_counts = _allocate_stratified_counts(
        scenario_counts,
        target_second_group_count,
    )
    first_group_users: list[str] = []
    second_group_users: list[str] = []

    for scenario_id in sorted(scenario_to_users):
        users = _shuffle_users(sorted(scenario_to_users[scenario_id]), rng)
        split_count = second_group_counts[scenario_id]
        second_group_users.extend(users[:split_count])
        first_group_users.extend(users[split_count:])

    return sorted(first_group_users), sorted(second_group_users)


def _split_users_for_validation(
    users: list[str],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    shuffled_users = _shuffle_users(users, rng)
    validation_user_count = _calculate_validation_user_count(shuffled_users)

    validation_users = shuffled_users[:validation_user_count]
    train_users = shuffled_users[validation_user_count:]

    return sorted(train_users), sorted(validation_users)


def _validate_article_user_split(user_split: ArticleUserSplit) -> None:
    groups = {
        "normal_pretrain": user_split.normal_pretrain_users,
        "finetune_train_normal": user_split.finetune_train_normal_users,
        "finetune_validation_normal": user_split.finetune_validation_normal_users,
        "test_normal": user_split.test_normal_users,
        "finetune_train_malicious": user_split.finetune_train_malicious_users,
        "finetune_validation_malicious": user_split.finetune_validation_malicious_users,
        "test_malicious": user_split.test_malicious_users,
    }
    assigned_users: list[str] = []
    for users in groups.values():
        assigned_users.extend(users)

    if len(assigned_users) != len(set(assigned_users)):
        raise ValueError("Article user split assigns at least one user to multiple groups.")

    if len(assigned_users) != EXPECTED_CERT_R42_TOTAL_USERS:
        raise ValueError(
            "Article user split must assign all CERT r4.2 users. "
            f"Expected {EXPECTED_CERT_R42_TOTAL_USERS}, found {len(assigned_users)}."
        )

    expected_group_sizes = {
        "normal_pretrain": NORMAL_PRETRAIN_USER_COUNT,
        "test_normal": ARTICLE_STAGE2_NORMAL_TEST_USERS,
        "test_malicious": ARTICLE_STAGE2_MALICIOUS_TEST_USERS,
    }

    for group_name, expected_size in expected_group_sizes.items():
        actual_size = len(groups[group_name])
        if actual_size != expected_size:
            raise ValueError(
                f"Article user split group '{group_name}' has wrong size. "
                f"Expected {expected_size}, found {actual_size}."
            )

    if (
        len(user_split.finetune_train_malicious_users) +
        len(user_split.finetune_validation_malicious_users)
        != ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS
    ):
        raise ValueError(
            "finetune_train_malicious + finetune_validation_malicious must sum "
            f"to {ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS} users."
        )

    if (
        len(user_split.finetune_train_normal_users) +
        len(user_split.finetune_validation_normal_users)
        != ARTICLE_STAGE2_NORMAL_TRAIN_USERS
    ):
        raise ValueError(
            "finetune_train_normal + finetune_validation_normal must sum "
            f"to {ARTICLE_STAGE2_NORMAL_TRAIN_USERS} users."
        )
def _load_final_daily_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Final daily dataset file not found: {file_path}")

    print(f"Loading final daily dataset from {file_path} ...")

    return pd.read_parquet(file_path)


def _validate_required_columns(dataset: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataset.columns))

    if missing_columns:
        raise ValueError(
            "Final daily dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )


def _validate_split_ratios() -> None:
    ratio_sum = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO

    if abs(ratio_sum - 1.0) > RATIO_SUM_TOLERANCE:
        raise ValueError(
            "Train, validation and test ratios must sum to 1. "
            f"Current sum: {ratio_sum}. "
            f"Values: train={TRAIN_RATIO}, "
            f"validation={VALIDATION_RATIO}, "
            f"test={TEST_RATIO}"
        )


def _build_complete_calendar_days(dataset: pd.DataFrame) -> list[pd.Timestamp]:
    min_day = pd.Timestamp(dataset["day"].min()).floor("D")
    max_day = pd.Timestamp(dataset["day"].max()).floor("D")

    return list(pd.date_range(min_day, max_day, freq="D"))


def _build_temporal_split_metadata(
    dataset: pd.DataFrame,
    source_file: str,
    calendar_days: list[pd.Timestamp],
    split_days: dict[str, list[pd.Timestamp]]
) -> SplitMetadata:
    split_ratios = SplitRatios(
        train=TRAIN_RATIO,
        validation=VALIDATION_RATIO,
        test=TEST_RATIO,
    )

    window = WindowPolicy(
        window_size=int(WINDOW_SIZE),
        window_step=int(WINDOW_STEP),
        policy=(
            "Splits are defined over the complete calendar range between "
            "min(day) and max(day), including days without log rows. Future "
            "windows must be assigned to a split only when all days in the "
            "window are inside that same split."
        ),
        eligible_window_day_range_meaning=(
            "eligible_window_start_day and eligible_window_end_day are "
            "valid window start days for full windows contained in the split."
        ),
    )

    calendar_missing_rows = _count_calendar_missing_rows(dataset)
    observed_days = dataset.loc[
        ~dataset[CALENDAR_MISSING_DAY_COLUMN],
        "day"
    ].drop_duplicates()

    dataset_summary = DatasetSummary(
        total_rows=int(len(dataset.index)),
        observed_rows=int(len(dataset.index) - calendar_missing_rows),
        calendar_missing_rows=calendar_missing_rows,
        total_users=int(dataset["user"].nunique()),
        total_calendar_days=int(len(calendar_days)),
        total_observed_days=int(len(observed_days.index)),
        min_day=min(calendar_days),
        max_day=max(calendar_days),
    )

    splits = SplitsSummary(
        train=_build_split_summary("train", dataset, split_days["train"]),
        validation=_build_split_summary("validation", dataset, split_days["validation"]),
        test=_build_split_summary("test", dataset, split_days["test"]),
    )

    return SplitMetadata(
        source_file=source_file,
        strategy="temporal_complete_calendar_day",
        split_ratios=split_ratios,
        window=window,
        dataset=dataset_summary,
        splits=splits,
    )


def _build_split_day_ranges(
    unique_days: list[pd.Timestamp]
) -> dict[str, list[pd.Timestamp]]:
    total_days = len(unique_days)
    train_end_index = int(total_days * TRAIN_RATIO)
    validation_end_index = int(total_days * (TRAIN_RATIO + VALIDATION_RATIO))

    return {
        "train": unique_days[:train_end_index],
        "validation": unique_days[train_end_index:validation_end_index],
        "test": unique_days[validation_end_index:],
    }


def _build_split_datasets(
    dataset: pd.DataFrame,
    split_days: dict[str, list[pd.Timestamp]]
) -> dict[str, pd.DataFrame]:
    return {
        split_name: dataset[dataset["day"].isin(split_days[split_name])]
        for split_name in SPLIT_NAMES
    }


def _fill_missing_calendar_days_for_splits(
    split_datasets: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    return {
        split_name: _fill_missing_calendar_days_for_split(
            split_name,
            split_dataset
        )
        for split_name, split_dataset in split_datasets.items()
    }


def _add_observed_only_calendar_missing_column(
    split_datasets: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    observed_split_datasets: dict[str, pd.DataFrame] = {}

    for split_name, split_dataset in split_datasets.items():
        observed_split_dataset = split_dataset.copy()
        observed_split_dataset[CALENDAR_MISSING_DAY_COLUMN] = False
        observed_split_datasets[split_name] = (
            observed_split_dataset
            .sort_values(["user", "day"])
            .reset_index(drop=True)
        )

    return observed_split_datasets


def _fill_missing_calendar_days_for_split(
    split_name: str,
    split_dataset: pd.DataFrame
) -> pd.DataFrame:
    print(f"Filling missing calendar days for {split_name} split ...")

    filled_user_parts: list[pd.DataFrame] = []

    for user, user_dataset in split_dataset.groupby("user", sort=True):
        filled_user_parts.append(
            _fill_missing_calendar_days_for_user(str(user), user_dataset)
        )

    if not filled_user_parts:
        raise ValueError(f"Temporal split '{split_name}' has no user rows.")

    return (
        pd.concat(filled_user_parts, ignore_index=True)
        .sort_values(["user", "day"])
        .reset_index(drop=True)
    )


def _fill_missing_calendar_days_for_user(
    user: str,
    user_dataset: pd.DataFrame
) -> pd.DataFrame:
    user_dataset = user_dataset.copy()
    user_dataset["day"] = pd.to_datetime(user_dataset["day"]).dt.floor("D")
    user_dataset = user_dataset.sort_values("day")

    min_user_day = pd.Timestamp(user_dataset["day"].min()).floor("D")
    max_user_day = pd.Timestamp(user_dataset["day"].max()).floor("D")
    full_user_days = pd.date_range(min_user_day, max_user_day, freq="D")

    filled_user_dataset = (
        user_dataset
        .set_index("day")
        .reindex(full_user_days)
    )
    filled_user_dataset.index.name = "day"

    synthetic_day_mask = filled_user_dataset["user"].isna()
    filled_user_dataset["user"] = user
    filled_user_dataset[CALENDAR_MISSING_DAY_COLUMN] = synthetic_day_mask

    if synthetic_day_mask.any():
        _fill_synthetic_calendar_day_rows(
            filled_user_dataset,
            synthetic_day_mask,
            user_dataset,
        )

    filled_user_dataset[CALENDAR_MISSING_DAY_COLUMN] = (
        filled_user_dataset[CALENDAR_MISSING_DAY_COLUMN].astype("bool")
    )

    return filled_user_dataset.reset_index()


def _fill_synthetic_calendar_day_rows(
    filled_user_dataset: pd.DataFrame,
    synthetic_day_mask: pd.Series,
    original_user_dataset: pd.DataFrame,
) -> None:
    synthetic_days = pd.DatetimeIndex(filled_user_dataset.index[synthetic_day_mask])
    original_dtypes = original_user_dataset.dtypes

    for column in filled_user_dataset.columns:
        if column in ("user", CALENDAR_MISSING_DAY_COLUMN):
            continue

        if column == TARGET_COLUMN:
            filled_user_dataset.loc[synthetic_day_mask, column] = 0
            continue

        if column in HOUR_FEATURE_COLUMNS:
            filled_user_dataset.loc[synthetic_day_mask, column] = HOUR_MISSING_SENTINEL
            continue

        if column in HOUR_MISSING_FEATURE_COLUMNS:
            filled_user_dataset.loc[synthetic_day_mask, column] = True
            continue

        if column.startswith(EVENT_FEATURE_PREFIXES):
            fill_value = (
                False
                if pd.api.types.is_bool_dtype(original_dtypes[column])
                else 0
            )
            filled_user_dataset.loc[synthetic_day_mask, column] = fill_value
            continue

        if column.startswith(PROFILE_PSYCHOMETRIC_PREFIX):
            continue

        if column.startswith(PROFILE_FEATURE_PREFIX):
            continue

        raise ValueError(
            "No fill rule is defined for synthetic calendar day column: "
            f"{column}"
        )

    _fill_synthetic_profile_columns(
        filled_user_dataset,
        synthetic_days,
        original_user_dataset,
    )
    _restore_synthetic_calendar_fill_dtypes(
        filled_user_dataset,
        original_dtypes,
    )


def _fill_synthetic_profile_columns(
    filled_user_dataset: pd.DataFrame,
    synthetic_days: pd.DatetimeIndex,
    original_user_dataset: pd.DataFrame,
) -> None:
    profile_columns = [
        column
        for column in filled_user_dataset.columns
        if column.startswith(PROFILE_FEATURE_PREFIX)
    ]

    if not profile_columns:
        return

    profile_one_hot_columns = [
        column
        for column in profile_columns
        if not column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)
    ]
    psychometric_columns = [
        column
        for column in profile_columns
        if column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)
    ]
    unknown_profile_values = _build_unknown_profile_one_hot_values(
        profile_one_hot_columns
    )

    for day in synthetic_days:
        profile_values = _get_profile_values_for_day(
            original_user_dataset,
            day,
            profile_columns,
        )

        if profile_values is None:
            if psychometric_columns:
                raise ValueError(
                    "Cannot fill synthetic calendar day psychometric columns "
                    "without a known profile for the user."
                )

            for column, value in unknown_profile_values.items():
                filled_user_dataset.loc[day, column] = value

            continue

        if psychometric_columns and profile_values[psychometric_columns].isna().any():
            raise ValueError(
                "Cannot fill synthetic calendar day psychometric columns "
                "because the known profile contains missing psychometric values."
            )

        filled_user_dataset.loc[day, profile_columns] = profile_values[profile_columns]


def _get_profile_values_for_day(
    original_user_dataset: pd.DataFrame,
    day: pd.Timestamp,
    profile_columns: list[str],
) -> pd.Series | None:
    original_user_dataset = original_user_dataset.copy()
    original_user_dataset["day"] = pd.to_datetime(
        original_user_dataset["day"]
    ).dt.floor("D")

    same_month_profiles = original_user_dataset[
        original_user_dataset["day"].dt.to_period("M") == day.to_period("M")
    ][profile_columns].dropna(how="all")

    if not same_month_profiles.empty:
        return same_month_profiles.iloc[0]

    known_profiles = original_user_dataset[profile_columns].dropna(how="all")

    if not known_profiles.empty:
        return known_profiles.iloc[0]

    return None


def _build_unknown_profile_one_hot_values(
    profile_one_hot_columns: list[str]
) -> dict[str, int]:
    unknown_columns = [
        column
        for column in profile_one_hot_columns
        if column.endswith(PROFILE_UNKNOWN_SUFFIX)
    ]
    unknown_values = {column: 0 for column in profile_one_hot_columns}
    covered_columns: set[str] = set()

    for unknown_column in unknown_columns:
        profile_family_prefix = unknown_column.removesuffix("unknown")
        profile_family_columns = [
            column
            for column in profile_one_hot_columns
            if column.startswith(profile_family_prefix)
        ]

        covered_columns.update(profile_family_columns)
        unknown_values[unknown_column] = 1

    uncovered_columns = sorted(set(profile_one_hot_columns) - covered_columns)

    if uncovered_columns:
        raise ValueError(
            "Profile one-hot columns do not have a matching unknown column: "
            f"{', '.join(uncovered_columns)}"
        )

    return unknown_values


def _restore_synthetic_calendar_fill_dtypes(
    filled_user_dataset: pd.DataFrame,
    original_dtypes: pd.Series,
) -> None:
    for column, dtype in original_dtypes.items():
        if column not in filled_user_dataset.columns:
            continue

        if column == TARGET_COLUMN:
            filled_user_dataset[column] = filled_user_dataset[column].astype(dtype)
            continue

        if pd.api.types.is_bool_dtype(dtype):
            filled_user_dataset[column] = filled_user_dataset[column].astype("bool")


def _validate_split_days(split_days: dict[str, list[pd.Timestamp]]) -> None:
    for split_name in SPLIT_NAMES:
        days = split_days[split_name]

        if not days:
            raise ValueError(f"Temporal split '{split_name}' has no days.")

        if len(days) < WINDOW_SIZE:
            raise ValueError(
                f"Temporal split '{split_name}' has only {len(days)} days, "
                f"but WINDOW_SIZE is {WINDOW_SIZE}. "
                "Future windows would need to cross split boundaries."
            )

    train_days = set(split_days["train"])
    validation_days = set(split_days["validation"])
    test_days = set(split_days["test"])

    has_overlapping_days = (
        train_days & validation_days or
        train_days & test_days or
        validation_days & test_days
    )

    if has_overlapping_days:
        raise ValueError("Temporal splits contain overlapping days.")

    if not (
        max(split_days["train"]) < min(split_days["validation"]) and
        max(split_days["validation"]) < min(split_days["test"])
    ):
        raise ValueError("Temporal splits are not ordered as train, validation, test.")


def _count_calendar_missing_rows(dataset: pd.DataFrame) -> int:
    if CALENDAR_MISSING_DAY_COLUMN not in dataset.columns:
        return 0

    return int(dataset[CALENDAR_MISSING_DAY_COLUMN].astype("bool").sum())


def _build_split_summary(
    split_name: str,
    dataset: pd.DataFrame,
    days: list[pd.Timestamp]
) -> SplitSummary:
    split_dataset = dataset[dataset["day"].isin(days)]
    calendar_missing_rows = _count_calendar_missing_rows(split_dataset)
    positive_rows = int(split_dataset["label"].eq(1).sum())
    negative_rows = int(split_dataset["label"].eq(0).sum())
    eligible_window_end_index = len(days) - int(WINDOW_SIZE)
    eligible_window_count = (
        (len(days) - int(WINDOW_SIZE)) // int(WINDOW_STEP)
    ) + 1

    if positive_rows == 0:
        raise ValueError(f"Temporal split '{split_name}' has no positive rows.")

    return SplitSummary(
        start_day=days[0],
        end_day=days[-1],
        days=int(len(days)),
        rows=int(len(split_dataset.index)),
        observed_rows=int(len(split_dataset.index) - calendar_missing_rows),
        calendar_missing_rows=calendar_missing_rows,
        users=int(split_dataset["user"].nunique()),
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        eligible_window_start_day=days[0],
        eligible_window_end_day=days[eligible_window_end_index],
        eligible_window_count=int(eligible_window_count),
    )


def _build_feature_columns_report(
    dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    source_file: str,
    variance_scope: str = FEATURE_VARIANCE_SCOPE,
    scaler_fit_split: str = SCALER_FIT_SPLIT_NAME,
) -> FeatureColumnsReport:
    raw_candidate_feature_columns = _get_candidate_feature_columns(dataset)
    (
        candidate_feature_columns,
        removed_by_profile_mode_columns,
    ) = _filter_profile_features(raw_candidate_feature_columns)

    _validate_numeric_feature_columns(dataset, candidate_feature_columns)
    _validate_finite_feature_values(dataset, candidate_feature_columns)

    zero_variance_columns = _get_zero_variance_feature_columns(
        train_dataset,
        candidate_feature_columns
    )
    removed_zero_variance_columns = list(zero_variance_columns)
    removed_zero_variance_columns_set = set(removed_zero_variance_columns)
    feature_columns_after_zero_variance = [
        column
        for column in candidate_feature_columns
        if column not in removed_zero_variance_columns_set
    ]
    (
        feature_columns,
        exact_duplicate_feature_groups,
    ) = _select_exact_duplicate_feature_columns(
        train_dataset,
        feature_columns_after_zero_variance,
    )
    removed_exact_duplicate_columns = [
        column
        for group in exact_duplicate_feature_groups
        for column in group.removed_columns
    ]
    _validate_removed_exact_duplicates_are_absent(
        feature_columns,
        removed_exact_duplicate_columns,
    )
    _validate_exact_duplicate_feature_groups(
        train_dataset,
        feature_columns,
        exact_duplicate_feature_groups,
    )
    unscaled_feature_columns = [
        column
        for column in UNSCALED_FEATURE_COLUMNS
        if column in feature_columns
    ]
    unscaled_feature_columns_set = set(unscaled_feature_columns)
    scaled_feature_columns = [
        column
        for column in feature_columns
        if column not in unscaled_feature_columns_set
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns remain after removing zero-variance columns "
            "from the training split."
        )

    if not scaled_feature_columns:
        raise ValueError("No scaled feature columns remain for StandardScaler.")

    counts = FeatureColumnCounts(
        candidate_feature_columns=int(len(candidate_feature_columns)),
        removed_by_profile_mode_columns=int(len(removed_by_profile_mode_columns)),
        removed_zero_variance_columns=int(len(removed_zero_variance_columns)),
        removed_exact_duplicate_columns=int(len(removed_exact_duplicate_columns)),
        scaled_feature_columns=int(len(scaled_feature_columns)),
        unscaled_feature_columns=int(len(unscaled_feature_columns)),
        feature_columns=int(len(feature_columns)),
    )

    return FeatureColumnsReport(
        source_file=source_file,
        variance_scope=variance_scope,
        scaler_file=str(SCALERS_DIR / STANDARD_SCALER_FILE),
        scaler_fit_split=scaler_fit_split,
        feature_value_transform=FEATURE_VALUE_TRANSFORM,
        profile_feature_mode=str(PROFILE_FEATURE_MODE),
        exact_duplicate_feature_selection_enabled=bool(
            EXACT_DUPLICATE_FEATURE_SELECTION_ENABLED
        ),
        exact_duplicate_feature_selection_scope=str(
            EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE
        ),
        exact_duplicate_feature_keep_policy=str(
            EXACT_DUPLICATE_FEATURE_KEEP_POLICY
        ),
        metadata_columns=list(METADATA_COLUMNS),
        target_column=TARGET_COLUMN,
        excluded_identifier_columns=list(IDENTIFIER_COLUMNS),
        candidate_feature_columns=candidate_feature_columns,
        removed_by_profile_mode_columns=removed_by_profile_mode_columns,
        removed_zero_variance_columns=removed_zero_variance_columns,
        removed_exact_duplicate_columns=removed_exact_duplicate_columns,
        exact_duplicate_feature_groups=exact_duplicate_feature_groups,
        scaled_feature_columns=scaled_feature_columns,
        unscaled_feature_columns=unscaled_feature_columns,
        feature_columns=feature_columns,
        feature_group_counts=_get_feature_group_counts(feature_columns),
        behavioral_derived_feature_columns=_get_behavioral_derived_feature_columns(
            feature_columns
        ),
        counts=counts,
    )


def _get_candidate_feature_columns(dataset: pd.DataFrame) -> list[str]:
    candidate_feature_columns = [
        column
        for column in dataset.columns
        if column not in set(IDENTIFIER_COLUMNS).union(LABEL_METADATA_COLUMNS)
    ]

    if not candidate_feature_columns:
        raise ValueError("Final daily dataset has no candidate feature columns.")

    return candidate_feature_columns


def _filter_profile_features(
    candidate_feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    kept_columns: list[str] = []
    removed_columns: list[str] = []

    for column in candidate_feature_columns:
        is_profile_column = column.startswith(PROFILE_FEATURE_PREFIX)
        is_psychometric_column = column.startswith(PROFILE_PSYCHOMETRIC_PREFIX)

        should_keep = (
            PROFILE_FEATURE_MODE == "all"
            or not is_profile_column
            or (
                PROFILE_FEATURE_MODE == "psychometric_only"
                and is_psychometric_column
            )
            or (
                PROFILE_FEATURE_MODE == "static_only"
                and is_profile_column
                and not is_psychometric_column
            )
        )

        if should_keep:
            kept_columns.append(column)
        else:
            removed_columns.append(column)

    if not kept_columns:
        raise ValueError(
            "No candidate feature columns remain after applying "
            f"PROFILE_FEATURE_MODE={PROFILE_FEATURE_MODE}."
        )

    return kept_columns, removed_columns


def _select_exact_duplicate_feature_columns(
    train_dataset: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[ExactDuplicateFeatureGroup]]:
    if not EXACT_DUPLICATE_FEATURE_SELECTION_ENABLED:
        return list(feature_columns), []

    signature_groups: dict[str, list[str]] = {}

    for column in feature_columns:
        signature = _get_feature_value_signature(train_dataset, column)
        signature_groups.setdefault(signature, []).append(column)

    duplicate_groups: list[ExactDuplicateFeatureGroup] = []
    removed_columns: set[str] = set()

    for signature_columns in signature_groups.values():
        if len(signature_columns) < 2:
            continue

        exact_groups = _split_exact_duplicate_signature_group(
            train_dataset,
            signature_columns,
        )

        for duplicate_columns in exact_groups:
            if len(duplicate_columns) < 2:
                continue

            kept_column, keep_reason = _choose_base_feature_column(
                duplicate_columns
            )
            removed_group_columns = [
                column
                for column in duplicate_columns
                if column != kept_column
            ]
            duplicate_groups.append(
                ExactDuplicateFeatureGroup(
                    kept_column=kept_column,
                    removed_columns=removed_group_columns,
                    duplicate_columns=[kept_column, *removed_group_columns],
                    selection_scope=str(EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE),
                    keep_reason=keep_reason,
                )
            )
            removed_columns.update(removed_group_columns)

    selected_feature_columns = [
        column
        for column in feature_columns
        if column not in removed_columns
    ]

    return selected_feature_columns, duplicate_groups


def _split_exact_duplicate_signature_group(
    train_dataset: pd.DataFrame,
    columns: list[str],
) -> list[list[str]]:
    pending_columns = list(columns)
    exact_groups: list[list[str]] = []

    while pending_columns:
        base_column = pending_columns.pop(0)
        base_values = _get_feature_float_values(train_dataset, base_column)
        exact_group = [base_column]
        remaining_columns: list[str] = []

        for candidate_column in pending_columns:
            candidate_values = _get_feature_float_values(
                train_dataset,
                candidate_column,
            )

            if np.array_equal(base_values, candidate_values):
                exact_group.append(candidate_column)
            else:
                remaining_columns.append(candidate_column)

        exact_groups.append(exact_group)
        pending_columns = remaining_columns

    return exact_groups


def _get_feature_value_signature(
    dataset: pd.DataFrame,
    column: str,
) -> str:
    values = _get_feature_float_values(dataset, column)

    return hashlib.blake2b(values.tobytes(), digest_size=16).hexdigest()


def _get_feature_float_values(
    dataset: pd.DataFrame,
    column: str,
) -> np.ndarray:
    values = dataset[column].astype("float64").to_numpy(copy=True)
    values[values == 0.0] = 0.0

    return values


def _choose_base_feature_column(columns: list[str]) -> tuple[str, str]:
    if EXACT_DUPLICATE_FEATURE_KEEP_POLICY != "base_priority":
        raise ValueError(
            "Unsupported exact duplicate feature keep policy: "
            f"{EXACT_DUPLICATE_FEATURE_KEEP_POLICY}"
        )

    for canonical_column in EXACT_DUPLICATE_CANONICAL_FEATURE_PRIORITY:
        if canonical_column in columns:
            return canonical_column, "canonical_base_feature_priority"

    return (
        min(columns, key=_get_base_feature_priority_key),
        "base_priority_heuristic",
    )


def _get_base_feature_priority_key(column: str) -> tuple[int, int, int, str]:
    derived_marker_count = sum(
        marker in column
        for marker in DERIVED_FEATURE_NAME_MARKERS
    )
    base_name_priority = _get_base_name_priority(column)

    return derived_marker_count, base_name_priority, len(column), column


def _get_base_name_priority(column: str) -> int:
    for index, marker in enumerate(EXACT_DUPLICATE_BASE_NAME_PRIORITY):
        if marker in column:
            return index

    return len(EXACT_DUPLICATE_BASE_NAME_PRIORITY)


def _validate_removed_exact_duplicates_are_absent(
    feature_columns: list[str],
    removed_exact_duplicate_columns: list[str],
) -> None:
    remaining_removed_columns = sorted(
        set(feature_columns).intersection(removed_exact_duplicate_columns)
    )

    if remaining_removed_columns:
        raise ValueError(
            "Exact duplicate feature columns were marked as removed but still "
            f"exist in feature_columns: {remaining_removed_columns}"
        )


def _validate_exact_duplicate_feature_groups(
    train_dataset: pd.DataFrame,
    feature_columns: list[str],
    duplicate_groups: list[ExactDuplicateFeatureGroup],
) -> None:
    feature_column_set = set(feature_columns)

    for group in duplicate_groups:
        if len(group.duplicate_columns) < 2:
            raise ValueError("Exact duplicate feature groups must have at least 2 columns.")

        if group.kept_column in group.removed_columns:
            raise ValueError(
                "Exact duplicate kept_column cannot appear in removed_columns: "
                f"{group.kept_column}"
            )

        if group.kept_column not in feature_column_set:
            raise ValueError(
                "Exact duplicate kept_column must remain in feature_columns: "
                f"{group.kept_column}"
            )

        missing_removed_columns = [
            column
            for column in group.removed_columns
            if column in feature_column_set
        ]
        if missing_removed_columns:
            raise ValueError(
                "Exact duplicate removed_columns must be absent from "
                f"feature_columns: {missing_removed_columns}"
            )

        kept_values = _get_feature_float_values(train_dataset, group.kept_column)
        for removed_column in group.removed_columns:
            removed_values = _get_feature_float_values(train_dataset, removed_column)

            if not np.array_equal(kept_values, removed_values):
                raise ValueError(
                    "Exact duplicate report contains columns with different "
                    "values in the selection scope. "
                    f"kept={group.kept_column}, removed={removed_column}"
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
    behavioral_columns = set(_get_behavioral_derived_feature_columns(feature_columns))

    for column in feature_columns:
        if column in behavioral_columns:
            group_counts["behavioral_derived"] += 1
        elif column.startswith(PROFILE_PSYCHOMETRIC_PREFIX):
            group_counts["profile_psychometric"] += 1
        elif column.startswith(PROFILE_FEATURE_PREFIX):
            group_counts["profile_static"] += 1
        elif column == CALENDAR_MISSING_DAY_COLUMN:
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


def _get_behavioral_derived_feature_columns(feature_columns: list[str]) -> list[str]:
    behavioral_markers = (
        "_ratio",
        "_rolling_",
        "_vs_user_",
        "_vs_rolling_",
        "_new_",
        "_without_same_day_",
        "_active_span_",
        "_high_risk_",
        "_active_modality_",
        "_behavioral_intensity_",
        "_multi_domain_high_risk_",
    )

    return [
        column
        for column in feature_columns
        if column.startswith("daily_") or any(
            marker in column
            for marker in behavioral_markers
        )
    ]


def _validate_numeric_feature_columns(
    dataset: pd.DataFrame,
    feature_columns: list[str]
) -> None:
    non_numeric_columns = [
        column
        for column in feature_columns
        if not is_numeric_dtype(dataset[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "Feature columns must be numeric after excluding "
            f"{', '.join(IDENTIFIER_COLUMNS)}. "
            f"Non-numeric columns: {', '.join(non_numeric_columns)}"
        )


def _validate_finite_feature_values(
    dataset: pd.DataFrame,
    feature_columns: list[str]
) -> None:
    missing_value_columns: list[str] = []
    entirely_missing_columns: list[str] = []
    infinite_value_columns: list[str] = []

    for column in feature_columns:
        column_values = dataset[column]

        if column_values.isna().all():
            entirely_missing_columns.append(column)
            continue

        if column_values.isna().any():
            missing_value_columns.append(column)
            continue

        numeric_values = column_values.astype("float64").to_numpy()

        if not np.isfinite(numeric_values).all():
            infinite_value_columns.append(column)

    if (
        missing_value_columns or
        entirely_missing_columns or
        infinite_value_columns
    ):
        error_parts = []

        if missing_value_columns:
            error_parts.append(
                "columns with missing values: "
                f"{', '.join(missing_value_columns)}"
            )

        if entirely_missing_columns:
            error_parts.append(
                "columns with no valid values: "
                f"{', '.join(entirely_missing_columns)}"
            )

        if infinite_value_columns:
            error_parts.append(
                "columns with infinite values: "
                f"{', '.join(infinite_value_columns)}"
            )

        raise ValueError("Invalid feature values found: " + "; ".join(error_parts))


def _get_zero_variance_feature_columns(
    train_dataset: pd.DataFrame,
    feature_columns: list[str]
) -> list[str]:
    return [
        column
        for column in feature_columns
        if train_dataset[column].nunique(dropna=False) <= 1
    ]


def _fit_standard_scaler_on_train_features(
    train_dataset: pd.DataFrame,
    feature_columns: list[str]
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_dataset[feature_columns])

    return scaler


def _normalize_split_datasets(
    split_datasets: dict[str, pd.DataFrame],
    feature_columns: list[str],
    scaler: StandardScaler,
) -> dict[str, pd.DataFrame]:
    normalized_split_datasets: dict[str, pd.DataFrame] = {}

    for split_name, source_split_dataset in split_datasets.items():
        split_dataset = source_split_dataset.copy()
        normalized_values = scaler.transform(split_dataset[feature_columns])

        split_dataset[feature_columns] = pd.DataFrame(
            normalized_values,
            columns=feature_columns,
            index=split_dataset.index,
        )
        normalized_split_datasets[split_name] = split_dataset

    return normalized_split_datasets


def _validate_normalized_feature_values(
    split_datasets: dict[str, pd.DataFrame],
    feature_columns: list[str]
) -> None:
    for split_name in split_datasets:
        try:
            _validate_finite_feature_values(split_datasets[split_name], feature_columns)
        except ValueError as error:
            raise ValueError(
                f"Normalized feature values are invalid in split '{split_name}'. "
                f"{error}"
            ) from error


def _build_all_sequence_splits(
    split_datasets: dict[str, pd.DataFrame],
    feature_columns: list[str],
    split_user_group_maps: dict[str, dict[str, str]] | None = None,
    require_consecutive_calendar_days: bool = True,
) -> dict[str, SequenceSplit]:
    return {
        split_name: _build_sequence_split(
            split_name,
            split_datasets[split_name],
            feature_columns,
            user_group_by_user=(
                None
                if split_user_group_maps is None
                else split_user_group_maps.get(split_name)
            ),
            require_consecutive_calendar_days=require_consecutive_calendar_days,
        )
        for split_name in split_datasets
    }


def _build_sequence_split(
    split_name: str,
    split_dataset: pd.DataFrame,
    feature_columns: list[str],
    user_group_by_user: dict[str, str] | None = None,
    require_consecutive_calendar_days: bool = True,
) -> SequenceSplit:
    print(f"Building {split_name} sequences ...")

    X_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray | np.float32] = []
    metadata_rows: list[dict[str, object]] = []
    candidate_windows = 0
    rejected_windows = 0
    rejection_reasons: dict[str, int] = {}

    for user, user_dataset in split_dataset.groupby("user", sort=True):
        user_dataset = (
            user_dataset
            .sort_values("day")
            .reset_index(drop=True)
        )
        days = pd.to_datetime(user_dataset["day"]).dt.floor("D")
        feature_values = (
            user_dataset[feature_columns]
            .astype("float32")
            .to_numpy()
        )
        calendar_missing_day_values = (
            user_dataset[CALENDAR_MISSING_DAY_COLUMN]
            .astype("bool")
            .to_numpy()
        )
        labels = (
            user_dataset[TARGET_COLUMN]
            .astype("float32")
            .to_numpy()
        )

        for start_index in range(
            0,
            len(user_dataset.index) - int(WINDOW_SIZE) + 1,
            int(WINDOW_STEP)
        ):
            end_index = start_index + int(WINDOW_SIZE)
            window_days = days.iloc[start_index:end_index]
            candidate_windows += 1

            if (
                require_consecutive_calendar_days
                and not _is_calendar_consecutive_window(window_days)
            ):
                rejected_windows += 1
                _increment_rejection_reason(
                    rejection_reasons,
                    "non_consecutive_calendar_days"
                )
                continue

            window_missing_day_values = calendar_missing_day_values[
                start_index:end_index
            ]
            (
                missing_day_count,
                missing_day_ratio,
                max_consecutive_missing_days,
            ) = _calculate_window_missing_day_stats(window_missing_day_values)

            window_rejection_reasons = _get_window_rejection_reasons(
                missing_day_ratio,
                max_consecutive_missing_days,
            )

            if window_rejection_reasons:
                rejected_windows += 1

                for reason in window_rejection_reasons:
                    _increment_rejection_reason(rejection_reasons, reason)

                continue

            X_windows.append(feature_values[start_index:end_index])

            if PREDICT_ALL_DAYS:
                y_windows.append(labels[start_index:end_index].reshape(-1, 1))
            else:
                y_windows.append(np.float32(labels[end_index - 1]))

            metadata_rows.append(
                {
                    "window_index": len(metadata_rows),
                    "split_name": split_name,
                    "user": str(user),
                    "user_group": (
                        user_group_by_user.get(str(user), split_name)
                        if user_group_by_user is not None
                        else split_name
                    ),
                    "start_day": window_days.iloc[0],
                    "end_day": window_days.iloc[-1],
                    "has_positive_label": bool(labels[start_index:end_index].max() == 1),
                    "calendar_missing_day_count": missing_day_count,
                    "calendar_missing_day_ratio": missing_day_ratio,
                    "max_consecutive_missing_days": max_consecutive_missing_days,
                    "window_days": [
                        day.date().isoformat()
                        for day in window_days
                    ],
                }
            )

    if not X_windows:
        raise ValueError(f"Temporal split '{split_name}' generated no valid windows.")

    X = torch.from_numpy(np.stack(X_windows).astype("float32"))
    y = torch.from_numpy(np.stack(y_windows).astype("float32"))
    metadata = pd.DataFrame(metadata_rows)

    return SequenceSplit(
        X=X,
        y=y,
        metadata=metadata,
        candidate_windows=int(candidate_windows),
        rejected_windows=int(rejected_windows),
        rejection_reasons=rejection_reasons,
    )


def _is_calendar_consecutive_window(window_days: pd.Series) -> bool:
    day_diffs = window_days.diff().dropna()

    return bool(day_diffs.eq(pd.Timedelta(days=1)).all())


def _calculate_window_missing_day_stats(
    calendar_missing_day_values: np.ndarray,
) -> tuple[int, float, int]:
    missing_day_count = int(calendar_missing_day_values.sum())
    missing_day_ratio = missing_day_count / len(calendar_missing_day_values)
    max_consecutive_missing_days = _calculate_max_consecutive_true_values(
        calendar_missing_day_values
    )

    return (
        missing_day_count,
        float(missing_day_ratio),
        int(max_consecutive_missing_days),
    )


def _calculate_max_consecutive_true_values(values: np.ndarray) -> int:
    max_consecutive_values = 0
    current_consecutive_values = 0

    for value in values:
        if bool(value):
            current_consecutive_values += 1
            max_consecutive_values = max(
                max_consecutive_values,
                current_consecutive_values
            )
            continue

        current_consecutive_values = 0

    return max_consecutive_values


def _get_window_rejection_reasons(
    missing_day_ratio: float,
    max_consecutive_missing_days: int,
) -> list[str]:
    rejection_reasons: list[str] = []

    if missing_day_ratio > MAX_MISSING_DAY_RATIO:
        rejection_reasons.append("max_missing_day_ratio_exceeded")

    if max_consecutive_missing_days > MAX_CONSECUTIVE_MISSING_DAYS:
        rejection_reasons.append("max_consecutive_missing_days_exceeded")

    return rejection_reasons


def _increment_rejection_reason(
    rejection_reasons: dict[str, int],
    reason: str,
) -> None:
    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1


def _build_sequences_metadata(
    sequence_splits: dict[str, SequenceSplit],
    required_positive_splits: set[str] | None = None,
    synthetic_calendar_days_enabled: bool = True,
    window_unit: str = "calendar_days",
    require_consecutive_calendar_days: bool = True,
) -> SequencesMetadata:
    required_positive_splits = (
        set(SPLIT_NAMES)
        if required_positive_splits is None
        else set(required_positive_splits)
    )
    split_summaries = _build_sequence_split_summaries(sequence_splits)
    for split_name, summary in split_summaries.items():
        if split_name in required_positive_splits and summary.y_positive_count == 0:
            raise ValueError(
                f"Sequence split '{split_name}' has no positive sequence labels."
            )

    return SequencesMetadata(
        window_size=int(WINDOW_SIZE),
        window_step=int(WINDOW_STEP),
        predict_all_days=bool(PREDICT_ALL_DAYS),
        synthetic_calendar_days_enabled=bool(synthetic_calendar_days_enabled),
        window_unit=str(window_unit),
        require_consecutive_calendar_days=bool(require_consecutive_calendar_days),
        max_missing_day_ratio=float(MAX_MISSING_DAY_RATIO),
        max_consecutive_missing_days=int(MAX_CONSECUTIVE_MISSING_DAYS),
        total_candidate_windows=int(
            sum(summary.candidate_windows for summary in split_summaries.values())
        ),
        total_created_windows=int(
            sum(summary.created_windows for summary in split_summaries.values())
        ),
        total_rejected_windows=int(
            sum(summary.rejected_windows for summary in split_summaries.values())
        ),
        splits=split_summaries,
    )


def _build_sequence_split_summaries(
    sequence_splits: dict[str, SequenceSplit]
) -> dict[str, SequenceSplitSummary]:
    return {
        split_name: SequenceSplitSummary(
            candidate_windows=int(sequence_split.candidate_windows),
            created_windows=int(len(sequence_split.metadata.index)),
            rejected_windows=int(sequence_split.rejected_windows),
            y_positive_count=int(sequence_split.y.eq(1).sum()),
            y_negative_count=int(sequence_split.y.eq(0).sum()),
            X_shape=list(sequence_split.X.shape),
            y_shape=list(sequence_split.y.shape),
            n_features=int(sequence_split.X.shape[-1]),
            rejection_reasons=sequence_split.rejection_reasons,
        )
        for split_name, sequence_split in sequence_splits.items()
    }


def _save_split_daily_datasets(
    split_datasets: dict[str, pd.DataFrame]
) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in split_datasets.items():
        output_path = SPLITS_DIR / f"{split_name}{SPLIT_DAILY_DATASET_FILE_SUFFIX}"

        print(f"Saving {split_name} daily split dataset to {output_path} ...")
        split_dataset.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )


def _save_split_metadata(
    split_metadata: SplitMetadata | ArticleSplitMetadata
) -> None:
    output_path = SPLITS_DIR / SPLIT_METADATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving split metadata to {output_path} ...")
    save_json(split_metadata, output_path)


def _save_feature_columns_report(feature_columns_report: FeatureColumnsReport) -> None:
    output_path = SPLITS_DIR / FEATURE_COLUMNS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving feature columns report to {output_path} ...")
    save_json(feature_columns_report, output_path)


def _save_sequence_splits(sequence_splits: dict[str, SequenceSplit]) -> None:
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, sequence_split in sequence_splits.items():
        X_path = SEQUENCES_DIR / f"{split_name}{SEQUENCE_X_FILE_SUFFIX}"
        y_path = SEQUENCES_DIR / f"{split_name}{SEQUENCE_Y_FILE_SUFFIX}"
        metadata_path = SEQUENCES_DIR / f"{split_name}{SEQUENCE_METADATA_FILE_SUFFIX}"

        print(f"Saving {split_name} X sequences to {X_path} ...")
        torch.save(sequence_split.X, X_path)

        print(f"Saving {split_name} y sequences to {y_path} ...")
        torch.save(sequence_split.y, y_path)

        print(f"Saving {split_name} sequence metadata to {metadata_path} ...")
        sequence_split.metadata.to_parquet(
            metadata_path,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )


def _save_sequences_metadata(sequences_metadata: SequencesMetadata) -> None:
    output_path = SEQUENCES_DIR / SEQUENCES_METADATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving sequences metadata to {output_path} ...")
    save_json(sequences_metadata, output_path)


def _save_standard_scaler(standard_scaler: StandardScaler) -> None:
    output_path = SCALERS_DIR / STANDARD_SCALER_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving standard scaler to {output_path} ...")
    joblib.dump(standard_scaler, output_path)


if __name__ == "__main__":
    main()
