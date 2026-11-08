import pandas as pd

from src.itd_tcnTransformer.config.paths import (
    DAILY_FEATURES_DIR,
    DAILY_LABELS_DIR,
    FINAL_DAILY_DATASET_DIR,
)


DAILY_FEATURES_FILE = "daily_features.parquet"
DAILY_LABELS_FILE = "daily_labels.parquet"
FINAL_DAILY_DATASET_FILE = "final_daily_dataset.parquet"
USER_DAY_COLUMNS = ["user", "day"]
FORBIDDEN_LABEL_METADATA_COLUMNS = {
    "scenario_id",
    "malicious_user",
    "attack_start",
    "attack_end",
    "attack_start_day",
    "attack_end_day",
    "label_source",
}

__all__ = ["main"]


def main() -> None:
    print("Building final daily dataset ...")

    daily_features = _load_daily_features()
    daily_labels = _load_daily_labels()

    _validate_unique_user_day(daily_features, "daily features")
    _validate_unique_user_day(daily_labels, "daily labels")
    _validate_label_values(daily_labels)

    final_daily_dataset = _build_final_daily_dataset(daily_features, daily_labels)
    _validate_no_missing_labels(final_daily_dataset)
    _validate_no_label_metadata_columns(final_daily_dataset)
    _validate_unique_user_day(final_daily_dataset, "final daily dataset")

    final_daily_dataset["label"] = final_daily_dataset["label"].astype("int64")
    final_daily_dataset = final_daily_dataset.sort_values(USER_DAY_COLUMNS)

    _save_final_daily_dataset(final_daily_dataset)


def _load_daily_features() -> pd.DataFrame:
    daily_features_path = DAILY_FEATURES_DIR / DAILY_FEATURES_FILE

    if not daily_features_path.exists() or not daily_features_path.is_file():
        raise FileNotFoundError(f"Daily features file not found: {daily_features_path}")

    print(f"Loading daily features from {daily_features_path} ...")
    return pd.read_parquet(daily_features_path)


def _load_daily_labels() -> pd.DataFrame:
    daily_labels_path = DAILY_LABELS_DIR / DAILY_LABELS_FILE

    if not daily_labels_path.exists() or not daily_labels_path.is_file():
        raise FileNotFoundError(f"Daily labels file not found: {daily_labels_path}")

    print(f"Loading daily labels from {daily_labels_path} ...")

    return pd.read_parquet(
        daily_labels_path,
        columns=[*USER_DAY_COLUMNS, "label"]
    )


def _validate_unique_user_day(df: pd.DataFrame, source_name: str) -> None:
    duplicated_rows = df.duplicated(subset=USER_DAY_COLUMNS, keep=False)

    if not duplicated_rows.any():
        return

    duplicated_count = int(duplicated_rows.sum())
    sample = (
        df.loc[duplicated_rows, USER_DAY_COLUMNS]
        .head(10)
        .to_dict(orient="records")
    )

    raise ValueError(
        f"{source_name} contains duplicated user/day pairs. "
        f"Duplicated rows: {duplicated_count}. "
        f"Sample: {sample}"
    )


def _validate_label_values(labels: pd.DataFrame) -> None:
    if labels["label"].isna().any():
        raise ValueError("Daily labels contain missing label values.")

    if not pd.api.types.is_integer_dtype(labels["label"]):
        raise ValueError("Daily labels must be stored as an integer column.")

    invalid_labels = labels[~labels["label"].isin([0, 1])]

    if invalid_labels.empty:
        return

    sample = (
        invalid_labels[[*USER_DAY_COLUMNS, "label"]]
        .head(10)
        .to_dict(orient="records")
    )

    raise ValueError(
        "Daily labels contain values outside the expected binary range {0, 1}. "
        f"Invalid rows: {len(invalid_labels.index)}. "
        f"Sample: {sample}"
    )


def _build_final_daily_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    print("Joining daily features with daily labels ...")

    return features.merge(
        labels[[*USER_DAY_COLUMNS, "label"]],
        on=USER_DAY_COLUMNS,
        how="left"
    )


def _validate_no_missing_labels(final_dataset: pd.DataFrame) -> None:
    missing_labels = final_dataset[final_dataset["label"].isna()]

    if missing_labels.empty:
        return

    sample = (
        missing_labels[USER_DAY_COLUMNS]
        .head(10)
        .to_dict(orient="records")
    )

    raise ValueError(
        "Some daily feature rows do not have a matching daily label. "
        f"Missing labels: {len(missing_labels.index)}. "
        f"Sample: {sample}"
    )


def _validate_no_label_metadata_columns(final_dataset: pd.DataFrame) -> None:
    forbidden_columns = sorted(
        FORBIDDEN_LABEL_METADATA_COLUMNS.intersection(final_dataset.columns)
    )

    if not forbidden_columns:
        return

    raise ValueError(
        "Final daily dataset contains label metadata columns that must not be used "
        f"as features: {forbidden_columns}"
    )


def _save_final_daily_dataset(final_dataset: pd.DataFrame) -> None:
    output_path = FINAL_DAILY_DATASET_DIR / FINAL_DAILY_DATASET_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving final daily dataset to {output_path} ...")
    final_dataset.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )


if __name__ == "__main__":
    main()
