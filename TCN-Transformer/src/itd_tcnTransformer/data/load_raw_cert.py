from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pandas.io.parsers.readers import TextFileReader
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import pandas as pd

from src.itd_tcnTransformer.config.paths import RAW_CERT_DIR, INTERIM_DIR
from src.itd_tcnTransformer.data.validate_raw_cert import find_answers_directory
from src.itd_tcnTransformer.utils.io import get_csv_files_from_directory


@dataclass
class CsvConfig(TypedDict):
    usecols: list[str]
    dtype: dict[str, str]


DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

CSV_CONFIGS: dict[str, CsvConfig] = {
    "device.csv": {
        "usecols": ["id", "date", "user", "pc", "activity"],
        "dtype": {
            "id": "string",
            "user": "string",
            "pc": "string",
            "activity": "string"
        }
    },
    "file.csv": {
        "usecols": ["id", "date", "user", "pc", "filename", "content"],
        "dtype": {
            "id": "string",
            "user": "string",
            "pc": "string",
            "filename": "string",
            "content": "string"
        }
    },
    "http.csv": {
        "usecols": ["id", "date", "user", "pc", "url", "content"],
        "dtype": {
            "id": "string",
            "user": "string",
            "pc": "string",
            "url": "string",
            "content": "string"
        }
    },
    "email.csv": {
        "usecols": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
        "dtype": {
            "id": "string",
            "user": "string",
            "pc": "string",
            "to": "string",
            "cc": "string",
            "bcc": "string",
            "from": "string",
            "size": "int64",
            "attachments": "int64",
            "content": "string"
        }
    },
    "logon.csv": {
        "usecols": ["id", "date", "user", "pc", "activity"],
        "dtype": {
            "id": "string",
            "user": "string",
            "pc": "string",
            "activity": "string"
        }
    },
    "psychometric.csv": {
        "usecols": ["user_id", "O", "C", "E", "A", "N"],
        "dtype": {
            "user_id": "string",
            "O": "int64",
            "C": "int64",
            "E": "int64",
            "A": "int64",
            "N": "int64"
        }
    }
}

LDAP_CSV_CONFIG: CsvConfig = {
    "usecols": ["user_id", "role", "business_unit", "functional_unit", "department", "team", "supervisor"],
    "dtype": {
        "user_id": "string",
        "role": "string",
        "business_unit": "int64",
        "functional_unit": "string",
        "department": "string",
        "team": "string",
        "supervisor": "string"
    }
}

# A partir da coluna "value_1" são colunas que podem ou não existir, dependendo do tipo de evento/log
ANSWER_CSV_CONFIG: CsvConfig = {
    "usecols": ["event_type", "event_id", "date", "user", "pc", "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "value_7"],
    "dtype": {
        "event_type": "string",
        "event_id": "string",
        "date": "string",
        "user": "string",
        "pc": "string",
        "value_1": "string",
        "value_2": "string",
        "value_3": "string",
        "value_4": "string",
        "value_5": "string",
        "value_6": "string",
        "value_7": "string",
    }
}

INSIDERS_CSV_CONFIG: CsvConfig = {
    "usecols": ["dataset", "scenario", "details", "user", "start", "end"],
    "dtype": {
        "dataset": "string",
        "scenario": "int64",
        "details": "string",
        "user": "string",
        "start": "string",
        "end": "string"
    }
}


def main(keep_old_files: bool = False, row_chunk_size: int = 500_000) -> None:
    if INTERIM_DIR.exists():
        if not INTERIM_DIR.is_dir():
            raise NotADirectoryError(f"Expected a directory but found a file: {INTERIM_DIR}")
    else:
        INTERIM_DIR.mkdir(parents=True)

    _load_non_http_sources_in_parallel(keep_old_files, row_chunk_size)
    _load_http_file(keep_old_files, row_chunk_size)

    print("Data loaded successfully!")


def _load_non_http_sources_in_parallel(keep_old_files: bool, row_chunk_size: int) -> None:
    loaders = [
        _load_device_file,
        _load_file_file,
        _load_email_file,
        _load_logon_file,
        _load_psychometric_file,
        _load_ldap_files,
        _load_answer_files,
    ]

    with ThreadPoolExecutor(max_workers=len(loaders)) as executor:
        futures = [
            executor.submit(loader, keep_old_files, row_chunk_size)
            for loader in loaders
        ]

        for future in as_completed(futures):
            future.result()


def _load_device_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "device.csv",
        CSV_CONFIGS["device.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_device_chunk
    )


def _load_file_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "file.csv",
        CSV_CONFIGS["file.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_file_chunk
    )


def _load_http_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "http.csv",
        CSV_CONFIGS["http.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_http_chunk
    )


def _load_email_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "email.csv",
        CSV_CONFIGS["email.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_email_chunk
    )


def _load_logon_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "logon.csv",
        CSV_CONFIGS["logon.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_logon_chunk
    )


def _load_psychometric_file(keep_old_files: bool, row_chunk_size: int) -> None:
    _load_chunked_csv_file(
        "psychometric.csv",
        CSV_CONFIGS["psychometric.csv"],
        keep_old_files,
        row_chunk_size,
        _transform_psychometric_chunk
    )


def _load_chunked_csv_file(
    file_name: str,
    config: CsvConfig,
    keep_old_files: bool,
    row_chunk_size: int,
    transform_chunk: Callable[[pd.DataFrame], None]
) -> None:
    file_path = RAW_CERT_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Expected file not found: {file_path}")

    file_name_without_ext = file_path.stem
    interim_specific_dir = _create_interim_specific_directory(
        file_name_without_ext,
        keep_old_files
    )

    print(f"Loading {file_name} ...")

    reader = _read_csv_chunks(file_path, config, row_chunk_size)

    for index, chunk in enumerate(reader):
        chunk: pd.DataFrame

        parquet_file_path = _get_intermediate_parque_file_path(
            interim_specific_dir,
            file_name_without_ext,
            index,
            keep_old_files
        )

        if parquet_file_path is None:
            continue

        chunk = _normalize_csv_column_names(chunk)
        transform_chunk(chunk)

        _save_intermediate_parquet(chunk, parquet_file_path)


def _load_ldap_files(keep_old_files: bool, row_chunk_size: int) -> None:
    print(f"Loading ldap files ...")

    raw_ldap_dir = RAW_CERT_DIR / "LDAP"
    ldap_interim_dir = _create_interim_specific_directory("LDAP", keep_old_files)

    for file_path in raw_ldap_dir.glob("*.csv"):
        if not file_path.is_file():
            continue

        reader = _read_csv_chunks(file_path, LDAP_CSV_CONFIG, row_chunk_size)

        for index, chunk in enumerate(reader):
            parquet_file_path = _get_intermediate_parque_file_path(
                ldap_interim_dir,
                file_path.stem,
                index,
                keep_old_files
            )

            if parquet_file_path is None:
                continue

            chunk = _normalize_csv_column_names(chunk)

            _save_intermediate_parquet(chunk, parquet_file_path)


def _load_answer_files(keep_old_files: bool, row_chunk_size: int) -> None:
    print(f"Loading answer files ...")

    answers_dir = find_answers_directory()
    answer_dirs = [
        path
        for path in answers_dir.iterdir()
        if path.is_dir() and path.name.startswith("r4.2")
    ]

    answers_interim_dir = _create_interim_specific_directory("answers", keep_old_files)

    for answer_dir in sorted(answer_dirs):
        answer_files = get_csv_files_from_directory(answer_dir)

        for answer_file in answer_files:
            reader = pd.read_csv(
                answer_file,
                chunksize=row_chunk_size,
                header=None,
                names=ANSWER_CSV_CONFIG["usecols"],
                dtype=cast(Any, ANSWER_CSV_CONFIG["dtype"])
            )

            for index, chunk in enumerate(reader):
                chunk: pd.DataFrame

                parquet_file_path = _get_intermediate_parque_file_path(
                    answers_interim_dir,
                    answer_file.stem,
                    index,
                    keep_old_files
                )

                if parquet_file_path is None:
                    continue

                chunk = _normalize_csv_column_names(chunk)

                # Isto, provavelmente, não será útil, mas pra já fica aqui
                _create_date_derivate_columns(chunk)

                _save_intermediate_parquet(chunk, parquet_file_path)

    _load_insiders_file(answers_interim_dir, keep_old_files, row_chunk_size)


def _load_insiders_file(
    answers_interim_dir: Path,
    keep_old_files: bool,
    row_chunk_size: int
) -> None:
    print(f"Loading insiders.csv ...")

    insiders_file = find_answers_directory() / "insiders.csv"
    if not insiders_file.exists() or not insiders_file.is_file():
        raise FileNotFoundError(f"Expected file not found: {insiders_file}")

    reader = _read_csv_chunks(insiders_file, INSIDERS_CSV_CONFIG, row_chunk_size)

    for index, chunk in enumerate(reader):
        chunk: pd.DataFrame

        parquet_file_path = _get_intermediate_parque_file_path(
            answers_interim_dir,
            "insiders",
            index,
            keep_old_files
        )

        if parquet_file_path is None:
            continue

        chunk = _normalize_csv_column_names(chunk)
        _transform_insiders_chunk(chunk)

        if chunk.empty:
            continue

        _save_intermediate_parquet(chunk, parquet_file_path)


def _transform_device_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)
    _normalize_pc_column(df)
    _normalize_activity_column(df)
    _create_date_derivate_columns(df)


def _transform_file_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)
    _normalize_pc_column(df)
    _add_filename_extension_column(df)
    _create_date_derivate_columns(df)


def _transform_http_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)
    _normalize_pc_column(df)
    _normalize_url_column(df)
    _add_domain_column(df)
    _create_date_derivate_columns(df)


def _transform_email_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)
    _normalize_pc_column(df)
    _create_date_derivate_columns(df)


def _transform_logon_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)
    _normalize_pc_column(df)
    _normalize_activity_column(df)
    _create_date_derivate_columns(df)


def _transform_insiders_chunk(df: pd.DataFrame) -> None:
    cert_r42_rows = df["dataset"].astype("string").str.strip().eq("4.2")
    df.drop(index=df.index[~cert_r42_rows], inplace=True)

    if df.empty:
        return

    _normalize_user_column(df)

    required_columns = ["scenario", "user", "start", "end"]
    for column in required_columns:
        values = df[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"insiders.csv contains missing values in column {column}.")

    df["start"] = pd.to_datetime(df["start"], format=DATE_FORMAT)
    df["end"] = pd.to_datetime(df["end"], format=DATE_FORMAT)
    df["start_day"] = df["start"].dt.floor("D")
    df["end_day"] = df["end"].dt.floor("D")

    invalid_intervals = df[df["start"].gt(df["end"])]
    if not invalid_intervals.empty:
        sample = (
            invalid_intervals[["user", "scenario", "start", "end"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"insiders.csv contains invalid intervals. Sample: {sample}")

    if "start_day" not in df.columns or "end_day" not in df.columns:
        raise ValueError("Failed to create start_day/end_day columns for insiders.csv.")


def _transform_psychometric_chunk(df: pd.DataFrame) -> None:
    _normalize_user_column(df)


def _normalize_user_column(df: pd.DataFrame) -> None:
    df["user"] = df["user"].astype("string").str.strip()


def _normalize_pc_column(df: pd.DataFrame) -> None:
    df["pc"] = df["pc"].astype("string").str.strip()


def _normalize_activity_column(df: pd.DataFrame) -> None:
    df["activity"] = df["activity"].astype("string").str.strip().str.lower()


def _normalize_url_column(df: pd.DataFrame) -> None:
    df["url"] = df["url"].astype("string")


def _add_domain_column(df: pd.DataFrame) -> None:
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)", expand=False)


def _create_date_derivate_columns(df: pd.DataFrame) -> None:
    # Converter coluna de data para datetime
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)

    # Criar colunas adicionais para facilitar o processamento posterior
    df["day"] = df["date"].dt.floor("D")
    df["hour"] = df["date"].dt.hour
    # Assume after-hours antes de 8h e depois de 18h
    df["is_after_hours"] = (df["hour"] < 8) | (df["hour"] >= 18)


def _add_filename_extension_column(df: pd.DataFrame) -> None:
    filenames = df["filename"].astype("string").str.strip()
    df["extension"] = (
        filenames
        .str.extract(r"\.([^./\\]+)$", expand=False)
        .str.lower()
        .astype("string")
    )


def _create_interim_specific_directory(dir_name: str | Path, keep_old_files: bool) -> Path:
    interim_specific_dir = INTERIM_DIR / dir_name

    if interim_specific_dir.exists():
        if not interim_specific_dir.is_dir():
            raise NotADirectoryError(
                f"Expected a directory but found a file: {interim_specific_dir}")

        if not keep_old_files:
            # Apagar ficheiros antigos para garantir que não se misturam dados antigos com os novos
            for old_file in interim_specific_dir.glob("*.parquet"):
                old_file.unlink()
    else:
        interim_specific_dir.mkdir(parents=True)

    return interim_specific_dir


def _get_intermediate_parque_file_path(
    interim_specific_dir: Path,
    file_name_without_ext: str,
    index: int,
    keep_old_files: bool
) -> Path | None:
    """ Constrói o nome a usar para o ficheiro parquet intermédio a criar. Se keep_old_files = True e o ficheiro já existir, devolve None (indicando que não é suposto criar o ficheiro) """
    parquet_file_path = interim_specific_dir / f"{file_name_without_ext}_{index:05d}.parquet"

    if keep_old_files and parquet_file_path.exists() and parquet_file_path.is_file():
        return None

    return parquet_file_path


def _normalize_csv_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()

    if "user_id" in df.columns and "user" not in df.columns:
        df["user"] = df["user_id"].astype("string")

        return df.drop(columns=["user_id"])

    return df


def _read_csv_chunks(
    file_path: Path,
    config: CsvConfig,
    row_chunk_size: int
) -> TextFileReader:
    return pd.read_csv(
        file_path,
        chunksize=row_chunk_size,
        usecols=config["usecols"],
        dtype=cast(Any, config["dtype"])
    )


def _save_intermediate_parquet(chunk: pd.DataFrame, parquet_file_path: Path) -> None:
    chunk.to_parquet(parquet_file_path, index=False, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    main()
