import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime

from src.itd_tcnTransformer.config.paths import INTERIM_DIR, REPORTS_DIR
from src.itd_tcnTransformer.utils.io import save_json


@dataclass
class UserActivity():
    first_day: datetime
    last_day: datetime
    expected_days_count: int = 0
    active_days_count: int = 0
    inactive_days_count: int = 0
    inactive_days_ratio: float = float(0)


@dataclass
class DirReport():
    directory: str
    total_events: int = 0
    total_users: int = 0
    total_unique_days: int | None = None
    min_date: datetime | None = None
    max_date: datetime | None = None
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_ids: set[str] = field(default_factory=set[str])
    events_per_user: dict[str, int] = field(default_factory=dict)
    events_per_day: dict[str, int] = field(default_factory=dict)


@dataclass
class ProfileDirReport():
    directory: str
    total_records: int = 0
    total_users: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    records_per_user: dict[str, int] = field(default_factory=dict)
    duplicate_users: set[str] = field(default_factory=set)


@dataclass
class ExplorationReport():
    created_at: datetime | None = None
    total_events: int = 0
    event_sources: dict[str, DirReport] = field(default_factory=dict)
    profile_sources: dict[str, ProfileDirReport] = field(default_factory=dict)
    user_activity: dict[str, UserActivity] = field(default_factory=dict)


# frozen=True é parecido a um readonly de C#, mas "mais forte", porque impede a modificação dos atributos do objeto também, não apenas a atribuição de uma nova instância à variável.
@dataclass(frozen=True)
class ProfileSourceConfig:
    directory: str
    one_record_per_user: bool


LOG_SOURCES = {
    "device",
    "file",
    "http",
    "email",
    "logon"
}

PROFILE_SOURCES = {
    "ldap": ProfileSourceConfig("LDAP", False),
    "psychometric":  ProfileSourceConfig("psychometric", True)
}


def main() -> None:
    final_report = ExplorationReport()

    active_days_per_user: dict[str, set[pd.Timestamp]] = dict()

    # region Exploração de logs
    for log_source in LOG_SOURCES:
        log_dir_path = INTERIM_DIR / log_source

        if not log_dir_path.exists() or not log_dir_path.is_dir():
            raise NotADirectoryError(f"Path '{log_dir_path}' is not a valid log directory")

        print(f"Analysing '{log_source}' logs ...")

        log_dir = str(log_dir_path.resolve())
        log_report = DirReport(log_dir)

        # usar `set()` em caso de incompatibilidades com numpy.datetime64
        unique_days = set[pd.Timestamp]()
        unique_users = set[str]()
        log_ids: set[str] = set()

        parquet_files = log_dir_path.glob("*.parquet")
        for parquet_file in parquet_files:
            if not parquet_file.is_file():
                continue

            df = pd.read_parquet(parquet_file)

            # region Verificar IDs duplicados
            ids = df["id"].astype("string").dropna()

            unique_ids_file = set(ids.unique())
            duplicate_ids_file = set(ids[ids.duplicated()].unique())
            duplicated_ids_previous = unique_ids_file.intersection(
                log_ids)  # equivalente a `unique_ids_file & log_ids`

            log_report.duplicate_ids.update(duplicate_ids_file)
            log_report.duplicate_ids.update(duplicated_ids_previous)

            log_ids.update(unique_ids_file)
            # endregion

            # region Calcular dias e utilizadores únicos
            days_file = df["day"].dropna().unique()
            users_file = df["user"].dropna().unique()

            unique_days.update(days_file)
            unique_users.update(users_file)
            # endregion

            # Somar eventos do ficheiro ao total do log
            log_report.total_events += len(df.index)

            # region Calcular datas mínima e máxima
            min_date_file = df["date"].min()
            max_date_file = df["date"].max()

            if log_report.min_date is None or min_date_file < log_report.min_date:
                log_report.min_date = min_date_file

            if log_report.max_date is None or max_date_file > log_report.max_date:
                log_report.max_date = max_date_file
            # endregion

            # Calcular valores em falta por coluna
            add_missing_values_to_report(df, log_report.missing_values)

            # Calcular eventos por utilizador
            update_row_count_groupby_col(df, "user", log_report.events_per_user)

            # Calcular eventos por dia
            update_row_count_groupby_col(df, "day", log_report.events_per_day)

            # Obter atividade utilizador
            data = df.groupby("user")

            user_day_pairs = df[["user", "day"]].dropna().drop_duplicates()

            for user, days in user_day_pairs.groupby("user")["day"]:
                user_key = str(user)

                normalized_days = pd.to_datetime(days).dt.normalize()

                if user_key not in active_days_per_user:
                    active_days_per_user[user_key] = set()

                active_days_per_user[user_key].update(normalized_days)

        log_report.total_users = len(unique_users)
        log_report.total_unique_days = len(unique_days)

        final_report.event_sources[log_source] = log_report
    # endregion

    # region Exploração de atividade de utilizadores
    for user, days in active_days_per_user.items():
        user_activity = UserActivity(min(days), max(days))

        user_activity.expected_days_count = (
            user_activity.last_day - user_activity.first_day).days + 1
        user_activity.active_days_count = len(days)
        user_activity.inactive_days_count = user_activity.expected_days_count - user_activity.active_days_count
        user_activity.inactive_days_ratio = user_activity.inactive_days_count / user_activity.expected_days_count

        if user_activity.inactive_days_ratio < 0:
            raise ValueError(
                f"Invalid inactive_days_ratio for user {user}: {user_activity.inactive_days_ratio}")

        final_report.user_activity[user] = user_activity
    # endregion

    # region Exploração de perfis
    for profile_source, config in PROFILE_SOURCES.items():
        profile_dir_path = INTERIM_DIR / config.directory

        if not profile_dir_path.exists() or not profile_dir_path.is_dir():
            raise NotADirectoryError(f"Path '{profile_dir_path}' is not a valid profile directory")

        print(f"Analysing '{profile_source}' profile data ...")

        profile_report = ProfileDirReport(str(profile_dir_path.resolve()))
        seen_users: set[str] = set()

        parquet_files = profile_dir_path.glob("*.parquet")
        for parquet_file in parquet_files:
            if not parquet_file.is_file():
                continue

            df = pd.read_parquet(parquet_file)

            profile_report.total_records += len(df.index)

            add_missing_values_to_report(df, profile_report.missing_values)

            users = df["user"].dropna()
            unique_users_file = {str(user) for user in users.unique()}

            if config.one_record_per_user:
                duplicate_users_file = {str(user) for user in users[users.duplicated()].unique()}

                duplicate_users_previous = unique_users_file.intersection(seen_users)
                
                profile_report.duplicate_users.update(duplicate_users_file)
                profile_report.duplicate_users.update(duplicate_users_previous)
            else:
                update_row_count_groupby_col(df, "user", profile_report.records_per_user)
            
            seen_users.update(unique_users_file)

        profile_report.total_users = len(seen_users)
        
        final_report.profile_sources[profile_source] = profile_report
    # endregion

    # Acrescentar dados do relatório final
    final_report.created_at = datetime.now()
    final_report.total_events = sum(
        source.total_events for source in final_report.event_sources.values())

    report_path = REPORTS_DIR / "loaded_cert_exploration_report.json"

    save_json(final_report, report_path)

    print("Exploration report saved successfully!")


def add_missing_values_to_report(df: pd.DataFrame, missing_values: dict[str, int]) -> None:
    missing_values_file = df.isna().sum()

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        empty_strings = df[column].astype("string").str.strip().eq("").sum()
        missing_values_file[column] += empty_strings

    for column, count in missing_values_file.items():
        col_name = str(column)

        missing_values[col_name] = (missing_values.get(col_name, 0) + int(count))


def update_row_count_groupby_col(df: pd.DataFrame, column: str, data: dict[str, int]):
    """Atualiza as contagens de linhas agrupadas por uma coluna."""
    df_count = df.groupby(column).size()

    for key, count in df_count.items():
        key_name = str(key)

        current_row_count = data.get(key_name, 0)
        data[key_name] = current_row_count + int(count)


if __name__ == "__main__":
    main()
