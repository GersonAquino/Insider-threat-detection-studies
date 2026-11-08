import csv
import pandas as pd

from collections import Counter
from pathlib import Path

from src.itd_tcnTransformer.config.paths import RAW_CERT_DIR
from src.itd_tcnTransformer.utils.io import get_csv_files_from_directory


# Número máximo de linhas de dados a ler quando queremos apenas validar uma amostra.
# Isto evita carregar ficheiros CSV grandes só para fazer validações iniciais.
SAMPLE_ROWS = 10


# Colunas mínimas que todos os logs de eventos devem ter.
# Estes ficheiros representam eventos temporais ligados a utilizador e computador.
LOG_FILES_REQUIRED_COLUMNS = {"id", "date", "user", "pc"}


# Ficheiros principais de eventos do CERT r4.2.
# Cada entrada associa o nome do ficheiro ao conjunto de colunas obrigatórias.
LOG_FILES = {
    "device.csv": LOG_FILES_REQUIRED_COLUMNS,
    "email.csv": LOG_FILES_REQUIRED_COLUMNS,
    "file.csv": LOG_FILES_REQUIRED_COLUMNS,
    "http.csv": LOG_FILES_REQUIRED_COLUMNS,
    "logon.csv": LOG_FILES_REQUIRED_COLUMNS,
}


# Alguns ficheiros têm coluna activity e valores esperados.
# Não aplicamos isto a todos os logs porque email, file e http já representam
# implicitamente o tipo de atividade.
ACTIVITY_VALIDATION = {
    "device.csv": {"connect", "disconnect"},
    "logon.csv": {"logon", "logoff"},
}


# O ficheiro psychometric.csv não é um log de eventos.
# Por isso, não deve ter de cumprir o schema id/date/user/pc.
PSYCHOMETRIC_REQUIRED_COLUMNS = {"user_id"}


# Os ficheiros LDAP também são dados de perfil, não eventos temporais.
# A coluna user_id será usada mais tarde para ligar estes dados aos logs.
LDAP_REQUIRED_COLUMNS = {"user_id"}


def main() -> None:
    # Validar primeiro se a diretoria base dos dados raw existe.
    validate_raw_cert_directory()

    # Validar os logs principais: device, email, file, http e logon.
    validate_log_files()

    # Validar o ficheiro psicométrico.
    validate_psychometric_file()

    # Validar se a pasta LDAP existe e contém ficheiros CSV válidos.
    validate_ldap_directory()

    # Validar se existem pastas de respostas e se cada uma contém CSVs válidos.
    validate_answers_files()

    print("Validation completed successfully.")


def validate_raw_cert_directory() -> None:
    """Validate that the raw CERT directory exists and is a directory."""

    print("Validating raw CERT directory ...")

    if not RAW_CERT_DIR.exists():
        raise FileNotFoundError(
            f"Raw CERT directory not found: {RAW_CERT_DIR}")

    if not RAW_CERT_DIR.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found a file: {RAW_CERT_DIR}")


def validate_log_files() -> None:
    """Validate the required CERT event log files."""

    print("Validating log files ...")

    for file_name, required_columns in LOG_FILES.items():
        file_path = RAW_CERT_DIR / file_name

        # Validar existência, cabeçalho, colunas obrigatórias e pelo menos uma linha de dados.
        check_csv_has_required_data(file_path, required_columns)

        # Validar, numa pequena amostra, se a coluna date é convertível para datetime.
        # A validação completa de todas as linhas deve ficar para o carregamento completo.
        check_sample_date_column(file_path)

        # Validar valores da coluna activity apenas nos ficheiros onde isso se aplica.
        if file_name in ACTIVITY_VALIDATION:
            check_activity_values(file_path, ACTIVITY_VALIDATION[file_name])


def validate_psychometric_file() -> None:
    """Validate the psychometric file."""

    print("Validating psychometric files ...")

    file_path = RAW_CERT_DIR / "psychometric.csv"

    # O ficheiro psicométrico deve ter pelo menos user_id para poder ser ligado aos logs.
    check_csv_has_required_data(file_path, PSYCHOMETRIC_REQUIRED_COLUMNS)

    # Validar numa amostra se user_id está preenchido.
    check_sample_required_values_not_empty(file_path, "user_id")


def validate_ldap_directory() -> None:
    """Validate the LDAP directory and its CSV files."""

    print("Validating LDAP files ...")

    ldap_dir = RAW_CERT_DIR / "LDAP"

    # Validar se a pasta LDAP existe e contém pelo menos um ficheiro CSV.
    csv_files = get_csv_files_from_directory(ldap_dir)

    # Validar cada ficheiro CSV dentro da pasta LDAP.
    for file_path in csv_files:
        check_csv_has_required_data(file_path, LDAP_REQUIRED_COLUMNS)
        check_sample_required_values_not_empty(file_path, "user_id")


def validate_answers_files() -> None:
    """Validate the CERT answers directory structure."""
    print("Validating answer files ...")

    # A pasta pode estar como answers ou Answers, dependendo da extração/dataset.
    answers_dir = find_answers_directory()

    # Todas as subpastas dentro de answers que começam com "r4.2".
    answer_dirs = [
        path
        for path in answers_dir.iterdir()
        if path.is_dir() and path.name.startswith("r4.2")
    ]

    # Validar se existem pastas de respostas para r4.2.
    if not answer_dirs:
        raise FileNotFoundError(
            f"No r4.2 answer directories found in: {answers_dir}")

    # Validar se cada pasta de respostas contém pelo menos um ficheiro CSV.
    # Nesta fase não validamos ainda o conteúdo completo das respostas.
    for answer_dir in sorted(answer_dirs):
        answer_files = get_csv_files_from_directory(answer_dir)

        # print(f"Validating answer directory '{answer_dir}' ...")

        for answer_file in answer_files:
            # print(f"Validating answer file '{answer_file}' ...")

            df = pd.read_csv(answer_file, header=None, names=[
                "event_type",
                "event_id",
                "date",
                "user_id",
                "pc",
                "value_1",  # A partir daqui são colunas que podem ou não existir, dependendo do tipo de evento/log
                "value_2",
                "value_3",
                "value_4",
                "value_5",
                "value_6",
                "value_7"
            ], dtype="string")

            if df.columns.empty or df.size == 0:
                raise ValueError(f"The file {answer_file.name} is empty")

            if df.columns.size < 6:
                raise ValueError(
                    f"The file {answer_file.name} doesn't have the expected amount of columns")

            # Validar as primeiras cinco colunas, que são comuns entre todos
            x = range(5)

            for i in x:
                column = df.columns[i]

                values = df[column].astype(str).str.strip()
                empty_rows_validation = values.isna() | values.eq("")

                if empty_rows_validation.any():
                    raise ValueError(
                        f"The file '{answer_file.name}' is missing data on the column '{column}'")


def find_answers_directory() -> Path:
    """Find the answers directory, accepting either 'answers' or 'Answers'."""

    candidates = [
        RAW_CERT_DIR.parent / "answers",
        RAW_CERT_DIR.parent / "Answers",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Answers directory not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def check_csv_has_required_data(file_path: Path, required_columns: set[str]) -> None:
    """
    Validate basic CSV structure without loading the full file.

    This function checks:
    - file exists;
    - path is really a file;
    - file has a header;
    - header has no empty column names;
    - header has no duplicate column names;
    - required columns exist;
    - file has at least one data row.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path.name} not found at: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            # Tentar ler o ficheiro usando o módulo csv para validar a estrutura inicial.
            # Isto é mais leve do que carregar o CSV completo com pandas.
            reader = csv.reader(file)

            # Ler a primeira linha, que deve conter o cabeçalho.
            header = next(reader, None)

            if not header:
                raise ValueError(f"File has no header: {file_path}")

            # Remover espaços antes/depois dos nomes das colunas.
            normalized_header = [column.strip() for column in header]

            # Validar se existem colunas sem nome.
            if any(column == "" for column in normalized_header):
                raise ValueError(f"File has empty column names: {file_path}")

            # Validar se existem colunas duplicadas.
            # Aqui usamos a lista original, não um set, porque o set remove duplicados.
            duplicate_columns = [
                column
                for column, count in Counter(normalized_header).items()
                if count > 1
            ]

            if duplicate_columns:
                raise ValueError(
                    f"Duplicate columns found in {file_path}: {duplicate_columns}"
                )

            # Converter para set só depois de validar duplicados.
            # O set é útil para comparar colunas obrigatórias de forma simples.
            columns = set(normalized_header)

            # Verificar se faltam colunas obrigatórias.
            missing_columns = required_columns - columns

            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {file_path}: {sorted(missing_columns)}"
                )

            # Tentar ler a segunda linha, ou seja, a primeira linha de dados.
            data_row = next(reader, None)

            if data_row is None:
                raise ValueError(f"File has no data rows: {file_path}")

    except UnicodeDecodeError as ex:
        raise ValueError(f"Could not decode CSV file: {file_path}") from ex
    except csv.Error as ex:
        raise ValueError(f"Could not parse CSV file: {file_path}") from ex


def check_sample_date_column(file_path: Path) -> None:
    """Validate that the date column is parseable in a small sample."""

    df = read_csv_sample(file_path)

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {file_path}")

    # Converter apenas a amostra para datetime.
    # errors="coerce" transforma valores inválidos em NaT, tornando mais difícil encontrar falhas.
    # Por isso, usamos errors="raise" para que qualquer valor inválido levante uma exceção imediatamente.
    parsed_dates = pd.to_datetime(df["date"], errors="raise")

    if parsed_dates.isna().any():
        # +2 porque:
        # - o índice do pandas começa em 0;
        # - a linha 1 do CSV é o cabeçalho;
        # - logo, o índice 0 corresponde à linha 2 do CSV.
        invalid_rows = parsed_dates[parsed_dates.isna()].index + 2

        raise ValueError(
            f"Invalid date values found in {file_path}. "
            f"Invalid CSV rows in sample: {invalid_rows.tolist()}"
        )


def check_activity_values(file_path: Path, allowed_values: set[str]) -> None:
    """Validate activity values in a small sample."""

    df = read_csv_sample(file_path)

    if "activity" not in df.columns:
        raise ValueError(f"Missing 'activity' column in {file_path}")

    # Normalizar os valores para evitar problemas com maiúsculas/minúsculas e espaços.
    activity = df["activity"].astype("string").str.strip().str.lower()

    # Um valor é inválido se estiver vazio ou se não pertencer à lista permitida.
    invalid_activity = activity.isna() | ~activity.isin(allowed_values)

    if invalid_activity.any():
        # +2 para ajustar o índice e considerar o cabeçalho.
        invalid_rows = df.loc[invalid_activity].index + 2
        invalid_values = sorted(
            activity.loc[invalid_activity].dropna().unique())

        raise ValueError(
            f"Invalid activity values found in {file_path}. "
            f"Allowed values: {sorted(allowed_values)}. "
            f"Invalid values: {invalid_values}. "
            f"Invalid CSV rows in sample: {invalid_rows.tolist()}"
        )


def check_sample_required_values_not_empty(file_path: Path, column_name: str) -> None:
    """Validate that a required column has values in a small sample."""

    df = read_csv_sample(file_path)

    if column_name not in df.columns:
        raise ValueError(f"Missing '{column_name}' column in {file_path}")

    # Converter para string e remover espaços para detetar valores vazios.
    values = df[column_name].astype("string").str.strip()

    invalid_values = values.isna() | (values == "")

    if invalid_values.any():
        # +2 para ajustar o índice e considerar o cabeçalho.
        invalid_rows = df.loc[invalid_values].index + 2

        raise ValueError(
            f"Empty values found in column '{column_name}' in {file_path}. "
            f"Invalid CSV rows in sample: {invalid_rows.tolist()}"
        )


def read_csv_sample(file_path: Path) -> pd.DataFrame:
    """
    Read only a small sample from a CSV file.

    This is used for lightweight validation. The full CSV files should only be
    loaded later in load_raw_cert.py, to avoid reading large files multiple times.
    """

    try:
        df = pd.read_csv(file_path, nrows=SAMPLE_ROWS, encoding="utf-8-sig")
    except UnicodeDecodeError as ex:
        raise ValueError(f"Could not decode CSV file: {file_path}") from ex
    except pd.errors.ParserError as ex:
        raise ValueError(f"Could not parse CSV file: {file_path}") from ex

    if df.empty:
        raise ValueError(f"CSV sample has no data rows: {file_path}")

    # Normalizar nomes das colunas lidas pelo pandas.
    df.columns = [str(column).strip() for column in df.columns]

    return df


# Fazer isto desta forma permite declarar as funções auxiliares depois da função main,
# o que pode ser mais organizado e fácil de ler.
# Além disso, a função main fica mais limpa e focada na lógica principal do programa.
if __name__ == "__main__":
    main()
