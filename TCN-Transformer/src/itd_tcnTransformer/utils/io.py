import json

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path


def get_csv_files_from_directory(directory: Path) -> list[Path]:
    """Validate that a directory exists and contains at least one CSV file."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found a file: {directory}")

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in directory: {directory}")

    return csv_files


def make_json_safe(value):
    # Se se receber uma instância de classe (dataclass), converte-se em dicionário
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))

    # Se se receber um dicionário, convertem-se as chaves em string e usa-se recursividade neste método para converter os valores do dicionário
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    # Se se receber um set, converte-se numa lista ordenada de strings
    if isinstance(value, set):
        return sorted(str(item) for item in value)

    # Se se receber uma lista ou tuplo, convertem-se recursivamente os seus valores
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    # Se se receber uma data, converte-se em string no formato ISO
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, ensure_ascii=False, indent=2)
