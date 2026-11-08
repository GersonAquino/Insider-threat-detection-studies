from pathlib import Path
import os

# Raiz do repositório:
# Assume que paths.py está localizado em src/itd_tcnTransformer/config
# parents[0] = config
# parents[1] = itd_tcnTransformer
# parents[2] = src
# parents[3] = raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _configured_root(variable_name: str, default_name: str) -> Path:
    """Resolve relative overrides from the repository root, independently of cwd."""
    configured = os.environ.get(variable_name)
    root = Path(configured).expanduser() if configured else PROJECT_ROOT / default_name
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


DATA_ROOT = _configured_root("TCN_DATA_ROOT", "data")
OUTPUT_ROOT = _configured_root("TCN_OUTPUT_ROOT", "outputs")
RAW_CERT_DIR = DATA_ROOT / "raw" / "CERT" / "r4.2"
PROCESSED_DIR = DATA_ROOT / "processed" / "itd_tcnTransformer"
OUTPUT_DIR = OUTPUT_ROOT / "itd_tcnTransformer"

# Subdiretorias de dados processados
INTERIM_DIR = PROCESSED_DIR / "interim"
DAILY_FEATURES_DIR = PROCESSED_DIR / "daily_features"
DAILY_LABELS_DIR = PROCESSED_DIR / "daily_labels"
FINAL_DAILY_DATASET_DIR = PROCESSED_DIR / "final_daily_dataset"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"
SPLITS_DIR = PROCESSED_DIR / "splits"
SCALERS_DIR = PROCESSED_DIR / "scalers"

# Subdiretorias de saída
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
HPO_DIR = OUTPUT_DIR / "hpo"
HPO_STUDY_DB_PATH = HPO_DIR / "optuna_study.db"
HPO_BEST_PARAMS_PATH = HPO_DIR / "best_params.json"
HPO_TRIALS_CSV_PATH = HPO_DIR / "trials.csv"
HPO_BEST_TRIAL_REPORT_PATH = HPO_DIR / "best_trial_report.json"
HPO_TRIALS_DIR = HPO_DIR / "trials"


def ensure_directories():
    """Garante que as diretorias necessárias existem."""
    directories = [
        # PROCESSED_DIR,
        # OUTPUT_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR,
        HPO_DIR,
        HPO_TRIALS_DIR,
        INTERIM_DIR,
        DAILY_FEATURES_DIR,
        DAILY_LABELS_DIR,
        FINAL_DAILY_DATASET_DIR,
        SEQUENCES_DIR,
        SPLITS_DIR,
        SCALERS_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
