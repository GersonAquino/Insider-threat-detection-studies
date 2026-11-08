


# Central SPEDIA configuration: paths, hyperparameters, hashing and temporal splitting.
from __future__ import annotations

import os
from pathlib import Path

# Resolve relative input/output paths against this study directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: Path) -> Path:
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


SPEDIA_EXPERIMENT_TAG = "balanced50_test"
SPEDIA_CSV_PATH = _configured_path(
    "SPEDIA_CSV_PATH",
    PROJECT_ROOT / "data" / "raw" / "SPEDIA" / "logs_SPEDIA_annotated_en_extended_6m_x6_balanced.csv",
)

# Saídas
OUTPUTS_DIR = _configured_path("SPEDIA_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# ====== Reprodutibilidade ======
RANDOM_SEED = 1337

# ====== Janela temporal ======
WINDOW_SIZE = "90min"
MAX_EVENTS_PER_WINDOW = 150
SPEDIA_USE_DYNAMIC_WINDOW_HPO = False
SPEDIA_HPO_WINDOW_SIZES = ["90min"] # "30min", "60min", "120min"
SPEDIA_DYNAMIC_CAP_QUANTILES = [0.995, 0.999]
SPEDIA_DYNAMIC_CAP_ROUND_STEP = 25
SPEDIA_DYNAMIC_CAP_MIN_GAP = 25
SPEDIA_WINDOWS_BASE_DIR = OUTPUTS_DIR / f"spedia_windows_{SPEDIA_EXPERIMENT_TAG}"
SPEDIA_WINDOWS_MANIFEST = SPEDIA_WINDOWS_BASE_DIR / "candidate_manifest.json"
SPEDIA_HPO_DIR = OUTPUTS_DIR / "hpo" / f"spedia_{SPEDIA_EXPERIMENT_TAG}"
SPEDIA_RUN_DIR = MODELS_DIR / f"spedia_tf_lstm_windows_binary_{SPEDIA_EXPERIMENT_TAG}"
SPEDIA_REPORT_PATH = REPORTS_DIR / f"spedia_report_{SPEDIA_EXPERIMENT_TAG}.json"
SPEDIA_VAL_REPORT_PATH = REPORTS_DIR / f"spedia_val_report_{SPEDIA_EXPERIMENT_TAG}.json"
SPEDIA_TEST_REPORT_PATH = REPORTS_DIR / f"spedia_test_report_{SPEDIA_EXPERIMENT_TAG}.json"

# ====== Features numéricas (ordem canónica) ======
NUM_FEATURE_NAMES = [
    "delta_t_log1p",
    "prev_user_gap_log1p",
    "attachments_count", "is_missing_attachments",
    "size_log1p", "is_missing_size",
    "size_before_log1p", "is_missing_size_before",
    "size_after_log1p", "is_missing_size_after",
    "size_delta_log1p", "is_missing_size_delta",
]
NUM_FEATURES = len(NUM_FEATURE_NAMES)

# ====== Features agregadas por janela ======
WIN_FEATURE_NAMES = [
    "seq_len_log1p",
    "events_per_min",
    "unique_event_ratio",
    "unique_from_ratio",
    "unique_to_ratio",
    "unique_url_ratio",
    "unique_ext_ratio",
    "mean_delta_t_log1p",
    "std_delta_t_log1p",
    "mean_attachments_count",
    "missing_attachments_ratio",
    "mean_size_log1p",
    "missing_size_ratio",
    "mean_size_delta_log1p",
    "off_hours_ratio",
    "weekend_ratio",
    "user_hist_window_count_log1p",
    "user_events_per_min_ratio_log1p",
    "user_offhours_ratio_delta",
    "user_weekend_ratio_delta",
    "user_new_event_ratio",
    "user_new_from_ratio",
    "user_new_to_ratio",
    "user_new_url_ratio",
    "user_new_ext_ratio",
]
WIN_FEATURES = len(WIN_FEATURE_NAMES)

# ====== Feature hashing (bucketização estável) ======
# 0 é reservado para padding nas sequências
HASH_BINS_EVENT = 16384
HASH_BINS_DOMAIN = 65536
HASH_BINS_URL = 65536
HASH_BINS_EXT = 4096

# ====== Split temporal por fracções do intervalo total ======
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # resto é teste

# ====== Treino ======
BATCH_SIZE = 64
USE_SEQUENCE_LENGTH_BUCKETING = True
SEQUENCE_LENGTH_BUCKET_BOUNDARIES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160]
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.0008  # ganho mínimo na métrica monitorizada para contar como melhoria
EARLY_ReduceLROnPlateau_FACTOR = 0.5
EARLY_ReduceLROnPlateau_PATIENCE = 3

# ====== HPO (Hyperparameter Optimisation; otimização de hiperparâmetros) ======
USE_HPO = True
HPO_MAX_TRIALS = 30
HPO_EXECUTIONS_PER_TRIAL = 2
HPO_OBJECTIVE_METRIC = "val_auprc"  # "val_f1" | "val_balanced_accuracy" | "val_auprc" | "val_auroc"
HPO_SEARCH_EMB_EVENT = [64, 96, 128]
HPO_SEARCH_EMB_AUX = [24, 32]
HPO_SEARCH_LSTM_UNITS = [64, 80, 96, 112, 128, 160]
HPO_SEARCH_DENSE_UNITS = [128, 256]
HPO_SEARCH_DROPOUT_MIN = 0.20
HPO_SEARCH_DROPOUT_MAX = 0.40
HPO_SEARCH_DROPOUT_STEP = 0.05
HPO_SEARCH_LR_MIN = 2e-4
HPO_SEARCH_LR_MAX = 3e-3
HPO_SEARCH_WD_MIN = 1e-5
HPO_SEARCH_WD_MAX = 1e-3
HPO_SEARCH_CLIPNORM = [0.5, 1.0]
USE_HPO_ROBUST_SELECTION = True
HPO_ROBUST_TOP_K = 3
HPO_ROBUST_SEEDS = [1337, 2027, 4242]
HPO_ROBUST_VERBOSE = 0

# ====== Selecção de threshold na validação ======
FIXED_THRESHOLD = None
THRESHOLD_STRATEGY = "min_precision"  # "min_precision" | "best_f1" | "best_balanced_accuracy"
MIN_PRECISION_FOR_THRESHOLD = 0.80
PROBABILITY_CALIBRATION_METHOD = "platt"
PROBABILITY_CALIBRATION_EPS = 1e-6

# ====== Threshold segmentado (opcional) ======
USE_SEGMENTED_THRESHOLD = True
SEGMENT_THRESHOLD_KEY = "is_offhours"  # "is_offhours" | "is_weekend" | "hour_bin" | "seq_len_bin" | "user"
SEGMENT_THRESHOLD_MIN_SAMPLES = 200
USE_SEGMENTED_THRESHOLD_FOR_FINAL = False

# ====== Reweighting de segmentos difíceis no treino ======
# Objetivo: dar mais peso a amostras de horário laboral (is_offhours=0)
# e janelas muito curtas (seq_len=1), que foram os segmentos com mais erro.
USE_HARD_SEGMENT_WEIGHTING = True
HARD_SEGMENT_WORKHOURS_BONUS = 0.30
HARD_SEGMENT_LEN1_BONUS = 0.0
HARD_SEGMENT_FALSE_POS_BONUS = 0.25
HARD_SEGMENT_MAX_WEIGHT = 1.90

# ====== Hard Negative Mining (foco em falsos positivos) ======
USE_HARD_NEGATIVE_MINING = False
HNM_MIN_FP_COUNT = 200
HNM_TOP_FP_FRACTION = 0.50
HNM_FINE_TUNE_EPOCHS = 4
HNM_FINE_TUNE_PATIENCE = 2
HNM_LR_MULTIPLIER = 0.5
