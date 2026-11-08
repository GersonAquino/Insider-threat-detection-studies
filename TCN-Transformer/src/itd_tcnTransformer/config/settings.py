import torch

# FORMULAÇÃO DO PROBLEMA

TASK_TYPE = "binary"
BASE_UNIT = "user_day"

# Número de unidades de base (por exemplo, dias-utilizador) que o modelo observa para fazer uma previsão.
WINDOW_SIZE = 30

# Número de unidades de base a avançar a janela para a previsão seguinte. Um passo de 1 significa que fazemos uma previsão para cada janela possível, o que resulta em janelas sobrepostas e múltiplas previsões por dia-utilizador.
WINDOW_STEP = 1

PREDICT_ALL_DAYS = True
USE_CAUSAL_TRANSFORMER_MASK = False

MAX_MISSING_DAY_RATIO = 0.35
MAX_CONSECUTIVE_MISSING_DAYS = 5

# LABELS

LABEL_NORMAL = 0
LABEL_ANOMALOUS = 1
LABEL_SOURCE = "answers_events" # opções: "insiders", "answers_events"
LABEL_MODE = "event_day" # opções: "event_day", "attack_interval_day", "attack_interval_observed_day"

ALLOW_LABEL_DAYS_WITHOUT_FEATURES = False
ANSWER_LABEL_USER_SOURCE = "filename"  # opções: "filename", "event_user"

EXPECTED_CERT_R42_TOTAL_USERS = 1000
EXPECTED_CERT_R42_MALICIOUS_USERS = 70
EXPECTED_CERT_R42_SCENARIO_IDS = {1, 2, 3}

# REPRODUTIBILIDADE

RANDOM_SEED = 42 #+ 9
# A seed torna splits, inicialização do modelo, DataLoaders e treino mais
# reprodutíveis. Em CUDA, reprodutibilidade bit-a-bit pode não ser garantida em
# todos os ambientes devido a operações internas não determinísticas.
DETERMINISTIC_TRAINING = True
DETERMINISTIC_WARN_ONLY = True
CUDNN_BENCHMARK = False
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# FEATURES

VALID_PROFILE_FEATURE_MODES = {
    "none",
    "psychometric_only",
    "static_only",
    "all",
}

# Controla que subconjunto de features de perfil deve ser usado.
# Opções:
#   "none"               -> remove todas as features profile_*.
#   "psychometric_only"  -> mantém apenas features psicométricas profile_psychometric_*.
#   "static_only"        -> mantém apenas features estáticas de perfil, como função/equipa/departamento, removendo psicométricas.
#   "all"                -> mantém todas as features de perfil.
PROFILE_FEATURE_MODE = "none"

VALID_EXACT_DUPLICATE_FEATURE_SELECTION_SCOPES = {
    "scaler_fit",
}
VALID_EXACT_DUPLICATE_FEATURE_KEEP_POLICIES = {
    "base_priority",
}
EXACT_DUPLICATE_FEATURE_SELECTION_ENABLED = True
EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE = "scaler_fit"
EXACT_DUPLICATE_FEATURE_KEEP_POLICY = "base_priority"

# TREINO

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_STRATEGY = "user_article_two_phase"  # opções: "temporal", "user_article_two_phase"
TRAINING_MODE = "two_stage"  # opções: "single_stage", "two_stage"

NORMAL_PRETRAIN_USER_COUNT = 860

ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS = 56
ARTICLE_STAGE2_MALICIOUS_TEST_USERS = 14
ARTICLE_STAGE2_NORMAL_TRAIN_USERS = 56
ARTICLE_STAGE2_NORMAL_TEST_USERS = 14

FINETUNE_VALIDATION_RATIO = 0.15

ARTICLE_SPLIT_STRATIFY_BY_SCENARIO = True

BATCH_SIZE = 32
PRETRAIN_EPOCHS = 10
PRETRAIN_CHECKPOINT_STRATEGY = "last_epoch"  # opções: "last_epoch", "best_validation_auprc"
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Controla se a BCEWithLogitsLoss deve ponderar mais fortemente as etiquetas positivas
# para compensar o desequilíbrio de classes. POS_WEIGHT é normalmente calculado como
# n_negative / n_positive usando apenas a partição de treino.
# Usado apenas quando LOSS_TYPE = "bce"
USE_POS_WEIGHT = False

GRADIENT_CLIP_NORM = 1.0

# Arquitetura do modelo

MODEL_DIM = 128
HIDDEN_DIM = MODEL_DIM

TCN_ACTIVATION = "relu"  # relu/gelu
TCN_DILATIONS = [1, 2, 4, 8]
TCN_KERNEL_SIZE = 7
TCN_NORM_POSITION = "after_dropout_before_residual"
TCN_USE_LAYER_NORM = True

TRANSFORMER_ACTIVATION = "relu"
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_NUM_HEADS = 8  # HIDDEN_DIM deve ser divisível por TRANSFORMER_NUM_HEADS

# Dimensão feedforward alinhada com a tabela do artigo base.
TRANSFORMER_FF_DIM = MODEL_DIM

POSITIONAL_ENCODING_TYPE = "sinusoidal"

DROPOUT = 0.2

LOSS_TYPE = "focal"  # opções: "focal", "bce"

# Usado apenas quando LOSS_TYPE = "focal"
FOCAL_ALPHA = 0.8
FOCAL_GAMMA = 2.0

# AVALIAÇÃO

PRIMARY_METRIC = "val_auprc"

SECONDARY_METRICS = [
    "auroc",
    "precision",
    "recall",
    "f1",
    "confusion_matrix"
]

DEFAULT_THRESHOLD = 0.5

# Opções:
#   "f1" -> escolhe o threshold com melhor F1 na validação.
#   "recall_at_min_precision" -> maximiza recall mantendo uma precisão mínima.
THRESHOLD_SELECTION_METRIC = "recall_at_min_precision"
# Restrição na validação; não garante a mesma precision no teste.
MIN_THRESHOLD_SELECTION_PRECISION = 0.70
THRESHOLD_SELECTION_STEP = 0.001
FIXED_EVALUATION_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]
EVALUATION_UNIT = "window_timestep"  # opções: "window_timestep", "user_day_aggregated"
AGGREGATE_OVERLAPPING_PREDICTIONS = True

# Opções: "mean", "max", "min", "last". Isto será usado para agregar múltiplas previsões para o mesmo dia-utilizador quando PREDICT_ALL_DAYS é True e WINDOW_STEP < WINDOW_SIZE.
OVERLAPPING_PREDICTION_AGGREGATION = "mean"

# DISPOSITIVO

# Para permitir treino em CPU, usar: torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda"

# HPO / OPTUNA

OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_SECONDS = None
OPTUNA_DIRECTION = "maximize"
OPTUNA_STUDY_NAME = "tcn_transformer_hpo"
OPTUNA_OBJECTIVE_METRIC = "validation_auprc"
OPTUNA_MIN_PRECISION = MIN_THRESHOLD_SELECTION_PRECISION
OPTUNA_MAX_EPOCHS_PER_TRIAL = 20
OPTUNA_MAX_PRETRAIN_EPOCHS_PER_TRIAL = 10
OPTUNA_EARLY_STOPPING_PATIENCE = 5
OPTUNA_PRUNING_ENABLED = False
OPTUNA_SEED = RANDOM_SEED
OPTUNA_USE_FIXED_TRIAL_SEED = True
OPTUNA_TRIAL_SEED_OFFSET = 10000
OPTUNA_ENQUEUE_CURRENT_SETTINGS_TRIAL = True

OPTUNA_MODEL_DIM_VALUES = [96, 104, 128]  # opções anteriores: [64, 96, 104, 128]
OPTUNA_TRANSFORMER_NUM_HEADS_VALUES = [8]  # opções anteriores: [4, 8]
OPTUNA_TRANSFORMER_FF_DIM_VALUES = [104, 128, 192, 256]  # opções anteriores: [64, 104, 128, 192, 256]

OPTUNA_LEARNING_RATE_MIN = 1e-4 # LEARNING_RATE  # intervalo anterior: 1e-4
OPTUNA_LEARNING_RATE_MAX = 1e-3 # LEARNING_RATE  # intervalo anterior: 1e-3
OPTUNA_LEARNING_RATE_LOG = True

OPTUNA_WEIGHT_DECAY_MIN = 1e-5 #WEIGHT_DECAY  # intervalo anterior: 1e-5
OPTUNA_WEIGHT_DECAY_MAX = 1e-3 #WEIGHT_DECAY  # intervalo anterior: 1e-3
OPTUNA_WEIGHT_DECAY_LOG = True

OPTUNA_DROPOUT_MIN = 0.10 #DROPOUT  # intervalo anterior: 0.10
OPTUNA_DROPOUT_MAX = 0.30 #DROPOUT  # intervalo anterior: 0.30

OPTUNA_BATCH_SIZE_VALUES = [16, 32, 64] # [32]  # opções anteriores: [16, 32, 64]
OPTUNA_GRADIENT_CLIP_NORM_VALUES = [1.0, 2.0]  # opções anteriores: [0.5, 1.0, 2.0, 5.0]
OPTUNA_LOSS_TYPE_VALUES = ["focal"]  # opções: "focal", "bce"

OPTUNA_FOCAL_ALPHA_MIN = FOCAL_ALPHA  # intervalo anterior: 0.5
OPTUNA_FOCAL_ALPHA_MAX = FOCAL_ALPHA  # intervalo anterior: 0.95
OPTUNA_FOCAL_GAMMA_MIN = FOCAL_GAMMA  # intervalo anterior: 1.0
OPTUNA_FOCAL_GAMMA_MAX = FOCAL_GAMMA  # intervalo anterior: 4.0

OPTUNA_USE_POS_WEIGHT_VALUES = [False]  # opções: False, True


def validate_device(device: torch.device | str = DEVICE) -> None:
    """Check the requested runtime device only when processing is executed."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device!s} requires CUDA, but PyTorch cannot access CUDA. "
            "Check the CUDA environment or explicitly configure a CPU device."
        )

def _validate_settings() -> None:
    valid_activations = {"relu", "gelu"}
    valid_answer_label_user_sources = {"filename", "event_user"}
    valid_label_modes = {
        "event_day",
        "attack_interval_day",
        "attack_interval_observed_day",
    }
    valid_label_sources = {"insiders", "answers_events"}
    valid_positional_encodings = {"sinusoidal", "learned"}
    valid_losses = {"focal", "bce"}
    valid_evaluation_units = {"window_timestep", "user_day_aggregated"}
    valid_overlapping_prediction_aggregations = {"mean", "max", "min", "last"}
    valid_threshold_selection_metrics = {"f1", "recall_at_min_precision"}
    valid_split_strategies = {"temporal", "user_article_two_phase"}
    valid_training_modes = {"single_stage", "two_stage"}
    valid_tcn_norm_positions = {"after_dropout_before_residual"}
    valid_cublas_workspace_configs = {":4096:8", ":16:8"}
    valid_checkpoint_strategies = {"best_validation_auprc", "last_epoch"}
    valid_optuna_directions = {"maximize", "minimize"}
    valid_optuna_objective_metrics = {
        "validation_auprc",
        "validation_f1",
        "recall_at_min_precision",
    }

    if TASK_TYPE != "binary":
        raise ValueError(f"TASK_TYPE inválido: {TASK_TYPE}")

    if WINDOW_SIZE <= 0:
        raise ValueError("WINDOW_SIZE deve ser positivo.")

    if WINDOW_STEP <= 0:
        raise ValueError("WINDOW_STEP deve ser positivo.")

    if not isinstance(RANDOM_SEED, int) or isinstance(RANDOM_SEED, bool):
        raise TypeError("RANDOM_SEED deve ser um inteiro.")

    if RANDOM_SEED < 0:
        raise ValueError("RANDOM_SEED deve ser maior ou igual a 0.")

    if not isinstance(DETERMINISTIC_TRAINING, bool):
        raise TypeError("DETERMINISTIC_TRAINING deve ser booleano.")

    if not isinstance(DETERMINISTIC_WARN_ONLY, bool):
        raise TypeError("DETERMINISTIC_WARN_ONLY deve ser booleano.")

    if not isinstance(CUDNN_BENCHMARK, bool):
        raise TypeError("CUDNN_BENCHMARK deve ser booleano.")

    if CUBLAS_WORKSPACE_CONFIG not in valid_cublas_workspace_configs:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG inválido. "
            f"Opções válidas: {sorted(valid_cublas_workspace_configs)}."
        )

    if LABEL_SOURCE not in valid_label_sources:
        raise ValueError(f"LABEL_SOURCE inválido: {LABEL_SOURCE}")

    if LABEL_MODE not in valid_label_modes:
        raise ValueError(f"LABEL_MODE inválido: {LABEL_MODE}")

    if ANSWER_LABEL_USER_SOURCE not in valid_answer_label_user_sources:
        raise ValueError(f"ANSWER_LABEL_USER_SOURCE inválido: {ANSWER_LABEL_USER_SOURCE}")

    if LABEL_SOURCE == "insiders" and LABEL_MODE not in {
        "attack_interval_day",
        "attack_interval_observed_day",
    }:
        raise ValueError(
            'LABEL_MODE deve ser "attack_interval_day" ou '
            '"attack_interval_observed_day" quando LABEL_SOURCE = "insiders".'
        )

    if not EXPECTED_CERT_R42_SCENARIO_IDS:
        raise ValueError("EXPECTED_CERT_R42_SCENARIO_IDS não pode estar vazio.")

    if any(scenario_id not in {1, 2, 3} for scenario_id in EXPECTED_CERT_R42_SCENARIO_IDS):
        raise ValueError(
            "EXPECTED_CERT_R42_SCENARIO_IDS deve conter apenas cenários CERT r4.2 "
            f"válidos: {EXPECTED_CERT_R42_SCENARIO_IDS}."
        )

    if PROFILE_FEATURE_MODE not in VALID_PROFILE_FEATURE_MODES:
        raise ValueError(f"PROFILE_FEATURE_MODE inválido: {PROFILE_FEATURE_MODE}")

    if (
        EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE
        not in VALID_EXACT_DUPLICATE_FEATURE_SELECTION_SCOPES
    ):
        raise ValueError(
            "EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE inválido: "
            f"{EXACT_DUPLICATE_FEATURE_SELECTION_SCOPE}"
        )

    if (
        EXACT_DUPLICATE_FEATURE_KEEP_POLICY
        not in VALID_EXACT_DUPLICATE_FEATURE_KEEP_POLICIES
    ):
        raise ValueError(
            "EXACT_DUPLICATE_FEATURE_KEEP_POLICY inválido: "
            f"{EXACT_DUPLICATE_FEATURE_KEEP_POLICY}"
        )

    split_ratio_sum = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO
    if abs(split_ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_RATIO, VALIDATION_RATIO e TEST_RATIO devem somar 1. "
            f"Soma atual: {split_ratio_sum}."
        )

    if SPLIT_STRATEGY not in valid_split_strategies:
        raise ValueError(f"SPLIT_STRATEGY inválido: {SPLIT_STRATEGY}")

    if TRAINING_MODE not in valid_training_modes:
        raise ValueError(f"TRAINING_MODE inválido: {TRAINING_MODE}")

    if TRAINING_MODE == "two_stage" and SPLIT_STRATEGY != "user_article_two_phase":
        raise ValueError(
            'SPLIT_STRATEGY deve ser "user_article_two_phase" quando '
            'TRAINING_MODE = "two_stage".'
        )

    article_user_counts = {
        "NORMAL_PRETRAIN_USER_COUNT": NORMAL_PRETRAIN_USER_COUNT,
        "ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS": ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS,
        "ARTICLE_STAGE2_MALICIOUS_TEST_USERS": ARTICLE_STAGE2_MALICIOUS_TEST_USERS,
        "ARTICLE_STAGE2_NORMAL_TRAIN_USERS": ARTICLE_STAGE2_NORMAL_TRAIN_USERS,
        "ARTICLE_STAGE2_NORMAL_TEST_USERS": ARTICLE_STAGE2_NORMAL_TEST_USERS,
    }
    for name, value in article_user_counts.items():
        if value <= 0:
            raise ValueError(f"{name} deve ser positivo.")

    if FINETUNE_VALIDATION_RATIO <= 0 or FINETUNE_VALIDATION_RATIO >= 1:
        raise ValueError("FINETUNE_VALIDATION_RATIO deve estar entre 0 e 1.")

    malicious_stage2_total = (
        ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS +
        ARTICLE_STAGE2_MALICIOUS_TEST_USERS
    )
    if malicious_stage2_total != EXPECTED_CERT_R42_MALICIOUS_USERS:
        raise ValueError(
            "ARTICLE_STAGE2_MALICIOUS_TRAIN_USERS + "
            "ARTICLE_STAGE2_MALICIOUS_TEST_USERS deve ser "
            f"{EXPECTED_CERT_R42_MALICIOUS_USERS}. "
            f"Soma atual: {malicious_stage2_total}."
        )

    normal_stage2_total = (
        ARTICLE_STAGE2_NORMAL_TRAIN_USERS +
        ARTICLE_STAGE2_NORMAL_TEST_USERS
    )
    if normal_stage2_total != EXPECTED_CERT_R42_MALICIOUS_USERS:
        raise ValueError(
            "ARTICLE_STAGE2_NORMAL_TRAIN_USERS + "
            "ARTICLE_STAGE2_NORMAL_TEST_USERS deve ser "
            f"{EXPECTED_CERT_R42_MALICIOUS_USERS}. "
            f"Soma atual: {normal_stage2_total}."
        )

    if NORMAL_PRETRAIN_USER_COUNT + normal_stage2_total > 930:
        raise ValueError(
            "NORMAL_PRETRAIN_USER_COUNT + ARTICLE_STAGE2_NORMAL_TRAIN_USERS + "
            "ARTICLE_STAGE2_NORMAL_TEST_USERS deve ser <= 930."
        )

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE deve ser positivo.")

    if EVALUATION_UNIT not in valid_evaluation_units:
        raise ValueError(f"EVALUATION_UNIT inválido: {EVALUATION_UNIT}")

    if (
        EVALUATION_UNIT == "user_day_aggregated"
        and OVERLAPPING_PREDICTION_AGGREGATION not in valid_overlapping_prediction_aggregations
    ):
        raise ValueError(
            "OVERLAPPING_PREDICTION_AGGREGATION inválido: "
            f"{OVERLAPPING_PREDICTION_AGGREGATION}"
        )

    if PRETRAIN_EPOCHS <= 0:
        raise ValueError("PRETRAIN_EPOCHS deve ser positivo.")

    if PRETRAIN_CHECKPOINT_STRATEGY not in valid_checkpoint_strategies:
        raise ValueError(
            "PRETRAIN_CHECKPOINT_STRATEGY inválido: "
            f"{PRETRAIN_CHECKPOINT_STRATEGY}. "
            f"Opções válidas: {sorted(valid_checkpoint_strategies)}."
        )

    if MAX_EPOCHS <= 0:
        raise ValueError("MAX_EPOCHS deve ser positivo.")

    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE deve ser positivo.")

    if WEIGHT_DECAY < 0:
        raise ValueError("WEIGHT_DECAY não pode ser negativo.")

    if EARLY_STOPPING_PATIENCE <= 0:
        raise ValueError("EARLY_STOPPING_PATIENCE deve ser positivo.")

    if MODEL_DIM <= 0:
        raise ValueError("MODEL_DIM deve ser positivo.")

    if HIDDEN_DIM != MODEL_DIM:
        raise ValueError("HIDDEN_DIM deve ser igual a MODEL_DIM.")

    if TRANSFORMER_NUM_HEADS <= 0:
        raise ValueError("TRANSFORMER_NUM_HEADS deve ser positivo.")

    if MODEL_DIM % TRANSFORMER_NUM_HEADS != 0:
        raise ValueError(
            "MODEL_DIM deve ser divisível por TRANSFORMER_NUM_HEADS. "
            f"MODEL_DIM={MODEL_DIM}, TRANSFORMER_NUM_HEADS={TRANSFORMER_NUM_HEADS}."
        )

    if TRANSFORMER_NUM_LAYERS <= 0:
        raise ValueError("TRANSFORMER_NUM_LAYERS deve ser positivo.")

    if TRANSFORMER_FF_DIM <= 0:
        raise ValueError("TRANSFORMER_FF_DIM deve ser positivo.")

    if TCN_KERNEL_SIZE <= 0:
        raise ValueError("TCN_KERNEL_SIZE deve ser positivo.")

    if not TCN_DILATIONS:
        raise ValueError("TCN_DILATIONS não pode estar vazio.")

    if any(dilation <= 0 for dilation in TCN_DILATIONS):
        raise ValueError("Todos os valores de TCN_DILATIONS devem ser positivos.")

    if TCN_ACTIVATION not in valid_activations:
        raise ValueError(f"TCN_ACTIVATION inválido: {TCN_ACTIVATION}")

    if TRANSFORMER_ACTIVATION not in valid_activations:
        raise ValueError(f"TRANSFORMER_ACTIVATION inválido: {TRANSFORMER_ACTIVATION}")

    if TCN_NORM_POSITION not in valid_tcn_norm_positions:
        raise ValueError(f"TCN_NORM_POSITION inválido: {TCN_NORM_POSITION}")

    if POSITIONAL_ENCODING_TYPE not in valid_positional_encodings:
        raise ValueError(f"POSITIONAL_ENCODING_TYPE inválido: {POSITIONAL_ENCODING_TYPE}")

    if LOSS_TYPE not in valid_losses:
        raise ValueError(f"LOSS_TYPE inválido: {LOSS_TYPE}")

    if LOSS_TYPE == "focal" and USE_POS_WEIGHT:
        raise ValueError("USE_POS_WEIGHT deve estar False quando LOSS_TYPE='focal'.")

    if LOSS_TYPE == "focal":
        if FOCAL_ALPHA <= 0 or FOCAL_ALPHA >= 1:
            raise ValueError("FOCAL_ALPHA deve estar entre 0 e 1.")
        if FOCAL_GAMMA < 0:
            raise ValueError("FOCAL_GAMMA não pode ser negativo.")

    if DROPOUT < 0 or DROPOUT > 1:
        raise ValueError("DROPOUT deve estar entre 0 e 1.")

    if DEFAULT_THRESHOLD < 0 or DEFAULT_THRESHOLD > 1:
        raise ValueError("DEFAULT_THRESHOLD deve estar entre 0 e 1.")

    if THRESHOLD_SELECTION_METRIC not in valid_threshold_selection_metrics:
        raise ValueError(
            "THRESHOLD_SELECTION_METRIC inválido: "
            f"{THRESHOLD_SELECTION_METRIC}"
        )

    if (
        MIN_THRESHOLD_SELECTION_PRECISION < 0
        or MIN_THRESHOLD_SELECTION_PRECISION > 1
    ):
        raise ValueError(
            "MIN_THRESHOLD_SELECTION_PRECISION deve estar entre 0 e 1."
        )

    if not 0 < THRESHOLD_SELECTION_STEP <= 1:
        raise ValueError(
            "THRESHOLD_SELECTION_STEP deve ser maior que 0 e menor ou igual a 1."
        )

    if not FIXED_EVALUATION_THRESHOLDS:
        raise ValueError("FIXED_EVALUATION_THRESHOLDS não pode estar vazio.")

    if any(
        threshold < 0 or threshold > 1
        for threshold in FIXED_EVALUATION_THRESHOLDS
    ):
        raise ValueError(
            "Todos os valores de FIXED_EVALUATION_THRESHOLDS devem estar "
            "entre 0 e 1."
        )

    if len(set(FIXED_EVALUATION_THRESHOLDS)) != len(FIXED_EVALUATION_THRESHOLDS):
        raise ValueError("FIXED_EVALUATION_THRESHOLDS não pode ter duplicados.")

    if SPLIT_STRATEGY == "user_article_two_phase" and TRAINING_MODE != "two_stage":
        raise ValueError(
            'TRAINING_MODE deve ser "two_stage" quando '
            'SPLIT_STRATEGY = "user_article_two_phase".'
        )

    if OPTUNA_N_TRIALS <= 0:
        raise ValueError("OPTUNA_N_TRIALS deve ser positivo.")

    if OPTUNA_TIMEOUT_SECONDS is not None and OPTUNA_TIMEOUT_SECONDS <= 0:
        raise ValueError("OPTUNA_TIMEOUT_SECONDS deve ser positivo ou None.")

    if OPTUNA_DIRECTION not in valid_optuna_directions:
        raise ValueError(f"OPTUNA_DIRECTION inválido: {OPTUNA_DIRECTION}")

    if OPTUNA_OBJECTIVE_METRIC not in valid_optuna_objective_metrics:
        raise ValueError(
            "OPTUNA_OBJECTIVE_METRIC inválido: "
            f"{OPTUNA_OBJECTIVE_METRIC}"
        )

    if OPTUNA_MIN_PRECISION < 0 or OPTUNA_MIN_PRECISION > 1:
        raise ValueError("OPTUNA_MIN_PRECISION deve estar entre 0 e 1.")

    if OPTUNA_MAX_EPOCHS_PER_TRIAL <= 0:
        raise ValueError("OPTUNA_MAX_EPOCHS_PER_TRIAL deve ser positivo.")

    if OPTUNA_MAX_PRETRAIN_EPOCHS_PER_TRIAL <= 0:
        raise ValueError(
            "OPTUNA_MAX_PRETRAIN_EPOCHS_PER_TRIAL deve ser positivo."
        )

    if OPTUNA_EARLY_STOPPING_PATIENCE <= 0:
        raise ValueError("OPTUNA_EARLY_STOPPING_PATIENCE deve ser positivo.")

    if not isinstance(OPTUNA_PRUNING_ENABLED, bool):
        raise TypeError("OPTUNA_PRUNING_ENABLED deve ser booleano.")

    if not isinstance(OPTUNA_SEED, int) or isinstance(OPTUNA_SEED, bool):
        raise TypeError("OPTUNA_SEED deve ser um inteiro.")

    if OPTUNA_SEED < 0:
        raise ValueError("OPTUNA_SEED deve ser maior ou igual a 0.")

    if not isinstance(OPTUNA_USE_FIXED_TRIAL_SEED, bool):
        raise TypeError("OPTUNA_USE_FIXED_TRIAL_SEED deve ser booleano.")

    if not isinstance(OPTUNA_ENQUEUE_CURRENT_SETTINGS_TRIAL, bool):
        raise TypeError("OPTUNA_ENQUEUE_CURRENT_SETTINGS_TRIAL deve ser booleano.")

    if (
        not isinstance(OPTUNA_TRIAL_SEED_OFFSET, int)
        or isinstance(OPTUNA_TRIAL_SEED_OFFSET, bool)
    ):
        raise TypeError("OPTUNA_TRIAL_SEED_OFFSET deve ser um inteiro.")

    if OPTUNA_TRIAL_SEED_OFFSET < 0:
        raise ValueError("OPTUNA_TRIAL_SEED_OFFSET deve ser maior ou igual a 0.")

    optuna_numeric_ranges = {
        "OPTUNA_LEARNING_RATE": (
            OPTUNA_LEARNING_RATE_MIN,
            OPTUNA_LEARNING_RATE_MAX,
        ),
        "OPTUNA_WEIGHT_DECAY": (
            OPTUNA_WEIGHT_DECAY_MIN,
            OPTUNA_WEIGHT_DECAY_MAX,
        ),
        "OPTUNA_DROPOUT": (OPTUNA_DROPOUT_MIN, OPTUNA_DROPOUT_MAX),
        "OPTUNA_FOCAL_ALPHA": (
            OPTUNA_FOCAL_ALPHA_MIN,
            OPTUNA_FOCAL_ALPHA_MAX,
        ),
        "OPTUNA_FOCAL_GAMMA": (
            OPTUNA_FOCAL_GAMMA_MIN,
            OPTUNA_FOCAL_GAMMA_MAX,
        ),
    }
    for name, values in optuna_numeric_ranges.items():
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            for value in values
        ):
            raise TypeError(f"{name}_MIN/MAX devem ser valores numéricos.")

    optuna_positive_int_lists = {
        "OPTUNA_MODEL_DIM_VALUES": OPTUNA_MODEL_DIM_VALUES,
        "OPTUNA_TRANSFORMER_NUM_HEADS_VALUES": OPTUNA_TRANSFORMER_NUM_HEADS_VALUES,
        "OPTUNA_TRANSFORMER_FF_DIM_VALUES": OPTUNA_TRANSFORMER_FF_DIM_VALUES,
        "OPTUNA_BATCH_SIZE_VALUES": OPTUNA_BATCH_SIZE_VALUES,
    }
    for name, values in optuna_positive_int_lists.items():
        if not values:
            raise ValueError(f"{name} não pode estar vazio.")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise ValueError(f"Todos os valores de {name} devem ser inteiros positivos.")

    if not any(
        model_dim % num_heads == 0
        for model_dim in OPTUNA_MODEL_DIM_VALUES
        for num_heads in OPTUNA_TRANSFORMER_NUM_HEADS_VALUES
    ):
        raise ValueError(
            "Deve existir pelo menos uma combinação válida entre "
            "OPTUNA_MODEL_DIM_VALUES e OPTUNA_TRANSFORMER_NUM_HEADS_VALUES."
        )

    if not OPTUNA_GRADIENT_CLIP_NORM_VALUES:
        raise ValueError("OPTUNA_GRADIENT_CLIP_NORM_VALUES não pode estar vazio.")

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        for value in OPTUNA_GRADIENT_CLIP_NORM_VALUES
    ):
        raise ValueError(
            "Todos os valores de OPTUNA_GRADIENT_CLIP_NORM_VALUES devem ser "
            "numéricos positivos."
        )

    if OPTUNA_LEARNING_RATE_MIN <= 0 or OPTUNA_LEARNING_RATE_MAX <= 0:
        raise ValueError("OPTUNA_LEARNING_RATE_MIN/MAX devem ser positivos.")

    if OPTUNA_LEARNING_RATE_MIN > OPTUNA_LEARNING_RATE_MAX:
        raise ValueError(
            "OPTUNA_LEARNING_RATE_MIN deve ser menor ou igual a "
            "OPTUNA_LEARNING_RATE_MAX."
        )

    if not isinstance(OPTUNA_LEARNING_RATE_LOG, bool):
        raise TypeError("OPTUNA_LEARNING_RATE_LOG deve ser booleano.")

    if OPTUNA_WEIGHT_DECAY_MIN <= 0 or OPTUNA_WEIGHT_DECAY_MAX <= 0:
        raise ValueError("OPTUNA_WEIGHT_DECAY_MIN/MAX devem ser positivos.")

    if OPTUNA_WEIGHT_DECAY_MIN > OPTUNA_WEIGHT_DECAY_MAX:
        raise ValueError(
            "OPTUNA_WEIGHT_DECAY_MIN deve ser menor ou igual a "
            "OPTUNA_WEIGHT_DECAY_MAX."
        )

    if not isinstance(OPTUNA_WEIGHT_DECAY_LOG, bool):
        raise TypeError("OPTUNA_WEIGHT_DECAY_LOG deve ser booleano.")

    if (
        OPTUNA_DROPOUT_MIN < 0
        or OPTUNA_DROPOUT_MAX > 1
        or OPTUNA_DROPOUT_MIN > OPTUNA_DROPOUT_MAX
    ):
        raise ValueError(
            "OPTUNA_DROPOUT_MIN/MAX devem estar entre 0 e 1, com MIN <= MAX."
        )

    if not OPTUNA_LOSS_TYPE_VALUES:
        raise ValueError("OPTUNA_LOSS_TYPE_VALUES não pode estar vazio.")

    if any(loss_type not in valid_losses for loss_type in OPTUNA_LOSS_TYPE_VALUES):
        raise ValueError(
            "OPTUNA_LOSS_TYPE_VALUES deve conter apenas valores válidos de loss: "
            f"{sorted(valid_losses)}."
        )

    if (
        OPTUNA_FOCAL_ALPHA_MIN <= 0
        or OPTUNA_FOCAL_ALPHA_MAX >= 1
        or OPTUNA_FOCAL_ALPHA_MIN > OPTUNA_FOCAL_ALPHA_MAX
    ):
        raise ValueError(
            "OPTUNA_FOCAL_ALPHA_MIN/MAX devem estar entre 0 e 1, com MIN <= MAX."
        )

    if (
        OPTUNA_FOCAL_GAMMA_MIN < 0
        or OPTUNA_FOCAL_GAMMA_MAX < 0
        or OPTUNA_FOCAL_GAMMA_MIN > OPTUNA_FOCAL_GAMMA_MAX
    ):
        raise ValueError(
            "OPTUNA_FOCAL_GAMMA_MIN/MAX devem ser >= 0, com MIN <= MAX."
        )

    if not OPTUNA_USE_POS_WEIGHT_VALUES:
        raise ValueError("OPTUNA_USE_POS_WEIGHT_VALUES não pode estar vazio.")

    if any(not isinstance(value, bool) for value in OPTUNA_USE_POS_WEIGHT_VALUES):
        raise TypeError(
            "OPTUNA_USE_POS_WEIGHT_VALUES deve conter apenas valores booleanos."
        )

_validate_settings()
