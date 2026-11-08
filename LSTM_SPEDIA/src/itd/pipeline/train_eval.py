


# Implementa treino, HPO (opcional), selecção de threshold e métricas detalhadas (matriz de confusão, AUROC, AUPRC, etc.)
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression

from .. import config

def maybe_keras_tuner_available() -> bool:
    try:
        import keras_tuner as kt  # noqa: F401
        return True
    except Exception:
        return False

def _monitor_mode(metric_name: str) -> str:
    metric_name = metric_name.lower()
    return "min" if "loss" in metric_name else "max"

def _training_callbacks(monitor: str, mode: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=config.EARLY_ReduceLROnPlateau_FACTOR,
            patience=config.EARLY_ReduceLROnPlateau_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

def _sample_model_hyperparameters(hp):
    return {
        "emb_dim_event": hp.Choice("emb_dim_event", list(config.HPO_SEARCH_EMB_EVENT)),
        "emb_dim_aux": hp.Choice("emb_dim_aux", list(config.HPO_SEARCH_EMB_AUX)),
        "lstm_units": hp.Choice("lstm_units", list(config.HPO_SEARCH_LSTM_UNITS)),
        "dense_units": hp.Choice("dense_units", list(config.HPO_SEARCH_DENSE_UNITS)),
        "dropout": hp.Float(
            "dropout",
            float(config.HPO_SEARCH_DROPOUT_MIN),
            float(config.HPO_SEARCH_DROPOUT_MAX),
            step=float(config.HPO_SEARCH_DROPOUT_STEP),
        ),
        "recurrent_dropout": 0.0, #hp.Choice("recurrent_dropout", [0.0, 0.1, 0.2]), # 0.0 para não perder cuDNN / performance
        "learning_rate": hp.Float(
            "learning_rate",
            float(config.HPO_SEARCH_LR_MIN),
            float(config.HPO_SEARCH_LR_MAX),
            sampling="log",
        ),
        "weight_decay": hp.Float(
            "weight_decay",
            float(config.HPO_SEARCH_WD_MIN),
            float(config.HPO_SEARCH_WD_MAX),
            sampling="log",
        ),
        "clipnorm": hp.Choice("clipnorm", list(config.HPO_SEARCH_CLIPNORM)),
    }

def run_hpo(build_model_fn, train_ds, val_ds, out_dir: Path, steps_per_epoch: int, validation_steps: int):
    import keras_tuner as kt

    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = str(config.HPO_OBJECTIVE_METRIC).strip()
    mode = _monitor_mode(monitor)

    def model_builder(hp):
        hp_dict = _sample_model_hyperparameters(hp)
        return build_model_fn(hp_dict)

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective(monitor, direction=mode),
        max_trials=config.HPO_MAX_TRIALS,
        executions_per_trial=config.HPO_EXECUTIONS_PER_TRIAL,
        directory=str(out_dir),
        project_name="hpo",
        overwrite=True,
    )

    callbacks = _training_callbacks(monitor, mode)

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    return tuner.get_best_hyperparameters(1)[0].values

def normalize_dataset_choice_hparams(hp_values: dict) -> dict:
    hp_dict = dict(hp_values)
    window_size = str(hp_dict.get("window_size", "")).strip()
    max_events = hp_dict.get("max_events")
    if max_events is None and window_size:
        conditional_key = f"max_events__{window_size}"
        max_events = hp_dict.get(conditional_key)

    cleaned = {
        k: v for k, v in hp_dict.items()
        if not str(k).startswith("max_events__")
    }
    if window_size:
        cleaned["window_size"] = window_size
    if max_events is not None:
        cleaned["max_events"] = int(max_events)
    return cleaned

def run_hpo_over_dataset_candidates(
    build_model_for_candidate_fn,
    candidate_search_space: dict[str, list[int]],
    candidate_context_getter,
    out_dir: Path,
):
    import keras_tuner as kt

    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = str(config.HPO_OBJECTIVE_METRIC).strip()
    mode = _monitor_mode(monitor)
    ordered_window_sizes = list(candidate_search_space.keys())
    if not ordered_window_sizes:
        raise ValueError("candidate_search_space vazio")

    class CandidateHyperModel(kt.HyperModel):
        def build(self, hp):
            window_size = hp.Choice("window_size", ordered_window_sizes)
            for ws in ordered_window_sizes:
                hp.Choice(
                    f"max_events__{ws}",
                    list(candidate_search_space[ws]),
                    parent_name="window_size",
                    parent_values=[ws],
                )

            hp_dict = normalize_dataset_choice_hparams(hp.values)
            hp_dict.update(_sample_model_hyperparameters(hp))
            if "window_size" not in hp_dict or "max_events" not in hp_dict:
                raise RuntimeError("Nao foi possivel resolver window_size/max_events para o trial atual")

            ctx = candidate_context_getter(hp_dict)
            return build_model_for_candidate_fn(hp_dict, ctx)

        def fit(self, hp, model, *args, **kwargs):
            hp_dict = normalize_dataset_choice_hparams(hp.values)
            ctx = candidate_context_getter(hp_dict)
            callbacks = list(kwargs.pop("callbacks", None) or []) + _training_callbacks(monitor, mode)
            epochs = int(kwargs.pop("epochs", config.EPOCHS))
            verbose = int(kwargs.pop("verbose", 1))
            return model.fit(
                ctx["train_ds"].repeat(),
                validation_data=ctx["val_ds"].repeat(),
                epochs=epochs,
                steps_per_epoch=ctx["steps_train"],
                validation_steps=ctx["steps_val"],
                callbacks=callbacks,
                verbose=verbose,
            )

    tuner = kt.RandomSearch(
        hypermodel=CandidateHyperModel(),
        objective=kt.Objective(monitor, direction=mode),
        max_trials=config.HPO_MAX_TRIALS,
        executions_per_trial=config.HPO_EXECUTIONS_PER_TRIAL,
        directory=str(out_dir),
        project_name="hpo",
        overwrite=True,
    )

    tuner.search(epochs=config.EPOCHS, verbose=1)
    best_hp = normalize_dataset_choice_hparams(tuner.get_best_hyperparameters(1)[0].values)
    return best_hp

def load_hpo_top_hyperparameters(out_dir: Path, top_k: int = 3, normalize_hp_fn=None):
    hpo_dir = out_dir / "hpo"
    if not hpo_dir.exists():
        return []

    trials = []
    for trial_json in sorted(hpo_dir.glob("trial_*/trial.json")):
        data = json.loads(trial_json.read_text(encoding="utf-8"))
        score = data.get("score")
        hp = data.get("hyperparameters", {}).get("values", {})
        status = data.get("status")
        if status != "COMPLETED" or not isinstance(score, (int, float)) or not hp:
            continue
        if normalize_hp_fn is not None:
            hp = normalize_hp_fn(hp)
        trials.append({
            "trial": trial_json.parent.name,
            "score": float(score),
            "hp": hp,
        })

    trials.sort(key=lambda x: x["score"], reverse=True)
    return trials[: max(1, int(top_k))]

def select_hparams_by_seed_robustness(
    build_model_fn,
    candidate_trials: list[dict],
    train_ds,
    val_ds,
    steps_per_epoch: int,
    validation_steps: int,
):
    if not candidate_trials:
        return None, []

    monitor = str(config.HPO_OBJECTIVE_METRIC).strip()
    mode = _monitor_mode(monitor)
    seeds = [int(s) for s in config.HPO_ROBUST_SEEDS]
    verbose = int(config.HPO_ROBUST_VERBOSE)

    best_hp = None
    best_value = None
    summary = []

    for cand in candidate_trials:
        hp = cand["hp"]
        seed_scores = []

        for seed in seeds:
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(seed)

            model = build_model_fn(hp)
            callbacks = _training_callbacks(monitor, mode)
            history = model.fit(
                train_ds.repeat(),
                validation_data=val_ds.repeat(),
                epochs=config.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=verbose,
            )

            metric_hist = history.history.get(monitor, [])
            if not metric_hist:
                raise RuntimeError(f"Métrica monitorizada não encontrada no histórico: {monitor}")

            score = float(np.max(metric_hist) if mode == "max" else np.min(metric_hist))
            seed_scores.append(score)

            del model, callbacks, history
            tf.keras.backend.clear_session()
            gc.collect()

        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        result = {
            "trial": cand["trial"],
            "hpo_score": float(cand["score"]),
            "seeds": seeds,
            "seed_scores": seed_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "hp": hp,
        }
        summary.append(result)

        if best_value is None:
            best_value = mean_score
            best_hp = hp
            continue

        is_better = mean_score > best_value if mode == "max" else mean_score < best_value
        if is_better:
            best_value = mean_score
            best_hp = hp

    summary.sort(key=lambda x: x["mean_score"], reverse=(mode == "max"))
    return best_hp, summary

def select_hparams_by_seed_robustness_over_dataset_candidates(
    build_model_for_candidate_fn,
    candidate_trials: list[dict],
    candidate_context_getter,
):
    if not candidate_trials:
        return None, []

    monitor = str(config.HPO_OBJECTIVE_METRIC).strip()
    mode = _monitor_mode(monitor)
    seeds = [int(s) for s in config.HPO_ROBUST_SEEDS]
    verbose = int(config.HPO_ROBUST_VERBOSE)

    best_hp = None
    best_value = None
    summary = []

    for cand in candidate_trials:
        hp = cand["hp"]
        ctx = candidate_context_getter(hp)
        seed_scores = []

        for seed in seeds:
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(seed)

            model = build_model_for_candidate_fn(hp, ctx)
            callbacks = _training_callbacks(monitor, mode)
            history = model.fit(
                ctx["train_ds"].repeat(),
                validation_data=ctx["val_ds"].repeat(),
                epochs=config.EPOCHS,
                steps_per_epoch=ctx["steps_train"],
                validation_steps=ctx["steps_val"],
                callbacks=callbacks,
                verbose=verbose,
            )

            metric_hist = history.history.get(monitor, [])
            if not metric_hist:
                raise RuntimeError(f"Métrica monitorizada não encontrada no histórico: {monitor}")

            score = float(np.max(metric_hist) if mode == "max" else np.min(metric_hist))
            seed_scores.append(score)

            del model, callbacks, history
            tf.keras.backend.clear_session()
            gc.collect()

        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        result = {
            "trial": cand["trial"],
            "hpo_score": float(cand["score"]),
            "seeds": seeds,
            "seed_scores": seed_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "hp": hp,
        }
        summary.append(result)

        if best_value is None:
            best_value = mean_score
            best_hp = hp
            continue

        is_better = mean_score > best_value if mode == "max" else mean_score < best_value
        if is_better:
            best_value = mean_score
            best_hp = hp

    summary.sort(key=lambda x: x["mean_score"], reverse=(mode == "max"))
    return best_hp, summary

def predict_probs(model, ds):
    y_true, y_prob = [], []
    for batch in ds:
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        p = model.predict(x, verbose=0).reshape(-1)
        y_prob.append(p)
        y_true.append(y.numpy().reshape(-1))
    return np.concatenate(y_true).astype(int), np.concatenate(y_prob).astype(float)


def fit_platt_scaler(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size:
        raise ValueError("y_true e y_prob devem ter o mesmo tamanho e nao podem estar vazios")
    if np.unique(y_true).size < 2:
        raise ValueError("Platt scaling requer ambas as classes na validacao")

    eps = float(config.PROBABILITY_CALIBRATION_EPS)
    p = np.clip(y_prob, eps, 1.0 - eps)
    score = np.log(p / (1.0 - p)).reshape(-1, 1)

    lr = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000,
        random_state=int(config.RANDOM_SEED),
    )
    lr.fit(score, y_true)

    return {
        "method": "platt",
        "fit_split": "spedia_val",
        "input_space": "logit_probability",
        "clip_eps": eps,
        "coef": float(lr.coef_.reshape(-1)[0]),
        "intercept": float(lr.intercept_.reshape(-1)[0]),
    }


def apply_probability_calibration(y_prob, calibrator: dict | None):
    if not calibrator:
        return np.asarray(y_prob).astype(float)

    method = str(calibrator.get("method", "")).strip().lower()
    if method != "platt":
        raise ValueError(f"Metodo de calibracao nao suportado: {method}")

    y_prob = np.asarray(y_prob).astype(float)
    eps = float(calibrator.get("clip_eps", config.PROBABILITY_CALIBRATION_EPS))
    coef = float(calibrator["coef"])
    intercept = float(calibrator["intercept"])

    p = np.clip(y_prob, eps, 1.0 - eps)
    score = np.log(p / (1.0 - p))
    z = np.clip((coef * score) + intercept, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_quality_report(y_true, y_prob_raw, y_prob_calibrated):
    y_true = np.asarray(y_true).astype(int)
    y_prob_raw = np.asarray(y_prob_raw).astype(float)
    y_prob_calibrated = np.asarray(y_prob_calibrated).astype(float)

    eps = float(config.PROBABILITY_CALIBRATION_EPS)
    raw = np.clip(y_prob_raw, eps, 1.0 - eps)
    cal = np.clip(y_prob_calibrated, eps, 1.0 - eps)

    return {
        "brier_score_raw": float(brier_score_loss(y_true, raw)),
        "brier_score_calibrated": float(brier_score_loss(y_true, cal)),
        "log_loss_raw": float(log_loss(y_true, raw, labels=[0, 1])),
        "log_loss_calibrated": float(log_loss(y_true, cal, labels=[0, 1])),
    }

def pick_threshold_by_min_precision(y_true, y_prob, min_precision: float):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Alinhar dimensões: thresholds tem menos 1 elemento
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Robustez contra NaN/inf
    precisions = np.nan_to_num(precisions, nan=0.0, posinf=1.0, neginf=0.0)
    recalls = np.nan_to_num(recalls, nan=0.0, posinf=1.0, neginf=0.0)

    if thresholds.size == 0:
        # caso extremo (ex.: y_prob constante)
        return 0.5

    ok = precisions >= min_precision

    if not np.any(ok):
        # fallback: escolhe o threshold com melhor precisão (e em empate, maior threshold)
        best_prec = precisions.max()
        cand = thresholds[precisions >= best_prec - 1e-12]
        return float(cand.max())

    best_recall = recalls[ok].max()

    # entre os que atingem o melhor recall possível mantendo a precisão mínima,
    # escolhe o MAIOR threshold (tende a reduzir falsos positivos)
    cand = thresholds[ok & (recalls >= best_recall - 1e-12)]
    return float(cand.max()) if cand.size else float(thresholds[ok].max())

def pick_threshold_by_best_f1(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1 = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])

def pick_threshold_by_best_balanced_accuracy(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    bal_acc = 0.5 * (tpr + (1.0 - fpr))
    best = np.nanmax(bal_acc)

    cand = thresholds[np.isclose(bal_acc, best, atol=1e-12)]
    cand = cand[np.isfinite(cand)]
    if cand.size == 0:
        return 0.5
    return float(cand.max())

def _seq_len_bin(v: int) -> str:
    if v <= 1:
        return "len=1"
    if v <= 3:
        return "len=2-3"
    if v <= 7:
        return "len=4-7"
    if v <= 15:
        return "len=8-15"
    return "len>=16"

def _hour_bin(h: int) -> str:
    if h < 0:
        return "unknown"
    if h <= 6:
        return "00-06"
    if h <= 12:
        return "07-12"
    if h <= 18:
        return "13-18"
    return "19-23"

def load_window_metadata(npz_files):
    out = {
        "user": [],
        "window_end_ts": [],
        "hour": [],
        "dow": [],
        "is_offhours": [],
        "is_weekend": [],
        "seq_len": [],
        "hour_bin": [],
        "seq_len_bin": [],
    }

    for f in npz_files:
        with np.load(f, allow_pickle=True) as z:
            x_event = z["X_event"]
            n = int(x_event.shape[0])

            users = z["user"].astype(str) if "user" in z.files else np.array([""] * n, dtype=object)
            w_end = z["window_end_ts"].astype(str) if "window_end_ts" in z.files else np.array([""] * n, dtype=object)

            seq_len = z["seq_len"].astype(np.int32) if "seq_len" in z.files else (x_event != 0).sum(axis=1).astype(np.int32)
            w_end_ts = pd.to_datetime(pd.Series(w_end), errors="coerce")
            hour = w_end_ts.dt.hour.fillna(-1).astype(int).to_numpy()
            dow = w_end_ts.dt.dayofweek.fillna(-1).astype(int).to_numpy()

            is_offhours = ((hour < 7) | (hour >= 19)).astype(np.int8)
            is_weekend = (dow >= 5).astype(np.int8)
            hour_bin = np.array([_hour_bin(int(h)) for h in hour], dtype=object)
            seq_len_bin = np.array([_seq_len_bin(int(v)) for v in seq_len], dtype=object)

            out["user"].append(users)
            out["window_end_ts"].append(w_end)
            out["hour"].append(hour)
            out["dow"].append(dow)
            out["is_offhours"].append(is_offhours)
            out["is_weekend"].append(is_weekend)
            out["seq_len"].append(seq_len)
            out["hour_bin"].append(hour_bin)
            out["seq_len_bin"].append(seq_len_bin)

    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}

def fit_segment_thresholds(
    y_true,
    y_prob,
    segment_values,
    min_precision: float,
    min_samples: int = 200,
    fallback_threshold: float = 0.5,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    seg = np.asarray(segment_values).astype(str)

    if not (len(y_true) == len(y_prob) == len(seg)):
        raise ValueError("y_true, y_prob e segment_values devem ter o mesmo tamanho")

    thresholds = {}
    stats = {}
    for sval in sorted(np.unique(seg).tolist()):
        mask = (seg == sval)
        n = int(np.sum(mask))
        if n < int(min_samples):
            thresholds[sval] = float(fallback_threshold)
            stats[sval] = {"n": n, "used_fallback": True}
            continue

        ys = y_true[mask]
        ps = y_prob[mask]
        if np.unique(ys).size < 2:
            thresholds[sval] = float(fallback_threshold)
            stats[sval] = {"n": n, "used_fallback": True}
            continue

        thresholds[sval] = float(pick_threshold_by_min_precision(ys, ps, float(min_precision)))
        stats[sval] = {"n": n, "used_fallback": False}

    return thresholds, stats

def predict_with_segment_thresholds(y_prob, segment_values, segment_thresholds: dict, default_threshold: float):
    y_prob = np.asarray(y_prob).astype(float)
    seg = np.asarray(segment_values).astype(str)

    thr = np.full(y_prob.shape, float(default_threshold), dtype=float)
    for sval, t in segment_thresholds.items():
        thr[seg == str(sval)] = float(t)

    y_pred = (y_prob >= thr).astype(int)
    return y_pred, thr

def eval_report_with_segment_thresholds(
    y_true,
    y_prob,
    segment_values,
    segment_thresholds: dict,
    default_threshold: float,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred, applied_thr = predict_with_segment_thresholds(
        y_prob=y_prob,
        segment_values=segment_values,
        segment_thresholds=segment_thresholds,
        default_threshold=default_threshold,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = float(tn / (tn + fp + 1e-12))
    fpr = float(fp / (fp + tn + 1e-12))

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "threshold_mode": "segmented",
        "default_threshold": float(default_threshold),
        "segment_thresholds": {str(k): float(v) for k, v in segment_thresholds.items()},
        "applied_threshold_min": float(np.min(applied_thr)) if applied_thr.size else float(default_threshold),
        "applied_threshold_max": float(np.max(applied_thr)) if applied_thr.size else float(default_threshold),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }

def eval_report(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = float(tn / (tn + fp + 1e-12))
    fpr = float(fp / (fp + tn + 1e-12))

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
