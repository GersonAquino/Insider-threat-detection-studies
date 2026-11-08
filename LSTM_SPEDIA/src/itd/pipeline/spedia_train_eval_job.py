
# Executa treino/avaliação SPEDIA em processo isolado para libertar VRAM ao terminar.
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from .. import config
from .datasets import list_npz_files, make_tf_dataset
from .modeling import build_model
from .train_eval import (
    apply_probability_calibration,
    calibration_quality_report,
    fit_platt_scaler,
    maybe_keras_tuner_available,
    run_hpo,
    run_hpo_over_dataset_candidates,
    load_hpo_top_hyperparameters,
    normalize_dataset_choice_hparams,
    select_hparams_by_seed_robustness,
    select_hparams_by_seed_robustness_over_dataset_candidates,
    load_window_metadata,
    fit_segment_thresholds,
    eval_report_with_segment_thresholds,
    predict_probs,
    pick_threshold_by_min_precision,
    pick_threshold_by_best_f1,
    pick_threshold_by_best_balanced_accuracy,
    eval_report,
    save_json,
)
from .window_candidates import (
    candidate_search_space_from_manifest,
    find_candidate_in_manifest,
    is_window_dir_compatible,
    load_window_candidates_manifest,
)


def _configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("Nao foi possivel ativar memory growth:", gpu, "->", exc)
    return gpus


def _default_windows_out_dir() -> Path:
    return config.SPEDIA_WINDOWS_BASE_DIR / f"w={config.WINDOW_SIZE}_m={config.MAX_EVENTS_PER_WINDOW}"


def _default_windows_manifest_path() -> Path:
    return config.SPEDIA_WINDOWS_MANIFEST


def _count_samples(npz_files):
    n = 0
    for f in npz_files:
        with np.load(f, allow_pickle=False) as z:
            n += int(z["y"].shape[0])
    return n


def _dataset_without_sample_weights(ds):
    spec = ds.element_spec
    if isinstance(spec, tuple) and len(spec) == 3:
        return ds.map(lambda x, y, w: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _build_and_adapt_normalizers(train_files, max_events: int, train_ds_for_norm=None):
    normalizer = tf.keras.layers.Normalization(axis=-1, name="num_norm")
    window_normalizer = tf.keras.layers.Normalization(axis=-1, name="win_norm")

    if train_ds_for_norm is None:
        train_ds_for_norm = make_tf_dataset(
            train_files,
            config.BATCH_SIZE,
            shuffle=False,
            seed=config.RANDOM_SEED,
            max_events=max_events,
            include_sample_weights=False,
        )
    else:
        train_ds_for_norm = _dataset_without_sample_weights(train_ds_for_norm)

    valid_num = (
        train_ds_for_norm.unbatch()
        .flat_map(
            lambda x, y: tf.data.Dataset.from_tensor_slices(  # noqa: ARG005
                tf.boolean_mask(x["num"], x["event_id"] != 0)
            )
        )
        .batch(4096)
    )
    valid_win = (
        train_ds_for_norm.unbatch()
        .map(lambda x, y: x["win_num"])  # noqa: ARG005
        .batch(4096)
    )

    normalizer.adapt(valid_num)
    window_normalizer.adapt(valid_win)
    return normalizer, window_normalizer


def _sample_key(user_value: object, window_end_value: object) -> str:
    return f"{str(user_value)}||{str(window_end_value)}"


def _load_sample_keys(npz_files):
    keys = []
    for f in npz_files:
        with np.load(f, allow_pickle=True) as z:
            if "sample_key" in z.files:
                keys.extend(z["sample_key"].astype(str).tolist())
                continue
            y = z["y"]
            n = int(y.shape[0])
            users = z["user"].astype(str) if "user" in z.files else np.array([""] * n, dtype=object)
            w_end = z["window_end_ts"].astype(str) if "window_end_ts" in z.files else np.array([""] * n, dtype=object)
            keys.extend(_sample_key(users[i], w_end[i]) for i in range(n))
    return np.array(keys, dtype=object)


def _set_optimizer_learning_rate(optimizer, new_lr: float) -> float:
    lr_obj = optimizer.learning_rate
    if hasattr(lr_obj, "numpy"):
        old_lr = float(lr_obj.numpy())
    else:
        old_lr = float(tf.keras.backend.get_value(lr_obj))

    if hasattr(lr_obj, "assign"):
        lr_obj.assign(new_lr)
    else:
        optimizer.learning_rate = new_lr
    return old_lr


def _get_optimizer_learning_rate(optimizer) -> float:
    lr_obj = optimizer.learning_rate
    if hasattr(lr_obj, "numpy"):
        return float(lr_obj.numpy())
    return float(tf.keras.backend.get_value(lr_obj))


def _select_thresholds(y_true_val, y_prob_val):
    target_precision = float(config.MIN_PRECISION_FOR_THRESHOLD)

    thr_minprec = pick_threshold_by_min_precision(y_true_val, y_prob_val, target_precision)
    thr_f1 = pick_threshold_by_best_f1(y_true_val, y_prob_val)
    thr_balacc = pick_threshold_by_best_balanced_accuracy(y_true_val, y_prob_val)

    thresholds = {
        "min_precision": float(thr_minprec),
        "best_f1": float(thr_f1),
        "best_balanced_accuracy": float(thr_balacc),
    }

    if config.FIXED_THRESHOLD is not None:
        thr_final = float(config.FIXED_THRESHOLD)
        chosen_strategy = "fixed"
    else:
        strategy = str(config.THRESHOLD_STRATEGY).strip().lower()
        if strategy not in thresholds:
            raise ValueError(f"THRESHOLD_STRATEGY invalida: {strategy}. Usa uma de: {list(thresholds.keys())}")
        thr_final = float(thresholds[strategy])
        chosen_strategy = strategy

    y_pred_val_final = (np.asarray(y_prob_val) >= thr_final).astype(int)
    pred_pos = int(np.sum(y_pred_val_final))
    pred_neg = int(y_pred_val_final.size - pred_pos)
    if pred_pos == 0 or pred_neg == 0:
        print(
            "Aviso: threshold final degenerado na validacao "
            f"(pred_pos={pred_pos}, pred_neg={pred_neg}). "
            "Aplicar fallback para best_f1."
        )
        thr_final = float(thr_f1)
        chosen_strategy = f"{chosen_strategy}_fallback_best_f1"
    return thresholds, chosen_strategy, float(thr_final)


def _mine_hard_negative_bonus_by_key(model, train_files, threshold: float, max_events: int):
    train_eval_ds = make_tf_dataset(
        train_files,
        config.BATCH_SIZE,
        shuffle=False,
        seed=config.RANDOM_SEED,
        max_events=max_events,
        include_sample_weights=False,
    )
    y_true_train, y_prob_train = predict_probs(model, train_eval_ds)
    keys = _load_sample_keys(train_files)

    if len(keys) != len(y_true_train):
        raise RuntimeError("Desalinhamento entre chaves de treino e predições no hard negative mining.")

    fp_mask = (y_true_train == 0) & (y_prob_train >= float(threshold))
    fp_idx = np.where(fp_mask)[0]

    stats = {
        "enabled": bool(config.USE_HARD_NEGATIVE_MINING),
        "threshold_used": float(threshold),
        "fp_candidates": int(fp_idx.size),
        "fp_candidate_rate": float(fp_idx.size / len(y_true_train)) if len(y_true_train) else 0.0,
        "selected": 0,
        "selected_rate": 0.0,
        "top_fraction": float(config.HNM_TOP_FP_FRACTION),
    }

    min_fp = int(config.HNM_MIN_FP_COUNT)
    if fp_idx.size < min_fp:
        return {}, stats

    top_fraction = float(config.HNM_TOP_FP_FRACTION)
    top_fraction = min(max(top_fraction, 0.0), 1.0)
    if top_fraction <= 0.0:
        return {}, stats

    if top_fraction < 1.0:
        keep_n = max(min_fp, int(np.ceil(fp_idx.size * top_fraction)))
        keep_n = min(keep_n, int(fp_idx.size))
        order = np.argsort(y_prob_train[fp_idx])[::-1]
        selected_idx = fp_idx[order[:keep_n]]
    else:
        selected_idx = fp_idx

    selected_keys = keys[selected_idx].astype(str)
    selected_probs = y_prob_train[selected_idx].astype(float)
    stats["selected"] = int(selected_keys.size)
    stats["selected_rate"] = float(selected_keys.size / len(y_true_train)) if len(y_true_train) else 0.0
    if selected_idx.size:
        p_min = float(np.min(selected_probs))
        p_max = float(np.max(selected_probs))
        if np.isclose(p_max, p_min):
            norm_scores = np.ones_like(selected_probs, dtype=np.float32)
        else:
            norm_scores = ((selected_probs - p_min) / (p_max - p_min + 1e-12)).astype(np.float32)
        bonuses = (float(config.HARD_SEGMENT_FALSE_POS_BONUS) * norm_scores).astype(np.float32)
        bonus_by_key = {
            str(k): float(b)
            for k, b in zip(selected_keys.tolist(), bonuses.tolist())
        }

        stats["selected_prob_mean"] = float(np.mean(selected_probs))
        stats["selected_prob_min"] = p_min
        stats["selected_prob_max"] = p_max
        stats["selected_bonus_mean"] = float(np.mean(bonuses))
        stats["selected_bonus_min"] = float(np.min(bonuses))
        stats["selected_bonus_max"] = float(np.max(bonuses))
        return bonus_by_key, stats

    return {}, stats


def _build_candidate_context(out_dir: Path, window_size: str, max_events: int, use_weighted_train: bool):
    out_dir = Path(out_dir)
    if not is_window_dir_compatible(out_dir, window_size=window_size, max_events=max_events):
        raise RuntimeError(
            f"Shards SPEDIA em {out_dir} estao em formato antigo ou incompatível. "
            "Regera as janelas antes de treinar."
        )
    train_files = list_npz_files(out_dir / "train")
    val_files = list_npz_files(out_dir / "val")
    test_files = list_npz_files(out_dir / "test")
    if not train_files or not val_files or not test_files:
        raise FileNotFoundError(
            f"Shards SPEDIA nao encontrados em {out_dir}. Executa primeiro a geracao de janelas."
        )

    train_ds = make_tf_dataset(
        train_files,
        config.BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
        max_events=max_events,
        include_sample_weights=use_weighted_train,
    )
    val_ds = make_tf_dataset(
        val_files,
        config.BATCH_SIZE,
        shuffle=False,
        seed=config.RANDOM_SEED,
        max_events=max_events,
        include_sample_weights=False,
    )
    test_ds = make_tf_dataset(
        test_files,
        config.BATCH_SIZE,
        shuffle=False,
        seed=config.RANDOM_SEED,
        max_events=max_events,
        include_sample_weights=False,
    )

    normalizer, window_normalizer = _build_and_adapt_normalizers(
        train_files,
        max_events=max_events,
        train_ds_for_norm=train_ds,
    )

    n_train = _count_samples(train_files)
    n_val = _count_samples(val_files)
    n_test = _count_samples(test_files)
    steps_train = math.ceil(n_train / config.BATCH_SIZE)
    steps_val = math.ceil(n_val / config.BATCH_SIZE)
    steps_test = math.ceil(n_test / config.BATCH_SIZE)

    return {
        "window_size": str(window_size),
        "max_events": int(max_events),
        "out_dir": str(out_dir),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "normalizer": normalizer,
        "window_normalizer": window_normalizer,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "steps_train": int(steps_train),
        "steps_val": int(steps_val),
        "steps_test": int(steps_test),
    }


def _default_best_hp(window_size: str, max_events: int) -> dict:
    return {
        "emb_dim_event": 96,
        "emb_dim_aux": 24,
        "lstm_units": 96,
        "dense_units": 256,
        "dropout": 0.35,
        "recurrent_dropout": 0.0,
        "learning_rate": 1.5e-3,
        "weight_decay": 3.7e-4,
        "clipnorm": 1.0,
        "window_size": str(window_size),
        "max_events": int(max_events),
    }


def run_spedia_train_eval(windows_out: Path | None = None, windows_manifest: Path | None = None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gpus = _configure_gpu_memory_growth()
    print("GPUs:", gpus if gpus else "(nenhuma)")

    use_weighted_train = bool(config.USE_HARD_SEGMENT_WEIGHTING)
    monitor_metric = str(config.HPO_OBJECTIVE_METRIC).strip()
    monitor_mode = "min" if "loss" in monitor_metric.lower() else "max"
    hpo_out_dir = config.SPEDIA_HPO_DIR

    selected_ctx = None
    best_hp = None
    hpo_robust_summary = []
    window_candidates_report = []
    window_derivation = {}
    window_selection_mode = "fixed"

    if windows_manifest is not None:
        windows_manifest = Path(windows_manifest)
        manifest = load_window_candidates_manifest(windows_manifest)
        candidate_search_space = candidate_search_space_from_manifest(manifest)
        window_candidates_report = list(manifest.get("candidates", []))
        window_derivation = dict(manifest.get("derivation", {}))
        window_selection_mode = "dynamic_hpo"

        print("SPEDIA windows manifest:", windows_manifest)
        print("Candidate search space:", candidate_search_space)

        candidate_cache: dict[str, dict] = {}

        def candidate_context_getter(hp_dict: dict):
            cand = find_candidate_in_manifest(manifest, hp_dict["window_size"], int(hp_dict["max_events"]))
            if cand is None:
                raise RuntimeError(
                    f"Candidato nao encontrado no manifest: {hp_dict['window_size']} / {hp_dict['max_events']}"
                )
            key = str(cand["key"])
            if key not in candidate_cache:
                candidate_cache[key] = _build_candidate_context(
                    out_dir=Path(cand["out_dir"]),
                    window_size=str(cand["window_size"]),
                    max_events=int(cand["max_events"]),
                    use_weighted_train=use_weighted_train,
                )
            return candidate_cache[key]

        default_candidate = find_candidate_in_manifest(
            manifest,
            window_size=str(config.WINDOW_SIZE),
            max_events=int(config.MAX_EVENTS_PER_WINDOW),
        )
        if default_candidate is None:
            default_candidate = manifest["candidates"][0]

        best_hp = _default_best_hp(
            window_size=str(default_candidate["window_size"]),
            max_events=int(default_candidate["max_events"]),
        )

        def build_model_for_candidate_fn(hp_dict: dict, ctx: dict):
            return build_model(
                hp=hp_dict,
                normalizer=ctx["normalizer"],
                window_normalizer=ctx["window_normalizer"],
                max_events=ctx["max_events"],
            )

        if config.USE_HPO:
            if not maybe_keras_tuner_available():
                raise RuntimeError("keras-tuner nao esta instalado. Instala com: pip install -U keras-tuner")

            best_hp = run_hpo_over_dataset_candidates(
                build_model_for_candidate_fn=build_model_for_candidate_fn,
                candidate_search_space=candidate_search_space,
                candidate_context_getter=candidate_context_getter,
                out_dir=hpo_out_dir,
            )

            if config.USE_HPO_ROBUST_SELECTION:
                top_trials = load_hpo_top_hyperparameters(
                    hpo_out_dir,
                    config.HPO_ROBUST_TOP_K,
                    normalize_hp_fn=normalize_dataset_choice_hparams,
                )
                print("Top trials para robustez:", [t["trial"] for t in top_trials])
                robust_hp, hpo_robust_summary = select_hparams_by_seed_robustness_over_dataset_candidates(
                    build_model_for_candidate_fn=build_model_for_candidate_fn,
                    candidate_trials=top_trials,
                    candidate_context_getter=candidate_context_getter,
                )
                if robust_hp is not None:
                    best_hp = robust_hp

        selected_ctx = candidate_context_getter(best_hp)
    else:
        windows_out = Path(windows_out or _default_windows_out_dir())
        print("SPEDIA windows dir:", windows_out)
        selected_ctx = _build_candidate_context(
            out_dir=windows_out,
            window_size=str(config.WINDOW_SIZE),
            max_events=int(config.MAX_EVENTS_PER_WINDOW),
            use_weighted_train=use_weighted_train,
        )

        def build_model_fn(hp_dict):
            return build_model(
                hp=hp_dict,
                normalizer=selected_ctx["normalizer"],
                window_normalizer=selected_ctx["window_normalizer"],
                max_events=selected_ctx["max_events"],
            )

        best_hp = _default_best_hp(
            window_size=selected_ctx["window_size"],
            max_events=selected_ctx["max_events"],
        )

        if config.USE_HPO:
            if not maybe_keras_tuner_available():
                raise RuntimeError("keras-tuner nao esta instalado. Instala com: pip install -U keras-tuner")

            hp_only = run_hpo(
                build_model_fn,
                selected_ctx["train_ds"].repeat(),
                selected_ctx["val_ds"].repeat(),
                out_dir=hpo_out_dir,
                steps_per_epoch=selected_ctx["steps_train"],
                validation_steps=selected_ctx["steps_val"],
            )
            best_hp.update(hp_only)

            if config.USE_HPO_ROBUST_SELECTION:
                top_trials = load_hpo_top_hyperparameters(hpo_out_dir, config.HPO_ROBUST_TOP_K)
                print("Top trials para robustez:", [t["trial"] for t in top_trials])
                robust_hp, hpo_robust_summary = select_hparams_by_seed_robustness(
                    build_model_fn,
                    top_trials,
                    selected_ctx["train_ds"],
                    selected_ctx["val_ds"],
                    steps_per_epoch=selected_ctx["steps_train"],
                    validation_steps=selected_ctx["steps_val"],
                )
                if robust_hp is not None:
                    best_hp.update(robust_hp)

    print("Normalizadores adaptados.")
    print("n_train:", selected_ctx["n_train"], "| steps_train:", selected_ctx["steps_train"])
    print("n_val:", selected_ctx["n_val"], "| steps_val:", selected_ctx["steps_val"])
    print("Treino com hard segment weighting:", use_weighted_train)
    print("Window config final:", selected_ctx["window_size"], "| max_events:", selected_ctx["max_events"])
    print("HP final selecionado:", best_hp)

    run_dir = config.SPEDIA_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config.RANDOM_SEED, int):
        tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    gc.collect()
    model = build_model(
        hp=best_hp,
        normalizer=selected_ctx["normalizer"],
        window_normalizer=selected_ctx["window_normalizer"],
        max_events=selected_ctx["max_events"],
    )
    train_ds_fit = selected_ctx["train_ds"].repeat()
    val_ds_fit = selected_ctx["val_ds"].repeat()

    backup_cb = tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(run_dir / "backup")
    )
    callbacks = [
        backup_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=monitor_mode,
            factor=config.EARLY_ReduceLROnPlateau_FACTOR,
            patience=config.EARLY_ReduceLROnPlateau_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    hist = model.fit(
        train_ds_fit,
        validation_data=val_ds_fit,
        epochs=config.EPOCHS,
        steps_per_epoch=selected_ctx["steps_train"],
        validation_steps=selected_ctx["steps_val"],
        callbacks=callbacks,
        verbose=1,
    )

    y_true_val, y_prob_val = predict_probs(model, selected_ctx["val_ds"])
    y_true_test, y_prob_test = predict_probs(model, selected_ctx["test_ds"])

    pos_rate = float(np.mean(y_true_val))
    prec, rec, thr_curve = precision_recall_curve(y_true_val, y_prob_val)
    prec2, rec2 = prec[:-1], rec[:-1]

    max_prec = float(np.max(prec2))
    thr_at_max_prec = float(thr_curve[int(np.argmax(prec2))])
    rec_at_max_prec = float(rec2[int(np.argmax(prec2))])

    target_precision = float(config.MIN_PRECISION_FOR_THRESHOLD)
    print("Prevalencia (positivos) na VAL:", pos_rate)
    print("Precisao maxima possivel na VAL (curva PR):", max_prec)
    print("Threshold na precisao maxima:", thr_at_max_prec, "| recall nesse ponto:", rec_at_max_prec)

    ok_precision = prec2 >= target_precision
    print(f"Existe threshold com precision >= {target_precision}? :", bool(np.any(ok_precision)))
    if np.any(ok_precision):
        best_recall = float(np.max(rec2[ok_precision]))
        print("Melhor recall possivel respeitando min_precision:", best_recall)

    thresholds, chosen_strategy, thr_final = _select_thresholds(y_true_val, y_prob_val)
    print("Thresholds calculados:", thresholds)
    print("Estrategia escolhida:", chosen_strategy, "| thr_final:", thr_final)

    hard_negative_mining = {
        "enabled": bool(config.USE_HARD_NEGATIVE_MINING),
        "performed": False,
        "lr_multiplier": float(config.HNM_LR_MULTIPLIER),
        "fine_tune_epochs": int(config.HNM_FINE_TUNE_EPOCHS),
        "fine_tune_patience": int(config.HNM_FINE_TUNE_PATIENCE),
        "min_fp_count": int(config.HNM_MIN_FP_COUNT),
        "top_fp_fraction": float(config.HNM_TOP_FP_FRACTION),
    }
    if bool(config.USE_HARD_NEGATIVE_MINING):
        hard_negative_bonus_by_key, hnm_stats = _mine_hard_negative_bonus_by_key(
            model,
            selected_ctx["train_files"],
            thr_final,
            max_events=selected_ctx["max_events"],
        )
        hard_negative_mining.update(hnm_stats)

        if hard_negative_bonus_by_key:
            print(f"Hard negative mining: selecionadas {len(hard_negative_bonus_by_key)} amostras para fine-tune.")
            train_ds_hnm = make_tf_dataset(
                selected_ctx["train_files"],
                config.BATCH_SIZE,
                shuffle=True,
                seed=config.RANDOM_SEED,
                max_events=selected_ctx["max_events"],
                include_sample_weights=True,
                hard_negative_bonus_by_key=hard_negative_bonus_by_key,
            )

            try:
                current_lr = _get_optimizer_learning_rate(model.optimizer)
                new_lr = max(1e-6, current_lr * float(config.HNM_LR_MULTIPLIER))
                _set_optimizer_learning_rate(model.optimizer, new_lr)
                hard_negative_mining["lr_before"] = current_lr
                hard_negative_mining["lr_after"] = new_lr
            except Exception as exc:
                hard_negative_mining["lr_update_error"] = str(exc)

            hnm_callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    patience=int(config.HNM_FINE_TUNE_PATIENCE),
                    min_delta=config.EARLY_STOPPING_MIN_DELTA,
                    restore_best_weights=True,
                ),
            ]

            model.fit(
                train_ds_hnm.repeat(),
                validation_data=selected_ctx["val_ds"].repeat(),
                epochs=int(config.HNM_FINE_TUNE_EPOCHS),
                steps_per_epoch=selected_ctx["steps_train"],
                validation_steps=selected_ctx["steps_val"],
                callbacks=hnm_callbacks,
                verbose=1,
            )

            hard_negative_mining["performed"] = True

            y_true_val, y_prob_val = predict_probs(model, selected_ctx["val_ds"])
            y_true_test, y_prob_test = predict_probs(model, selected_ctx["test_ds"])
            thresholds, chosen_strategy, thr_final = _select_thresholds(y_true_val, y_prob_val)

            print("Thresholds recalculados após HNM:", thresholds)
            print("Estrategia final após HNM:", chosen_strategy, "| thr_final:", thr_final)
        else:
            print("Hard negative mining: sem candidatos suficientes para fine-tune.")

    y_prob_val_raw = np.asarray(y_prob_val).astype(float)
    y_prob_test_raw = np.asarray(y_prob_test).astype(float)
    calibrator = fit_platt_scaler(y_true_val, y_prob_val_raw)
    y_prob_val = apply_probability_calibration(y_prob_val_raw, calibrator)
    y_prob_test = apply_probability_calibration(y_prob_test_raw, calibrator)
    calibration = {
        "enabled": True,
        **calibrator,
        "artifact_path": str(run_dir / "probability_calibrator.json"),
        "val": calibration_quality_report(y_true_val, y_prob_val_raw, y_prob_val),
        "test": calibration_quality_report(y_true_test, y_prob_test_raw, y_prob_test),
    }

    thresholds, chosen_strategy, thr_final = _select_thresholds(y_true_val, y_prob_val)
    print("Thresholds recalculados após calibracao:", thresholds)
    print("Estrategia final após calibracao:", chosen_strategy, "| thr_final:", thr_final)

    val_by_strategy = {name: eval_report(y_true_val, y_prob_val, t) for name, t in thresholds.items()}
    test_by_strategy = {name: eval_report(y_true_test, y_prob_test, t) for name, t in thresholds.items()}

    segment_thresholding = {}
    if config.USE_SEGMENTED_THRESHOLD:
        segment_key = str(config.SEGMENT_THRESHOLD_KEY).strip()
        val_meta = load_window_metadata(selected_ctx["val_files"])
        test_meta = load_window_metadata(selected_ctx["test_files"])
        if segment_key not in val_meta or segment_key not in test_meta:
            raise ValueError(f"SEGMENT_THRESHOLD_KEY invalida: {segment_key}. Chaves: {list(val_meta.keys())}")

        seg_thr, seg_stats = fit_segment_thresholds(
            y_true=y_true_val,
            y_prob=y_prob_val,
            segment_values=val_meta[segment_key],
            min_precision=target_precision,
            min_samples=int(config.SEGMENT_THRESHOLD_MIN_SAMPLES),
            fallback_threshold=float(thresholds["min_precision"]),
        )

        val_rep_segmented = eval_report_with_segment_thresholds(
            y_true=y_true_val,
            y_prob=y_prob_val,
            segment_values=val_meta[segment_key],
            segment_thresholds=seg_thr,
            default_threshold=float(thresholds["min_precision"]),
        )
        test_rep_segmented = eval_report_with_segment_thresholds(
            y_true=y_true_test,
            y_prob=y_prob_test,
            segment_values=test_meta[segment_key],
            segment_thresholds=seg_thr,
            default_threshold=float(thresholds["min_precision"]),
        )
        segment_thresholding = {
            "enabled": True,
            "segment_key": segment_key,
            "segment_stats_val": seg_stats,
            "segment_thresholds": seg_thr,
            "Val_segmented": val_rep_segmented,
            "Test_segmented": test_rep_segmented,
        }
        print("Threshold segmentado calculado para:", segment_key)
        print("Val_segmented:", val_rep_segmented)
        print("Test_segmented:", test_rep_segmented)

    val_rep = eval_report(y_true_val, y_prob_val, thr_final)
    test_rep = eval_report(y_true_test, y_prob_test, thr_final)
    if config.USE_SEGMENTED_THRESHOLD and config.USE_SEGMENTED_THRESHOLD_FOR_FINAL and segment_thresholding:
        chosen_strategy = f"segmented:{segment_thresholding['segment_key']}"
        val_rep = segment_thresholding["Val_segmented"]
        test_rep = segment_thresholding["Test_segmented"]

    model.save(run_dir / "model.keras")
    save_json(run_dir / "best_hp.json", best_hp)
    save_json(run_dir / "probability_calibrator.json", calibrator)

    selected_window_config = {
        "mode": window_selection_mode,
        "window_size": str(selected_ctx["window_size"]),
        "max_events": int(selected_ctx["max_events"]),
        "out_dir": str(selected_ctx["out_dir"]),
        "n_train": int(selected_ctx["n_train"]),
        "n_val": int(selected_ctx["n_val"]),
        "n_test": int(selected_ctx["n_test"]),
        "steps_train": int(selected_ctx["steps_train"]),
        "steps_val": int(selected_ctx["steps_val"]),
        "steps_test": int(selected_ctx["steps_test"]),
    }

    report = {
        "objective_monitor": monitor_metric,
        "selected_window_config": selected_window_config,
        "window_derivation": window_derivation,
        "window_candidates": window_candidates_report,
        "hard_segment_weighting": {
            "enabled": use_weighted_train,
            "workhours_bonus": float(config.HARD_SEGMENT_WORKHOURS_BONUS),
            "len1_bonus": float(config.HARD_SEGMENT_LEN1_BONUS),
            "false_pos_bonus": float(config.HARD_SEGMENT_FALSE_POS_BONUS),
            "max_weight": float(config.HARD_SEGMENT_MAX_WEIGHT),
        },
        "hard_negative_mining": hard_negative_mining,
        "calibration": calibration,
        "hpo": {
            "use_hpo": bool(config.USE_HPO),
            "max_trials": int(config.HPO_MAX_TRIALS),
            "executions_per_trial": int(config.HPO_EXECUTIONS_PER_TRIAL),
            "robust_selection_enabled": bool(config.USE_HPO_ROBUST_SELECTION),
            "robust_top_k": int(config.HPO_ROBUST_TOP_K),
            "candidate_search_space": candidate_search_space_from_manifest({"candidates": window_candidates_report}) if window_candidates_report else {},
            "robust_summary": hpo_robust_summary,
        },
        "thresholds": {
            **thresholds,
            "min_precision_target": target_precision,
            "chosen": chosen_strategy,
            "final": float(thr_final),
            "probability_space": "calibrated",
        },
        "segment_thresholding": segment_thresholding,
        "Val_by_strategy": val_by_strategy,
        "Test_by_strategy": test_by_strategy,
        "Val": val_rep,
        "Test": test_rep,
    }
    save_json(config.SPEDIA_REPORT_PATH, report)
    save_json(config.SPEDIA_VAL_REPORT_PATH, val_rep)
    save_json(config.SPEDIA_TEST_REPORT_PATH, test_rep)

    print("Val (final):", val_rep)
    print("Test (final):", test_rep)
    print("Modelo guardado em:", run_dir)
    print("Report guardado em:", config.SPEDIA_REPORT_PATH)

    del model, hist, train_ds_fit, val_ds_fit, callbacks, backup_cb
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Treino/avaliacao SPEDIA em subprocesso.")
    parser.add_argument(
        "--windows-out",
        type=Path,
        default=None,
        help="Diretorio de shards SPEDIA (default: <SPEDIA_WINDOWS_BASE_DIR>/w=<WINDOW_SIZE>_m=<MAX_EVENTS_PER_WINDOW>).",
    )
    parser.add_argument(
        "--windows-manifest",
        type=Path,
        default=None,
        help="Manifest JSON com candidatos de janela/cap para o HPO dinamico SPEDIA.",
    )
    args = parser.parse_args()

    windows_manifest = args.windows_manifest.expanduser() if args.windows_manifest else None
    windows_out = args.windows_out.expanduser() if args.windows_out else None

    if windows_manifest is None and bool(config.SPEDIA_USE_DYNAMIC_WINDOW_HPO):
        default_manifest = _default_windows_manifest_path()
        if default_manifest.exists():
            windows_manifest = default_manifest

    if windows_manifest is not None:
        run_spedia_train_eval(windows_manifest=windows_manifest)
    else:
        run_spedia_train_eval(windows_out=(windows_out or _default_windows_out_dir()))


if __name__ == "__main__":
    main()
