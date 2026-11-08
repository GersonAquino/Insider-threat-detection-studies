import argparse

from src.itd_tcnTransformer.config.settings import (
    OPTUNA_EARLY_STOPPING_PATIENCE,
    OPTUNA_MAX_EPOCHS_PER_TRIAL,
    OPTUNA_MIN_PRECISION,
    OPTUNA_N_TRIALS,
    OPTUNA_OBJECTIVE_METRIC,
    OPTUNA_PRUNING_ENABLED,
    OPTUNA_SEED,
    OPTUNA_STUDY_NAME,
    OPTUNA_TIMEOUT_SECONDS,
    OPTUNA_USE_FIXED_TRIAL_SEED,
)
from src.itd_tcnTransformer.training.hpo import run_optuna_search


def main() -> None:
    args = _parse_args()
    pruning_enabled = (
        False
        if args.no_pruning
        else OPTUNA_PRUNING_ENABLED
    )
    use_fixed_trial_seed = (
        OPTUNA_USE_FIXED_TRIAL_SEED
        if args.fixed_trial_seed is None
        else bool(args.fixed_trial_seed)
    )

    run_optuna_search(
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        objective_metric=args.objective_metric,
        min_precision=args.min_precision,
        max_epochs_per_trial=args.max_epochs_per_trial,
        early_stopping_patience=args.early_stopping_patience,
        pruning_enabled=pruning_enabled,
        seed=args.seed,
        use_fixed_trial_seed=use_fixed_trial_seed,
    )
    print("Optuna search completed successfully!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS)
    parser.add_argument("--timeout", type=int, default=OPTUNA_TIMEOUT_SECONDS)
    parser.add_argument("--study-name", type=str, default=OPTUNA_STUDY_NAME)
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=OPTUNA_OBJECTIVE_METRIC,
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=OPTUNA_MIN_PRECISION,
    )
    parser.add_argument(
        "--max-epochs-per-trial",
        type=int,
        default=OPTUNA_MAX_EPOCHS_PER_TRIAL,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=OPTUNA_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument("--no-pruning", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=OPTUNA_SEED)
    parser.add_argument(
        "--fixed-trial-seed",
        action="store_true",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
