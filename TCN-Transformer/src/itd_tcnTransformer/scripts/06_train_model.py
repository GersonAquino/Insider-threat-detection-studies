import argparse


PARAMS_JSON: str | None


def main() -> None:
    from src.itd_tcnTransformer.config.settings import (
        RANDOM_SEED,
        CUBLAS_WORKSPACE_CONFIG,
    )
    from src.itd_tcnTransformer.utils.training_bootstrap import prepare_training_process

    prepare_training_process(RANDOM_SEED, CUBLAS_WORKSPACE_CONFIG)
    from src.itd_tcnTransformer.training import train

    print("Starting model training ...")
    params_override = (
        None
        if PARAMS_JSON is None
        else train.load_hpo_params_json(PARAMS_JSON)
    )
    train.main(params_override=params_override)
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Path to an HPO best_params.json file with training overrides.",
    )
    args = parser.parse_args()

    PARAMS_JSON = args.params_json
    main()
