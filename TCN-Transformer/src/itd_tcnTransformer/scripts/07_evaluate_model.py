import argparse

from src.itd_tcnTransformer.evaluation import evaluate


def main() -> None:
    print("Starting model evaluation ...")
    evaluate.run_evaluation()
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()
