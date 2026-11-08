import argparse


from src.itd_tcnTransformer.features import build_daily_features


def main() -> None:
    print("Starting daily feature building ...")
    build_daily_features.main()
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()
