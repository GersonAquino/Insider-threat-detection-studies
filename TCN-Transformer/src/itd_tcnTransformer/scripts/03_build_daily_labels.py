import argparse


from src.itd_tcnTransformer.labels import build_daily_labels


def main() -> None:
    print("Starting daily label building ...")
    build_daily_labels.main()
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()
