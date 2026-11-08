import argparse


from src.itd_tcnTransformer.data import build_final_daily_dataset


def main() -> None:
    print("Starting final daily dataset building ...")
    build_final_daily_dataset.main()
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()
