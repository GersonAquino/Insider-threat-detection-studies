import argparse


from src.itd_tcnTransformer.sequences import build_sequences


def main() -> None:
    print("Starting sequence building ...")
    build_sequences.main()
    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()

    main()
