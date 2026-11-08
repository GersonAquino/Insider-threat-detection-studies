import argparse


from src.itd_tcnTransformer.data import explore_loaded_cert, load_raw_cert, validate_raw_cert

LOAD_KEEP_OLD_FILES: bool
LOAD_ROW_CHUNK_SIZE: int
SKIP_EXPLORATION: bool
SKIP_VALIDATION: bool


def main() -> None:
    if not SKIP_VALIDATION:
        print("Starting data validation ...")
        validate_raw_cert.main()
        # print("Starting data completed.")

    # print("Loading raw data ...")
    load_raw_cert.main(LOAD_KEEP_OLD_FILES, LOAD_ROW_CHUNK_SIZE)
    # print("Raw data loaded.")

    if not SKIP_EXPLORATION:
        print("Starting data exploration ...")
        explore_loaded_cert.main()
        # print("Data exploration completed.")

    print("Script executed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load-keep-old-files",
        # action=argparse.BooleanOptionalAction # Isto cria automaticamente a versão negativa da flag ("--no-keep-old-files"), mas eu não quero isso
        action="store_true",
        default=False,
        help="Use the last parquet files. Still generates new ones if missing, this allows for resuming progress."
    )

    parser.add_argument(
        "--load-row-chunk-size",
        type=int,
        default=None,
        help="Number of rows to read per CSV chunk. Setting this value will ignore \"--load-keep-old-files\"",
    )

    parser.add_argument(
        "--skip-exploration",
        action="store_true",
        default=False,
        help="Skip data exploration"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip data validation"
    )

    args = parser.parse_args()

    if args.load_row_chunk_size is None:
        LOAD_KEEP_OLD_FILES = args.load_keep_old_files
        LOAD_ROW_CHUNK_SIZE = 500_000
    else:
        LOAD_KEEP_OLD_FILES = False
        LOAD_ROW_CHUNK_SIZE = args.load_row_chunk_size

        if LOAD_ROW_CHUNK_SIZE < 1:
            raise ValueError(f"{LOAD_ROW_CHUNK_SIZE} is not a valid value for \"--row-chunk-size\"")

    SKIP_VALIDATION = args.skip_validation
    SKIP_EXPLORATION = args.skip_exploration
    main()
