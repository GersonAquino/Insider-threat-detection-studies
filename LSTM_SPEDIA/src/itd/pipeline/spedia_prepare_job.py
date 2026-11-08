"""Prepare SPEDIA window shards using the study's original notebook pipeline."""
from __future__ import annotations

import argparse

from .. import config
from ..adapters.spedia import load_spedia_events
from .window_candidates import build_spedia_windows_manifest
from .window_shards import write_window_shards_from_events_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SPEDIA temporal-window shards.")
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()
    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    if not config.SPEDIA_CSV_PATH.is_file():
        parser.error(f"Input CSV not found: {config.SPEDIA_CSV_PATH}; set SPEDIA_CSV_PATH.")

    events = load_spedia_events()
    if events.empty:
        raise ValueError("No valid SPEDIA events remain after input parsing.")

    if bool(config.SPEDIA_USE_DYNAMIC_WINDOW_HPO):
        manifest = build_spedia_windows_manifest(
            events=events,
            manifest_path=config.SPEDIA_WINDOWS_MANIFEST,
            base_dir=config.SPEDIA_WINDOWS_BASE_DIR,
            shard_size=args.shard_size,
            reuse_existing=True,
        )
        print("Windows manifest:", manifest["manifest_path"])
    else:
        out_dir = config.SPEDIA_WINDOWS_BASE_DIR / f"w={config.WINDOW_SIZE}_m={config.MAX_EVENTS_PER_WINDOW}"
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(
                f"Window destination is not empty: {out_dir}; choose a fresh SPEDIA_OUTPUT_ROOT."
            )
        write_window_shards_from_events_df(
            events=events,
            out_dir=out_dir,
            window_size=config.WINDOW_SIZE,
            max_events=config.MAX_EVENTS_PER_WINDOW,
            shard_size=args.shard_size,
        )
        print("Windows shards:", out_dir)


if __name__ == "__main__":
    main()
