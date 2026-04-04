from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export training rows to CSV.")
    parser.add_argument(
        "--dataset-path",
        default="data/processed/hq_multilingual/processed_dataset.parquet",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/hq_multilingual/train_clean_all_buckets.csv",
    )
    parser.add_argument("--noise-type", default="clean")
    parser.add_argument("--split", default="train")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.dataset_path)
    export_df = df[(df["split"] == args.split) & (df["noise_type"] == args.noise_type)].copy()
    export_df = export_df.sort_values(["len_bucket", "label", "source_id"]).reset_index(drop=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    print(f"Saved CSV to: {output_path}")
    print(f"Rows: {len(export_df)}")


if __name__ == "__main__":
    main()
