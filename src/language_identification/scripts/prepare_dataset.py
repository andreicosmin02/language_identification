from __future__ import annotations

import argparse
from pathlib import Path

from language_identification.data.prepare import (
    LENGTH_BUCKETS,
    NOISE_TYPES,
    TARGET_LANGUAGES,
    PrepareConfig,
    clean_and_filter_dataframe,
    create_splits,
    expand_conditions,
    load_source_dataframe,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare short-text language-ID datasets.")
    parser.add_argument("--source", choices=["hf", "csv", "parquet"], required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-examples-per-language", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-configs")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--local-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = PrepareConfig(
        source=args.source,
        output_dir=Path(args.output_dir),
        max_examples_per_language=args.max_examples_per_language,
        random_state=args.random_state,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        text_column=args.text_column,
        label_column=args.label_column,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_configs=tuple(part.strip() for part in args.hf_configs.split(","))
        if args.hf_configs
        else None,
        hf_split=args.hf_split,
        local_path=Path(args.local_path) if args.local_path else None,
    )

    print("Preparing dataset")
    print(f"Target languages: {', '.join(TARGET_LANGUAGES)}")
    print(f"Length buckets: {LENGTH_BUCKETS}")
    print(f"Noise types: {NOISE_TYPES}")

    source_df = load_source_dataframe(config)
    filtered_df = clean_and_filter_dataframe(
        source_df,
        target_languages=TARGET_LANGUAGES,
        min_words=max(LENGTH_BUCKETS),
        max_examples_per_language=config.max_examples_per_language,
        random_state=config.random_state,
    )
    split_df = create_splits(
        filtered_df,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    expanded_df = expand_conditions(split_df)
    write_outputs(split_df, expanded_df, config.output_dir)

    print(f"Saved base dataset to: {config.output_dir / 'base_dataset.parquet'}")
    print(f"Saved processed dataset to: {config.output_dir / 'processed_dataset.parquet'}")
    print(f"Saved dataset summary to: {config.output_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
