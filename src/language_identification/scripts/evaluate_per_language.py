from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from language_identification.evaluation.baselines import predict_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained models per language.")
    parser.add_argument(
        "--dataset-path",
        default="data/processed/hq_multilingual/processed_dataset.parquet",
    )
    parser.add_argument("--manifest-path", default="results/baselines/model_manifest.csv")
    parser.add_argument("--output-path", default="results/per_language_results.csv")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--noise-types", default="clean,no_diacritics,typo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.dataset_path)
    manifest_df = pd.read_csv(args.manifest_path)
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    noise_types = [part.strip() for part in args.noise_types.split(",") if part.strip()]

    rows: list[dict[str, object]] = []
    for item in manifest_df.itertuples(index=False):
        for split in splits:
            for noise_type in noise_types:
                eval_df = df[
                    (df["split"] == split)
                    & (df["len_bucket"] == item.train_len_bucket)
                    & (df["noise_type"] == noise_type)
                ].copy()
                if eval_df.empty:
                    continue

                predictions, runtime_sec = predict_labels(
                    model_path=Path(item.model_path),
                    model_backend=item.model_backend,
                    texts=eval_df["text"].tolist(),
                )
                eval_df["predicted_label"] = predictions

                for language, group in eval_df.groupby("label"):
                    rows.append(
                        {
                            "model": item.model,
                            "model_backend": item.model_backend,
                            "train_len_bucket": int(item.train_len_bucket),
                            "eval_len_bucket": int(item.train_len_bucket),
                            "split": split,
                            "noise_type": noise_type,
                            "language": language,
                            "n_examples": len(group),
                            "accuracy": (group["label"] == group["predicted_label"]).mean(),
                            "runtime_sec_total_batch": runtime_sec,
                        }
                    )

    results_df = pd.DataFrame(rows).sort_values(
        ["model", "split", "train_len_bucket", "noise_type", "language"]
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved per-language evaluation to: {output_path}")


if __name__ == "__main__":
    main()
