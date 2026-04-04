from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from language_identification.models.fasttext_baseline import train_fasttext_model
from language_identification.models.sklearn_baselines import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument(
        "--dataset-path",
        default="data/processed/hq_multilingual/processed_dataset.parquet",
    )
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--models", default="char_svm,tfidf_lr,fasttext")
    parser.add_argument("--train-noise", default="clean")
    parser.add_argument("--train-len-buckets", default="3,5,10")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    len_buckets = [int(part.strip()) for part in args.train_len_buckets.split(",") if part.strip()]

    manifest: list[dict[str, object]] = []

    for model_name in model_names:
        for len_bucket in len_buckets:
            train_df = df[
                (df["split"] == "train")
                & (df["noise_type"] == args.train_noise)
                & (df["len_bucket"] == len_bucket)
            ].copy()

            if train_df.empty:
                raise ValueError(
                    f"No training data found for model={model_name}, len_bucket={len_bucket}"
                )

            models_dir.mkdir(parents=True, exist_ok=True)
            if model_name == "fasttext":
                model_backend = "fasttext"
                model_path = models_dir / f"{model_name}_len{len_bucket}.bin"
                start = time.perf_counter()
                training_file_size_mb = train_fasttext_model(train_df, model_path)
                train_runtime_sec = time.perf_counter() - start
            else:
                model_backend = "sklearn"
                model = build_model(model_name)
                start = time.perf_counter()
                model.fit(train_df["text"].tolist(), train_df["label"].tolist())
                train_runtime_sec = time.perf_counter() - start
                model_path = models_dir / f"{model_name}_len{len_bucket}.joblib"
                joblib.dump(model, model_path)
                training_file_size_mb = None

            manifest.append(
                {
                    "model": model_name,
                    "model_backend": model_backend,
                    "train_len_bucket": len_bucket,
                    "train_noise_type": args.train_noise,
                    "n_train_examples": len(train_df),
                    "train_runtime_sec": train_runtime_sec,
                    "training_file_size_mb": training_file_size_mb,
                    "model_path": str(model_path),
                }
            )

            print(
                f"Trained {model_name} on len_bucket={len_bucket} "
                f"with {len(train_df)} examples in {train_runtime_sec:.3f}s"
            )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(output_dir / "model_manifest.csv", index=False)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "models": model_names,
                "train_noise": args.train_noise,
                "train_len_buckets": len_buckets,
            },
            handle,
            indent=2,
        )

    print(f"Saved manifest to: {output_dir / 'model_manifest.csv'}")


if __name__ == "__main__":
    main()
