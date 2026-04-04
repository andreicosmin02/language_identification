from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from language_identification.evaluation.baselines import predict_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate error analysis artifacts.")
    parser.add_argument(
        "--dataset-path",
        default="data/processed/hq_multilingual/processed_dataset.parquet",
    )
    parser.add_argument("--manifest-path", default="results/baselines/model_manifest.csv")
    parser.add_argument("--model", default="fasttext")
    parser.add_argument("--len-bucket", type=int, default=3)
    parser.add_argument("--noise-type", default="typo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="results/error_analysis")
    return parser


def plot_confusion_matrix(cm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm_df.values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            ax.text(j, i, f"{cm_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.dataset_path)
    manifest_df = pd.read_csv(args.manifest_path)

    selected = manifest_df[
        (manifest_df["model"] == args.model)
        & (manifest_df["train_len_bucket"] == args.len_bucket)
    ]
    if selected.empty:
        raise ValueError("Requested model/length combination not found in manifest.")

    item = selected.iloc[0]
    eval_df = df[
        (df["split"] == args.split)
        & (df["len_bucket"] == args.len_bucket)
        & (df["noise_type"] == args.noise_type)
    ].copy()
    if eval_df.empty:
        raise ValueError("No evaluation rows found for the requested condition.")

    predictions, _runtime_sec = predict_labels(
        model_path=Path(item["model_path"]),
        model_backend=item["model_backend"],
        texts=eval_df["text"].tolist(),
    )
    eval_df["predicted_label"] = predictions
    eval_df["is_correct"] = eval_df["label"] == eval_df["predicted_label"]

    labels = sorted(eval_df["label"].unique())
    cm = confusion_matrix(
        eval_df["label"],
        eval_df["predicted_label"],
        labels=labels,
        normalize="true",
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.model}_len{args.len_bucket}_{args.noise_type}_{args.split}"
    predictions_path = output_dir / f"{stem}_predictions.csv"
    cm_csv_path = output_dir / f"{stem}_confusion_matrix.csv"
    cm_png_path = output_dir / f"{stem}_confusion_matrix.png"
    confusions_path = output_dir / f"{stem}_top_confusions.csv"

    eval_df.to_csv(predictions_path, index=False)
    cm_df.to_csv(cm_csv_path)

    off_diag = cm_df.copy()
    for label in labels:
        off_diag.loc[label, label] = 0.0
    top_confusions = (
        off_diag.stack()
        .reset_index()
        .rename(columns={"level_0": "true_label", "level_1": "predicted_label", 0: "rate"})
        .sort_values("rate", ascending=False)
    )
    top_confusions = top_confusions[top_confusions["rate"] > 0].reset_index(drop=True)
    top_confusions.to_csv(confusions_path, index=False)

    plot_confusion_matrix(
        cm_df,
        cm_png_path,
        title=(
            f"Confusion Matrix: {args.model}, {args.len_bucket} words, "
            f"{args.noise_type}, {args.split}"
        ),
    )

    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved confusion matrix CSV to: {cm_csv_path}")
    print(f"Saved confusion matrix plot to: {cm_png_path}")
    print(f"Saved top confusions to: {confusions_path}")


if __name__ == "__main__":
    main()
