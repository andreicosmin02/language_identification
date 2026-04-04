from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_length(results_df: pd.DataFrame, output_dir: Path) -> None:
    clean_df = (
        results_df[results_df["noise_type"] == "clean"]
        .groupby(["model", "eval_len_bucket"], as_index=False)["accuracy"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in clean_df.groupby("model"):
        group = group.sort_values("eval_len_bucket")
        ax.plot(group["eval_len_bucket"], group["accuracy"], marker="o", label=model)

    ax.set_title("Accuracy vs Text Length (Clean)")
    ax.set_xlabel("Text length bucket")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(sorted(clean_df["eval_len_bucket"].unique()))
    ax.set_ylim(0.7, 1.01)
    ax.grid(alpha=0.3)
    ax.legend()
    _save_plot(fig, output_dir / "accuracy_by_length_clean.png")


def plot_accuracy_by_noise(results_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        results_df.groupby(["model", "noise_type"], as_index=False)["accuracy"]
        .mean()
        .pivot(index="noise_type", columns="model", values="accuracy")
        .reindex(["clean", "no_diacritics", "typo"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Average Accuracy by Noise Type")
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.01)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    _save_plot(fig, output_dir / "accuracy_by_noise.png")


def plot_accuracy_heatmap_table(results_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        results_df.groupby(["model", "eval_len_bucket", "noise_type"], as_index=False)["accuracy"]
        .mean()
        .round(4)
        .sort_values(["model", "eval_len_bucket", "noise_type"])
    )
    grouped.to_csv(output_dir / "accuracy_summary.csv", index=False)


def plot_speed_by_length(results_df: pd.DataFrame, output_dir: Path) -> None:
    speed_df = (
        results_df.groupby(["model", "eval_len_bucket"], as_index=False)["examples_per_sec"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in speed_df.groupby("model"):
        group = group.sort_values("eval_len_bucket")
        ax.plot(group["eval_len_bucket"], group["examples_per_sec"], marker="o", label=model)

    ax.set_title("Inference Speed vs Text Length")
    ax.set_xlabel("Text length bucket")
    ax.set_ylabel("Examples / second")
    ax.set_xticks(sorted(speed_df["eval_len_bucket"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    _save_plot(fig, output_dir / "speed_by_length.png")
