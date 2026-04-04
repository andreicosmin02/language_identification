from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from language_identification.evaluation.plots import (
    plot_accuracy_by_length,
    plot_accuracy_by_noise,
    plot_accuracy_heatmap_table,
    plot_speed_by_length,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plots from results.csv.")
    parser.add_argument("--results-path", default="results/results.csv")
    parser.add_argument("--output-dir", default="results/plots")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_df = pd.read_csv(args.results_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_length(results_df, output_dir)
    plot_accuracy_by_noise(results_df, output_dir)
    plot_speed_by_length(results_df, output_dir)
    plot_accuracy_heatmap_table(results_df, output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
