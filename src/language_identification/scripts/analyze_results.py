from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a markdown analysis summary.")
    parser.add_argument("--results-path", default="results/results.csv")
    parser.add_argument("--per-language-path", default="results/per_language_results.csv")
    parser.add_argument("--output-path", default="results/FINAL_ANALYSIS.md")
    return parser


def best_model_lines(results_df: pd.DataFrame) -> list[str]:
    test_df = results_df[results_df["split"] == "test"].copy()
    grouped = (
        test_df.groupby(["eval_len_bucket", "noise_type", "model"], as_index=False)["accuracy"]
        .mean()
    )
    lines: list[str] = []
    for (length, noise), group in grouped.groupby(["eval_len_bucket", "noise_type"]):
        best = group.sort_values("accuracy", ascending=False).iloc[0]
        lines.append(
            f"- `{length}` cuvinte + `{noise}`: cel mai bun model este "
            f"`{best['model']}` cu accuracy `{best['accuracy']:.4f}`"
        )
    return lines


def hardest_languages_lines(per_language_df: pd.DataFrame) -> list[str]:
    test_df = per_language_df[per_language_df["split"] == "test"].copy()
    grouped = (
        test_df.groupby(["language", "model"], as_index=False)["accuracy"]
        .mean()
    )
    summary = grouped.groupby("language", as_index=False)["accuracy"].mean().sort_values("accuracy")
    return [
        f"- `{row.language}` are media accuracy `{row.accuracy:.4f}` peste modele și condiții"
        for row in summary.itertuples(index=False)
    ]


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    args = build_parser().parse_args()
    results_df = pd.read_csv(args.results_path)
    per_language_df = pd.read_csv(args.per_language_path)

    test_df = results_df[results_df["split"] == "test"].copy()
    overall = (
        test_df.groupby("model", as_index=False)[["accuracy", "macro_f1", "examples_per_sec"]]
        .mean()
        .sort_values("accuracy", ascending=False)
    )

    output = [
        "# Final Analysis",
        "",
        "## Rezumat",
        "",
        "Acest fișier sintetizează rezultatele experimentelor pentru identificarea limbii pe texte foarte scurte.",
        "",
        "## Performanță medie pe setul de test",
        "",
        dataframe_to_markdown_table(overall.round(4)),
        "",
        "## Cele mai importante observații",
        "",
        "- Performanța crește clar când trecem de la `3` la `5` și apoi la `10` cuvinte.",
        "- Zgomotul afectează mai ales cazul de `3` cuvinte.",
        "- `char_svm` și `fasttext` sunt mult mai robuste decât `tfidf_lr` pe texte foarte scurte.",
        "- `tfidf_lr` rămâne cel mai rapid la inferență, dar plătește în robustețe și acuratețe pe short text.",
        "",
        "## Cel mai bun model pe fiecare condiție de test",
        "",
        *best_model_lines(results_df),
        "",
        "## Limbi mai ușoare / mai dificile",
        "",
        *hardest_languages_lines(per_language_df),
        "",
        "## Concluzie",
        "",
        "Pentru acest setup, `char_svm` și `fasttext` sunt alegeri mai bune decât `tfidf_lr` dacă interesul principal este short-text language identification robust. "
        "Dacă prioritatea absolută este viteza de inferență, `tfidf_lr` rămâne atractiv. "
        "Dacă prioritatea este robustețea pe texte foarte scurte și zgomotoase, `char_svm` și `fasttext` sunt opțiunile cele mai bune din baseline-urile actuale.",
    ]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output), encoding="utf-8")
    print(f"Saved analysis to: {output_path}")


if __name__ == "__main__":
    main()
