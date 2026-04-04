from __future__ import annotations

import tempfile
from pathlib import Path

import fasttext
import pandas as pd


def _to_fasttext_label(label: str) -> str:
    return f"__label__{label}"


def _from_fasttext_label(label: str) -> str:
    return label.removeprefix("__label__")


def write_fasttext_training_file(train_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in train_df.itertuples(index=False):
            handle.write(f"{_to_fasttext_label(row.label)} {row.text}\n")


def train_fasttext_model(train_df: pd.DataFrame, model_path: Path) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "fasttext_train.txt"
        write_fasttext_training_file(train_df, train_file)

        model = fasttext.train_supervised(
            input=str(train_file),
            lr=0.5,
            epoch=25,
            wordNgrams=2,
            minn=2,
            maxn=5,
            dim=100,
            loss="softmax",
            thread=1,
        )
        model.save_model(str(model_path))
        return train_file.stat().st_size / (1024 * 1024)


def predict_fasttext(model_path: Path, texts: list[str]) -> list[str]:
    model = fasttext.load_model(str(model_path))
    labels, _scores = model.predict(texts, k=1)
    return [_from_fasttext_label(item[0]) for item in labels]


def predict_fasttext_with_loaded_model(model: fasttext.FastText._FastText, texts: list[str]) -> list[str]:
    labels, _scores = model.predict(texts, k=1)
    return [_from_fasttext_label(item[0]) for item in labels]
