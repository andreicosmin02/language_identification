from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
import fasttext
from sklearn.metrics import accuracy_score, f1_score

from language_identification.models.fasttext_baseline import predict_fasttext_with_loaded_model


def predict_labels(
    model_path: Path,
    model_backend: str,
    texts: list[str],
) -> tuple[list[str], float]:
    if model_backend == "sklearn":
        model = joblib.load(model_path)
        start = time.perf_counter()
        predictions = model.predict(texts)
    elif model_backend == "fasttext":
        model = fasttext.load_model(str(model_path))
        start = time.perf_counter()
        predictions = predict_fasttext_with_loaded_model(model, texts)
    else:
        raise ValueError(f"Unsupported backend: {model_backend}")

    runtime_sec = time.perf_counter() - start
    return list(predictions), runtime_sec


def evaluate_model(
    model_path: Path,
    model_name: str,
    model_backend: str,
    train_len_bucket: int,
    split: str,
    noise_type: str,
    eval_df: pd.DataFrame,
) -> dict[str, object]:
    texts = eval_df["text"].tolist()
    labels = eval_df["label"].tolist()

    predictions, runtime_sec = predict_labels(model_path, model_backend, texts)

    return {
        "model": model_name,
        "model_backend": model_backend,
        "train_len_bucket": train_len_bucket,
        "eval_len_bucket": train_len_bucket,
        "split": split,
        "noise_type": noise_type,
        "n_examples": len(eval_df),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "runtime_sec": runtime_sec,
        "examples_per_sec": len(eval_df) / runtime_sec if runtime_sec > 0 else None,
    }
