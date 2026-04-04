from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from unidecode import unidecode

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
TARGET_LANGUAGES = ("en", "ro", "fr", "de", "es")
LENGTH_BUCKETS = (3, 5, 10)
NOISE_TYPES = ("clean", "no_diacritics", "typo")


@dataclass(frozen=True)
class PrepareConfig:
    source: str
    output_dir: Path
    max_examples_per_language: int
    random_state: int
    train_size: float
    val_size: float
    test_size: float
    text_column: str = "text"
    label_column: str = "label"
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_configs: tuple[str, ...] | None = None
    hf_split: str | None = None
    local_path: Path | None = None


def load_source_dataframe(config: PrepareConfig) -> pd.DataFrame:
    if config.source == "hf":
        if not config.hf_dataset or not config.hf_split:
            raise ValueError("HF source requires `hf_dataset` and `hf_split`.")
        if config.hf_configs:
            parts: list[pd.DataFrame] = []
            for hf_config in config.hf_configs:
                dataset = load_dataset(config.hf_dataset, hf_config, split=config.hf_split)
                partial = dataset.to_pandas()
                if config.label_column not in partial.columns:
                    partial[config.label_column] = hf_config
                parts.append(partial)
            df = pd.concat(parts, ignore_index=True)
        else:
            dataset = load_dataset(config.hf_dataset, config.hf_config, split=config.hf_split)
            df = dataset.to_pandas()
            if config.label_column not in df.columns:
                if config.hf_config is None:
                    raise ValueError(
                        "HF dataset does not contain a label column. "
                        "Provide `hf_config` or `hf_configs` so labels can be inferred."
                    )
                df[config.label_column] = config.hf_config
    elif config.source == "csv":
        if not config.local_path:
            raise ValueError("CSV source requires `local_path`.")
        df = pd.read_csv(config.local_path)
    elif config.source == "parquet":
        if not config.local_path:
            raise ValueError("Parquet source requires `local_path`.")
        df = pd.read_parquet(config.local_path)
    else:
        raise ValueError(f"Unsupported source: {config.source}")

    if config.text_column not in df.columns or config.label_column not in df.columns:
        raise ValueError(
            "Input data must contain the configured text/label columns. "
            f"Missing one of: {config.text_column}, {config.label_column}"
        )

    normalized = df[[config.text_column, config.label_column]].rename(
        columns={config.text_column: "text", config.label_column: "label"}
    )
    normalized["text"] = normalized["text"].astype(str)
    normalized["label"] = normalized["label"].astype(str).str.lower()
    return normalized


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def clean_and_filter_dataframe(
    df: pd.DataFrame,
    target_languages: tuple[str, ...],
    min_words: int,
    max_examples_per_language: int,
    random_state: int,
) -> pd.DataFrame:
    working = df.copy()
    working["text"] = working["text"].map(normalize_text)
    working = working[working["label"].isin(target_languages)]
    working = working[working["text"] != ""]
    working = working.dropna(subset=["text", "label"])
    working = working.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    working["word_count"] = working["text"].map(word_count)
    working = working[working["word_count"] >= min_words].copy()

    parts: list[pd.DataFrame] = []
    rng = random.Random(random_state)
    for label in target_languages:
        lang_df = working[working["label"] == label].copy()
        if lang_df.empty:
            continue
        n = min(max_examples_per_language, len(lang_df))
        sample_seed = rng.randint(0, 10_000_000)
        parts.append(lang_df.sample(n=n, random_state=sample_seed))

    if not parts:
        raise ValueError("No rows available after filtering.")

    filtered = pd.concat(parts, ignore_index=True)
    filtered["source_id"] = [
        f"{label}_{idx}" for idx, label in enumerate(filtered["label"].tolist())
    ]
    return filtered[["source_id", "text", "label", "word_count"]].reset_index(drop=True)


def create_splits(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df["label"],
        random_state=random_state,
    )

    val_ratio_within_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_within_temp),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def take_first_n_words(text: str, n_words: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:n_words])


def remove_diacritics(text: str) -> str:
    return unidecode(text)


def apply_single_typo(text: str, seed_text: str) -> str:
    alpha_positions = [idx for idx, char in enumerate(text) if char.isalpha()]
    if not alpha_positions:
        return text

    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    position = alpha_positions[rng.randrange(len(alpha_positions))]
    original_char = text[position]

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    replacement = rng.choice(alphabet)
    if original_char.lower() == replacement:
        replacement = alphabet[(alphabet.index(replacement) + 1) % len(alphabet)]

    if original_char.isupper():
        replacement = replacement.upper()

    return text[:position] + replacement + text[position + 1 :]


def apply_noise(text: str, noise_type: str, source_id: str, len_bucket: int) -> str:
    if noise_type == "clean":
        return text
    if noise_type == "no_diacritics":
        return remove_diacritics(text)
    if noise_type == "typo":
        return apply_single_typo(text, seed_text=f"{source_id}:{len_bucket}")
    raise ValueError(f"Unsupported noise type: {noise_type}")


def expand_conditions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        for len_bucket in LENGTH_BUCKETS:
            short_text = take_first_n_words(row.text, len_bucket)
            for noise_type in NOISE_TYPES:
                rows.append(
                    {
                        "source_id": row.source_id,
                        "split": row.split,
                        "label": row.label,
                        "len_bucket": len_bucket,
                        "noise_type": noise_type,
                        "text": apply_noise(short_text, noise_type, row.source_id, len_bucket),
                    }
                )
    return pd.DataFrame(rows)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "label", "len_bucket", "noise_type"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["split", "label", "len_bucket", "noise_type"])
    )


def write_outputs(base_df: pd.DataFrame, expanded_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_df.to_parquet(output_dir / "base_dataset.parquet", index=False)
    expanded_df.to_parquet(output_dir / "processed_dataset.parquet", index=False)
    summarize_dataframe(expanded_df).to_csv(output_dir / "dataset_summary.csv", index=False)
