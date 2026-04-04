# Language Identification

Research project for short-text language identification experiments on 5 languages:
EN, RO, FR, DE, ES.

## 1. Initialize the environment with uv

From the project root, run:

```bash
uv sync
```

This creates `.venv` and installs the core dependencies.

If you also want notebooks and linting:

```bash
uv sync --group dev
```

If you later add the transformer experiment:

```bash
uv sync --group dev --group transformers
```

## 2. Run Python inside the project

```bash
uv run python --version
uv run language-identification
```

## 3. Recommended project layout

```text
data/
  raw/          # downloaded datasets
  processed/    # train/val/test csv or parquet
notebooks/      # quick exploration
results/        # metrics, plots, confusion matrices
src/language_identification/
  data/         # data loading and preprocessing
  models/       # sklearn / fastText / transformer code
  evaluation/   # metrics and plots
  scripts/      # entry-point scripts
```

## 4. Current experiment setup

- Dataset: `agentlans/high-quality-multilingual-sentences`
- Languages: `EN`, `RO`, `FR`, `DE`, `ES`
- Examples per language: `2000`
- Split: `80/10/10`
- Length buckets: `3`, `5`, `10`
- Noise conditions: `clean`, `no_diacritics`, `typo`
- Baselines:
  - `tfidf_lr`
  - `char_svm`
  - `fasttext`

## 5. Suggested experiment flow

1. Download one multilingual public dataset into `data/raw/`.
2. Keep only EN, RO, FR, DE, ES.
3. Create train/val/test with stratification by language.
4. Generate controlled short texts with 3, 5, and 10 words.
5. Create noise variants:
   - `clean`
   - `no_diacritics`
   - `typo`
6. Train baseline models on clean train data.
7. Evaluate on all 9 test conditions.
8. Save one `results/results.csv` with columns like:
   - `model`
   - `language`
   - `len_bucket`
   - `noise_type`
   - `accuracy`
   - `macro_f1`
   - `runtime_sec`

## 6. Main commands

```bash
uv run python -m language_identification.scripts.prepare_dataset
uv run python -m language_identification.scripts.train_baselines
uv run python -m language_identification.scripts.evaluate
uv run python -m language_identification.scripts.evaluate_per_language
uv run python -m language_identification.scripts.error_analysis
uv run python -m language_identification.scripts.plot_results
uv run python -m language_identification.scripts.analyze_results
```

## 7. Practical command sequence

```bash
uv run python -m language_identification.scripts.prepare_dataset \
  --source hf \
  --hf-dataset agentlans/high-quality-multilingual-sentences \
  --hf-configs en,ro,fr,de,es \
  --hf-split train \
  --text-column text \
  --label-column label \
  --max-examples-per-language 2000 \
  --output-dir data/processed/hq_multilingual

uv run python -m language_identification.scripts.train_baselines
uv run python -m language_identification.scripts.evaluate
uv run python -m language_identification.scripts.evaluate_per_language
uv run python -m language_identification.scripts.error_analysis
uv run python -m language_identification.scripts.plot_results
uv run python -m language_identification.scripts.analyze_results
```

## 8. Main outputs

- Processed dataset:
  - `data/processed/hq_multilingual/processed_dataset.parquet`
- Global metrics:
  - `results/results.csv`
- Per-language metrics:
  - `results/per_language_results.csv`
- Plots:
  - `results/plots/`
- Error analysis for the hardest scenario:
  - `results/error_analysis/`
- Final written summary:
  - `results/FINAL_ANALYSIS.md`
- Paper:
  - `paper/main.tex`
  - `paper/main.pdf`
