from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_model(model_name: str) -> Pipeline:
    if model_name == "tfidf_lr":
        return Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        lowercase=True,
                        min_df=2,
                        max_features=100_000,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "char_svm":
        return Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        lowercase=True,
                        min_df=2,
                        max_features=200_000,
                    ),
                ),
                ("classifier", LinearSVC()),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")
