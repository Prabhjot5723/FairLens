"""Built-in model pipeline used when the user gives us a dataset (not predictions)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_CHOICES = ["Logistic Regression", "Random Forest"]


def detect_text_columns(X: pd.DataFrame, min_median_len: int = 60) -> list[str]:
    """Free-text columns = non-numeric columns whose typical value is long."""
    out = []
    for c in X.columns:
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]):
            continue
        lens = X[c].dropna().astype(str).str.len()
        if len(lens) and lens.median() >= min_median_len:
            out.append(c)
    return out


def build_pipeline(X: pd.DataFrame, model_name: str = "Logistic Regression", balance: bool = True, seed: int = 42) -> Pipeline:
    text_cols = detect_text_columns(X)
    num_cols = [c for c in X.columns if (is_numeric_dtype(X[c]) or is_bool_dtype(X[c])) and c not in text_cols]
    cat_cols = [c for c in X.columns if c not in text_cols and c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                cat_cols,
            )
        )
    for c in text_cols:
        transformers.append((f"txt_{c}", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2), c))

    cw = "balanced" if balance else None
    if model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=150, max_features="sqrt", class_weight=cw, n_jobs=-1, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed)

    return Pipeline([("prep", ColumnTransformer(transformers, sparse_threshold=0.3)), ("clf", clf)])


def fit_pipeline(pipe: Pipeline, X: pd.DataFrame, y, sample_weight=None) -> Pipeline:
    if sample_weight is None:
        pipe.fit(X, y)
    else:
        pipe.fit(X, y, clf__sample_weight=np.asarray(sample_weight))
    return pipe


def predict_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


def three_way_split(y, seed: int = 42):
    """60% train / 20% validation (fit mitigation) / 20% test (final audit)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=seed)
    except ValueError:  # a class is too small to stratify
        tr, te = train_test_split(idx, test_size=0.20, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.25, random_state=seed)
    return tr, va, te
