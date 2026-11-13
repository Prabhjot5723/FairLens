"""Dataset loading + helpers: synthetic demo data, the FraudShield (EMSCAD) preset, sensitive-column prep."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


# ---------------------------------------------------------------------------
# Sensitive attribute preparation
# ---------------------------------------------------------------------------
def prepare_sensitive(df: pd.DataFrame, cols: list[str], max_levels: int = 8, n_bins: int = 4) -> pd.DataFrame:
    """Return a DataFrame of string-valued group labels, one column per attribute.

    - numeric columns with many values are binned into quantile bins
    - categorical columns with many values keep the biggest levels + 'Other'
    - missing values become the group 'Missing'
    """
    out = {}
    for c in cols:
        s = df[c]
        if is_numeric_dtype(s) and not is_bool_dtype(s) and s.nunique() > max_levels:
            try:
                binned = pd.qcut(s, q=n_bins, duplicates="drop")
                out[c] = binned.astype(object).where(binned.notna(), "Missing").astype(str)
                continue
            except ValueError:
                pass
        s = s.astype(object).where(s.notna(), "Missing").astype(str)
        if s.nunique() > max_levels:
            keep = s.value_counts().index[: max_levels - 1]
            s = s.where(s.isin(keep), "Other")
        out[c] = s
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Synthetic demo: a deliberately biased hiring dataset
# ---------------------------------------------------------------------------
def make_synthetic_hiring(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Fake hiring data with *historical bias baked in* so the auditor has something to find.

    - women have the same skill distribution but were hired less often in the past
    - `career_gap_years` is a proxy: it correlates with gender, so removing the
      gender column from the model does NOT remove the bias
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age_group = rng.choice(["18-30", "31-45", "46+"], size=n, p=[0.40, 0.40, 0.20])
    education = rng.choice(["High school", "Bachelor", "Master", "PhD"], size=n, p=[0.20, 0.50, 0.22, 0.08])
    experience = np.clip(rng.normal(6, 4, n), 0, 30).round(1)
    skill_score = np.clip(rng.normal(65, 15, n), 0, 100).round(1)
    gap = np.where(gender == "Female", rng.poisson(1.6, n), rng.poisson(0.4, n))  # proxy for gender

    edu_pts = pd.Series(education).map({"High school": 0, "Bachelor": 0.5, "Master": 0.9, "PhD": 1.2}).to_numpy()
    logit = -5.2 + 0.045 * skill_score + 0.10 * experience + edu_pts - 0.18 * gap
    logit = logit - 0.9 * (gender == "Female") - 0.4 * (age_group == "46+")  # historical bias
    p = 1 / (1 + np.exp(-logit))
    hired = (rng.random(n) < p).astype(int)

    return pd.DataFrame(
        {
            "gender": gender,
            "age_group": age_group,
            "education": education,
            "experience_years": experience,
            "skill_score": skill_score,
            "career_gap_years": gap,
            "hired": hired,
        }
    )


# ---------------------------------------------------------------------------
# FraudShield preset (EMSCAD - Kaggle 'fake_job_postings.csv')
# ---------------------------------------------------------------------------
EMSCAD_TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
EMSCAD_REQUIRED = EMSCAD_TEXT_COLS + [
    "location", "salary_range", "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education", "fraudulent",
]


def load_emscad(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw EMSCAD table into a model-ready table with audit-friendly columns."""
    missing = [c for c in EMSCAD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"This does not look like fake_job_postings.csv - missing columns: {missing}")

    d = pd.DataFrame(index=df.index)
    d["text"] = df[EMSCAD_TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
    country = df["location"].fillna("Unknown").astype(str).str.split(",").str[0].str.strip().replace("", "Unknown")
    top = country.value_counts().index[:7]
    d["country"] = country.where(country.isin(top), "Other")
    d["employment_type"] = df["employment_type"].fillna("Unknown")
    d["required_experience"] = df["required_experience"].fillna("Unknown")
    d["required_education"] = df["required_education"].fillna("Unknown")
    d["has_company_logo"] = df["has_company_logo"].map({1: "Has logo", 0: "No logo"}).fillna("Unknown")
    d["has_questions"] = df["has_questions"].map({1: "Has screening questions", 0: "No screening questions"}).fillna("Unknown")
    d["telecommuting"] = df["telecommuting"].map({1: "Remote", 0: "On-site"}).fillna("Unknown")
    d["salary_listed"] = df["salary_range"].notna().map({True: "Salary listed", False: "No salary"})
    d["fraudulent"] = df["fraudulent"].astype(int)
    return d


FRAUDSHIELD_SENSITIVE_DEFAULT = ["country", "employment_type", "required_education", "has_company_logo"]
