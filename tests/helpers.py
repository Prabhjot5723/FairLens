"""Test-only helper: a fake table with the same columns as EMSCAD's fake_job_postings.csv."""
import numpy as np
import pandas as pd


def make_mock_emscad(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < 0.08).astype(int)
    scam = ["earn money fast no experience wire transfer", "urgent hiring work from home send fee"]
    ok = ["software engineer team collaborate product delivery quality testing", "marketing analyst campaign reporting stakeholders"]
    desc = [(" ".join(rng.choice(scam, 3)) if f else " ".join(rng.choice(ok, 3))) + " " * 40 for f in fraud]
    countries = rng.choice(["US", "GB", "IN", "NG", "CA", "AU", "DE", "PK"], size=n, p=[.4, .15, .15, .1, .08, .05, .04, .03])
    return pd.DataFrame({
        "job_id": range(n),
        "title": rng.choice(["Engineer", "Analyst", "Manager"], n),
        "location": [f"{c}, XX, City" for c in countries],
        "department": None,
        "salary_range": np.where(rng.random(n) < 0.2, "1000-2000", None),
        "company_profile": ["We are a company that builds things and delivers value to customers" for _ in range(n)],
        "description": desc,
        "requirements": ["Degree preferred and good communication skills required for this role" for _ in range(n)],
        "benefits": [None] * n,
        "telecommuting": rng.integers(0, 2, n),
        "has_company_logo": np.where(fraud == 1, (rng.random(n) < 0.3).astype(int), (rng.random(n) < 0.85).astype(int)),
        "has_questions": rng.integers(0, 2, n),
        "employment_type": rng.choice(["Full-time", "Part-time", "Contract", None], n),
        "required_experience": rng.choice(["Entry level", "Mid-Senior level", None], n),
        "required_education": rng.choice(["Bachelor's Degree", "High School or equivalent", None], n),
        "industry": None, "function": None,
        "fraudulent": fraud,
    })
