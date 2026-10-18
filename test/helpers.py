"""Synthetic data helper shared by tests (no network needed)."""
import numpy as np
import pandas as pd


def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(70, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n),
        "ca": rng.choice([0, 1, 2, 3, np.nan], n),
        "thal": rng.choice([3, 6, 7, np.nan], n),
    })
    df["target"] = ((df["age"] / 80 + df["exang"] + df["oldpeak"] / 4
                     + rng.normal(0, 0.5, n)) > 1.3).astype(int)
    df.loc[0, "target"] = 0
    df.loc[1, "target"] = 1
    return df
