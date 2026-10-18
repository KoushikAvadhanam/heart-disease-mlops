import numpy as np
import pandas as pd

from helpers import make_df
from preprocess import FEATURES, build_preprocessor, load_data


def test_load_data_returns_features_and_target(tmp_path):
    path = tmp_path / "heart.csv"
    make_df().to_csv(path, index=False)
    X, y = load_data(str(path))
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_imputes_missing_values():
    df = make_df()
    assert df[FEATURES].isna().sum().sum() > 0
    out = build_preprocessor().fit_transform(df[FEATURES])
    assert not np.isnan(np.asarray(out)).any()


def test_preprocessor_scales_numeric_columns():
    df = make_df()
    prep = build_preprocessor().fit(df[FEATURES])
    out = prep.transform(df[FEATURES])
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    # first five columns are the scaled numeric features
    assert np.allclose(out[:, :5].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out[:, :5].std(axis=0), 1, atol=1e-6)


def test_preprocessor_ignores_unseen_category():
    df = make_df()
    prep = build_preprocessor().fit(df[FEATURES])
    row = df[FEATURES].iloc[[0]].copy()
    row["cp"] = 99
    out = prep.transform(row)
    assert out.shape[0] == 1


def test_target_is_binary_after_download_logic():
    raw = pd.Series([0, 1, 2, 3, 4])
    assert list((raw > 0).astype(int)) == [0, 1, 1, 1, 1]
