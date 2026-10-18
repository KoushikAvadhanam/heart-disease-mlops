import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from helpers import make_df
from preprocess import FEATURES, build_preprocessor


def _fit(clf):
    df = make_df()
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    pipe.fit(df[FEATURES], df["target"])
    return pipe, df


def test_logistic_regression_pipeline_probabilities():
    pipe, df = _fit(LogisticRegression(max_iter=1000))
    proba = pipe.predict_proba(df[FEATURES])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_random_forest_beats_chance_on_train():
    pipe, df = _fit(RandomForestClassifier(n_estimators=30, random_state=0))
    assert (pipe.predict(df[FEATURES]) == df["target"]).mean() > 0.7


def test_saved_pipeline_reproduces_predictions(tmp_path):
    pipe, df = _fit(LogisticRegression(max_iter=1000))
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    assert np.allclose(pipe.predict_proba(df[FEATURES]),
                       loaded.predict_proba(df[FEATURES]))
