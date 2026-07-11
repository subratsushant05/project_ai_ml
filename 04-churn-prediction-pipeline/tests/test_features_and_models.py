"""Tests for feature engineering, preprocessing, and model quality."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from churn_pipeline.features import (
    TENURE_BUCKET_LABELS,
    build_preprocessor,
    engineer_features,
    get_feature_names,
)
from churn_pipeline.persistence import load_model, save_model


def test_engineer_features_adds_columns(small_df) -> None:
    """Engineered columns exist and are sane."""
    out = engineer_features(small_df.drop(columns=["churned", "customer_id"]))
    assert set(out["tenure_bucket"].dropna().unique()) <= set(TENURE_BUCKET_LABELS)
    assert (out["support_calls_per_year"].dropna() >= 0).all()
    positive = out["charges_per_tenure"].dropna()
    assert (positive > 0).all()


def test_preprocessor_output_shape_and_no_nans(split) -> None:
    """Preprocessor yields a dense numeric matrix without NaNs."""
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X_train)
    assert Xt.shape[0] == len(X_train)
    assert Xt.shape[1] == len(get_feature_names(preprocessor))
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()
    # Same width at inference time.
    assert preprocessor.transform(X_test).shape[1] == Xt.shape[1]


def test_pipeline_predict_proba_shape(fitted_pipeline, split) -> None:
    """predict_proba returns [n, 2] probabilities in [0, 1]."""
    _, X_test, _, _ = split
    proba = fitted_pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_model_beats_dummy_baseline(fitted_pipeline, split) -> None:
    """Trained model must clearly beat a stratified dummy on holdout AUC."""
    X_train, X_test, y_train, y_test = split
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    model_auc = roc_auc_score(y_test, fitted_pipeline.predict_proba(X_test)[:, 1])
    assert model_auc > 0.7
    assert model_auc > dummy_auc + 0.15


def test_save_and_load_roundtrip(fitted_pipeline, split, tmp_path) -> None:
    """Persisted model reproduces predictions and metadata."""
    _, X_test, _, _ = split
    save_model(
        fitted_pipeline,
        tmp_path,
        metrics={"roc_auc": 0.9},
        threshold=0.31,
        extra={"note": "test"},
    )
    bundle = load_model(tmp_path)
    np.testing.assert_allclose(
        bundle.pipeline.predict_proba(X_test)[:, 1],
        fitted_pipeline.predict_proba(X_test)[:, 1],
    )
    assert bundle.metadata["decision_threshold"] == 0.31
    assert bundle.metadata["metrics"]["roc_auc"] == 0.9
    assert bundle.metadata["model_class"] == "LGBMClassifier"
