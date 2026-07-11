"""Feature engineering and preprocessing for the churn pipeline.

Combines hand-engineered features (tenure buckets, spend ratios) with a
standard sklearn ``ColumnTransformer`` for imputation, scaling, and one-hot
encoding. Everything lives inside a single sklearn ``Pipeline`` so the same
transformations are applied at train and inference time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_pipeline.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

TENURE_BUCKET_EDGES = [-np.inf, 6, 12, 24, 48, np.inf]
TENURE_BUCKET_LABELS = ["0-6m", "6-12m", "12-24m", "24-48m", "48m+"]

ENGINEERED_NUMERIC = ["charges_per_tenure", "support_calls_per_year"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a raw feature frame.

    Adds:
        * ``tenure_bucket``: categorical tenure bands (0-6m ... 48m+).
        * ``charges_per_tenure``: total spend normalised by tenure, a proxy
          for the customer's realised monthly spend.
        * ``support_calls_per_year``: support-call intensity adjusted for
          how long the customer has been around.

    Args:
        df: Frame containing the raw feature columns.

    Returns:
        Copy of ``df`` with three additional columns.
    """
    out = df.copy()
    tenure = out["tenure_months"].astype(float)
    out["tenure_bucket"] = pd.cut(
        tenure, bins=TENURE_BUCKET_EDGES, labels=TENURE_BUCKET_LABELS
    ).astype(object)
    out["charges_per_tenure"] = out["total_charges"] / tenure.clip(lower=1.0)
    out["support_calls_per_year"] = (
        12.0 * out["num_support_calls"] / tenure.clip(lower=1.0)
    ).clip(upper=24.0)
    return out


def build_preprocessor() -> Pipeline:
    """Build the full preprocessing pipeline.

    Structure::

        engineer (FunctionTransformer)
          -> ColumnTransformer
               numeric:     median impute -> standard scale
               categorical: most-frequent impute -> one-hot (ignore unknown)

    Returns:
        Unfitted sklearn ``Pipeline`` mapping a raw feature frame to a dense
        numeric matrix.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            ("num", numeric, ALL_NUMERIC),
            ("cat", categorical, ALL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor = Pipeline(
        [
            ("engineer", FunctionTransformer(engineer_features, validate=False)),
            ("transform", column_transformer),
        ]
    )
    # Pandas output keeps feature names attached end-to-end, so the model,
    # the SHAP explanations, and the serving path all agree on column names.
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return output feature names of a fitted preprocessor.

    Args:
        preprocessor: A fitted pipeline from :func:`build_preprocessor`.

    Returns:
        Names of the columns in the transformed matrix, in order.
    """
    return list(preprocessor.named_steps["transform"].get_feature_names_out())
