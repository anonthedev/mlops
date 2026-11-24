"""Preprocessing and scikit-learn models for MP124 grain predictions."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mp124.checks import map_to_levels


def clean_features(frame, *, shop_area_levels, biometric_failures_sentinel):
    """Apply the feature corrections during both training and prediction."""
    features = frame.copy()
    features["shop_area"] = features["shop_area"].map(
        lambda value: map_to_levels(value, shop_area_levels)
    )
    # Use one missing-value representation for pandas and scikit-learn.
    features["shop_area"] = features["shop_area"].astype(object)
    features.loc[features["shop_area"].isna(), "shop_area"] = np.nan
    features["biometric_failures"] = features["biometric_failures"].replace(
        biometric_failures_sentinel, np.nan
    )
    return features


def build_model(columns, training, candidate):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, columns["numeric"]),
        ("categorical", categorical, columns["categorical"]),
    ])

    settings = candidate["settings"]
    if candidate["estimator"] == "ridge":
        regressor = Ridge(**settings)
    elif candidate["estimator"] == "histogram_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            **settings,
            early_stopping=False,
            loss="absolute_error",
            random_state=training["random_state"],
        )
    else:
        raise ValueError(f"Unknown estimator: {candidate['estimator']}")

    return Pipeline([
        ("clean", FunctionTransformer(clean_features, kw_args={
            "shop_area_levels": training["shop_area_levels"],
            "biometric_failures_sentinel": training["biometric_failures_sentinel"],
        })),
        ("preprocess", preprocess),
        ("predict", regressor),
    ])
