"""Pipeline, cleaning, Stage 4 artifact, and Kubernetes manifest tests (CSV-free)."""

import numpy as np
import pandas as pd
import yaml

from mp124.checks import check_duplicate_rows, check_sentinel_value, check_spelling_variants
from mp124.config import ROOT, load_params
from mp124.model import build_model, clean_features
from mp124.verify import verify_all


def test_clean_features_normalizes_shop_area_and_sentinel():
    df = pd.DataFrame({
        "cards_attached": [1200, 1500],
        "working_hours": [7.0, 8.0],
        "biometric_failures": [9999, 12],
        "temp_c": [30.0, 32.0],
        "stock_kg_morning": [1100.0, 1300.0],
        "days_to_month_end": [5, 10],
        "shop_area": ["tri.", " RURAL "],
        "block": ["Block North", "Block South"],
    })
    cleaned = clean_features(
        df,
        shop_area_levels=["rural", "tribal", "urban"],
        biometric_failures_sentinel=9999,
    )
    assert cleaned["shop_area"].tolist() == ["tribal", "rural"]
    assert np.isnan(cleaned["biometric_failures"].iloc[0])
    assert cleaned["biometric_failures"].iloc[1] == 12


def test_data_checks_detect_sentinel_duplicates_and_spellings():
    train = pd.DataFrame({
        "record_id": ["R1", "R1", "R2", "R3", "R4"],
        "biometric_failures": [9999, 9999, 10, 12, 15],
        "shop_area": ["rur", "rur", "rural", "URB.", "Tribal"],
    })
    later = pd.DataFrame({"shop_area": ["rural", "tribal", "urban"]})

    sentinel = check_sentinel_value(train, ["biometric_failures"], [-1, 9999])
    dups = check_duplicate_rows(train, "record_id")
    spell = check_spelling_variants(train, later, ["shop_area"])

    assert sentinel["present"] is True and sentinel["value"] == 9999 and sentinel["count"] == 2
    assert dups["present"] is True and dups["count"] == 1
    assert spell["present"] is True and spell["true_levels"] == 3


def test_build_model_fits_and_predicts_synthetic_data():
    params = load_params()
    columns, training = params["columns"], params["training"]
    feature_cols = columns["numeric"] + columns["categorical"]

    df = pd.DataFrame({
        "cards_attached": [1000, 1500, 2000, 2500, 1800, 2200],
        "working_hours": [6.0, 7.0, 8.0, 7.5, 6.5, 8.5],
        "biometric_failures": [9999, 10, 20, 5, 15, 0],
        "temp_c": [28.0, 30.0, 32.0, 35.0, 29.0, 31.0],
        "stock_kg_morning": [900.0, 1200.0, 1600.0, 2000.0, 1400.0, 1700.0],
        "days_to_month_end": [2, 5, 10, 15, 20, 25],
        "shop_area": ["rural", "tri.", "URBAN", "rural", "tribal", "urb"],
        "block": ["Block North", "Block South", "Block East", "Block North", "Block South", "Block East"],
        "grain_kg": [950.0, 1300.0, 1700.0, 2100.0, 1450.0, 1800.0],
    })
    pipeline = build_model(columns, training, training["candidates"][0])
    pipeline.fit(df[feature_cols], df[columns["target"]])
    preds = pipeline.predict(df[feature_cols])
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_stage4_artifacts_and_k8s_manifests():
    n_runs, n_preds = verify_all(ROOT)
    assert n_runs >= 4
    assert n_preds == 250

    deployment = yaml.safe_load((ROOT / "k8s" / "deployment.yaml").read_text(encoding="utf-8"))
    service = yaml.safe_load((ROOT / "k8s" / "service.yaml").read_text(encoding="utf-8"))

    assert deployment["metadata"]["name"] == "mp124-api"
    assert deployment["spec"]["replicas"] == 2
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp124-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["initialDelaySeconds"] > container["readinessProbe"]["initialDelaySeconds"]

    assert service["metadata"]["name"] == "mp124-svc"
    assert service["spec"]["selector"] == deployment["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["ports"][0]["port"] == 80
    assert service["spec"]["ports"][0]["targetPort"] == 8000
