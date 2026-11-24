"""Compute PSI drift and per-group MAE for MP124 monitoring (PROBLEM.md section 7)."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from mp124.config import ROOT, load_csv, load_params


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Stage 1 cleaning rules: deduplicate, replace sentinel 9999, normalize shop_area."""
    cleaned = df.drop_duplicates().copy()
    if "biometric_failures" in cleaned.columns:
        cleaned["biometric_failures"] = cleaned["biometric_failures"].replace(9999, np.nan)
    if "shop_area" in cleaned.columns:
        cleaned["shop_area"] = (
            cleaned["shop_area"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True)
        )
    return cleaned


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute 10-bin Population Stability Index using quantile edges from expected."""
    exp = expected.dropna().to_numpy()
    act = actual.dropna().to_numpy()

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp, quantiles))
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / len(exp), eps, None)
    act_pct = np.clip(act_counts / len(act), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-findings",
        action="store_true",
        help="Write computed monitoring metrics directly into reports/findings.yaml",
    )
    args = parser.parse_args()

    params = load_params()
    columns = params["columns"]
    numeric_cols = columns["numeric"]
    feature_cols = numeric_cols + columns["categorical"]
    target = columns["target"]

    train = clean_frame(load_csv("train", params))
    later = clean_frame(load_csv("later", params))

    # 1. PSI per numeric feature
    psi_scores = {col: round(compute_psi(train[col], later[col], bins=10), 6) for col in numeric_cols}
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 2. Load v1 from MLflow registry and compute MAE per block on later.csv
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    reg_name = params["training"]["registered_model_name"]
    v1_model = mlflow.sklearn.load_model(f"models:/{reg_name}/1")

    later_preds = v1_model.predict(later[feature_cols])
    later_eval = later.assign(_pred=later_preds)

    mae_by_group = {}
    for block_name in ("Block North", "Block South", "Block East"):
        subset = later_eval[later_eval["block"] == block_name]
        mae_by_group[block_name] = round(float(mean_absolute_error(subset[target], subset["_pred"])), 6)

    worst_group = max(mae_by_group, key=mae_by_group.get)

    summary = {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
    }
    print(yaml.safe_dump({"monitoring": summary}, sort_keys=False))

    if args.update_findings:
        findings_path = ROOT / "reports" / "findings.yaml"
        findings = yaml.safe_load(findings_path.read_text(encoding="utf-8"))
        findings["monitoring"]["psi"] = psi_scores
        findings["monitoring"]["most_drifted_feature"] = most_drifted
        findings["monitoring"]["mae_by_group"] = mae_by_group
        findings["monitoring"]["worst_served_group"] = worst_group
        findings_path.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")
        print(f"Updated {findings_path}")


if __name__ == "__main__":
    main()