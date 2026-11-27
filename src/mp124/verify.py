"""Stage 4 verification: audit findings.yaml, DVC pointers, MLflow DB, model, and holdout predictions."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from mp124.config import ROOT, load_params
from mp124.monitor import should_promote


def verify_dvc_pointers(root: Path = ROOT) -> None:
    """Validate DVC pointer metadata and verify local CSV hashes when CSVs exist."""
    raw_dir = root / "data" / "raw"
    for name in ("train", "later"):
        pointer_path = raw_dir / f"{name}.csv.dvc"
        assert pointer_path.is_file(), f"Missing {pointer_path}"
        pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
        out = pointer["outs"][0]
        assert out["path"] == f"{name}.csv"
        csv_path = raw_dir / f"{name}.csv"
        if csv_path.is_file():
            content = csv_path.read_bytes()
            assert len(content) == out["size"], f"{name}.csv size mismatch"
            assert hashlib.md5(content).hexdigest() == out["md5"], f"{name}.csv md5 mismatch"


def verify_findings_structure(findings: dict, params: dict) -> None:
    """Ensure findings.yaml has required keys, valid nulls, and consistent promotion decision."""
    assert findings.get("group_id") == "G013"
    assert findings.get("problem_id") == "MP124"

    issues = findings.get("data_issues", {})
    assert len(issues) == 7
    for issue_name, details in issues.items():
        present = details.get("present")
        assert isinstance(present, bool), f"{issue_name}.present must be bool"
        for field, value in details.items():
            if field == "present":
                continue
            if present:
                assert value is not None, f"{issue_name}.{field} must not be null when present=true"
            else:
                assert value is None, f"{issue_name}.{field} must be null when present=false"

    tracking = findings.get("tracking", {})
    for key in ("experiment_name", "best_run_id", "registered_model_name", "champion_version"):
        assert tracking.get(key) is not None, f"tracking.{key} is missing"

    monitoring = findings.get("monitoring", {})
    for key in (
        "psi",
        "most_drifted_feature",
        "mae_by_group",
        "worst_served_group",
        "mae_v1_on_check",
        "mae_v2_on_check",
        "decision",
    ):
        assert monitoring.get(key) is not None, f"monitoring.{key} is missing"

    assert monitoring["most_drifted_feature"] == max(monitoring["psi"], key=monitoring["psi"].get)
    assert monitoring["worst_served_group"] == max(
        monitoring["mae_by_group"], key=monitoring["mae_by_group"].get
    )
    expected_decision = (
        "promote"
        if should_promote(
            monitoring["mae_v1_on_check"],
            monitoring["mae_v2_on_check"],
            params["training"]["minimum_mae_reduction"],
        )
        else "keep"
    )
    assert monitoring["decision"] == expected_decision
    expected_champion = 2 if expected_decision == "promote" else 1
    assert int(tracking["champion_version"]) == expected_champion


def verify_mlflow_db(db_path: Path, findings: dict) -> int:
    """Verify MLflow SQLite integrity, runs, data_sha256, MAE metrics, versions, and @champion."""
    assert db_path.is_file(), f"Missing {db_path}"
    tracking = findings["tracking"]
    reg_name = tracking["registered_model_name"]

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        runs = db.execute("SELECT run_uuid, status FROM runs").fetchall()
        assert len(runs) >= 3, "Expected at least 3 MLflow runs"
        for run_uuid, status in runs:
            assert status == "FINISHED", f"Run {run_uuid} has status {status}"
            sha_rows = db.execute(
                "SELECT value FROM params WHERE run_uuid = ? AND key = 'data_sha256'",
                (run_uuid,),
            ).fetchall()
            assert len(sha_rows) == 1 and len(sha_rows[0][0]) == 64
            mae_rows = db.execute(
                "SELECT key FROM metrics WHERE run_uuid = ? AND LOWER(key) LIKE '%mae%'",
                (run_uuid,),
            ).fetchall()
            assert len(mae_rows) >= 1, f"Run {run_uuid} missing MAE metric"

        versions = db.execute(
            "SELECT version, run_id FROM model_versions WHERE name = ?",
            (reg_name,),
        ).fetchall()
        assert len(versions) >= 2, "Expected at least 2 registered model versions"
        v1_runs = [run_id for ver, run_id in versions if int(ver) == 1]
        assert v1_runs and v1_runs[0] == tracking["best_run_id"]

        aliases = db.execute(
            "SELECT version FROM registered_model_aliases WHERE name = ? AND alias = 'champion'",
            (reg_name,),
        ).fetchall()
        assert len(aliases) == 1
        assert int(aliases[0][0]) == int(tracking["champion_version"])

    return len(runs)


def verify_holdout_outputs(root: Path, params: dict) -> int:
    """Check models/model.joblib and predictions/holdout_predictions.csv alignment."""
    columns = params["columns"]
    feature_cols = columns["numeric"] + columns["categorical"]
    id_col = columns["id"]

    holdout = pd.read_csv(root / params["data"]["holdout"])
    preds = pd.read_csv(root / "predictions" / "holdout_predictions.csv")
    assert list(preds.columns) == ["record_id", "prediction"]
    assert preds["record_id"].is_unique
    assert preds["record_id"].equals(holdout[id_col])
    assert np.isfinite(preds["prediction"]).all()

    model = joblib.load(root / "models" / "model.joblib")
    recomputed = model.predict(holdout[feature_cols])
    np.testing.assert_allclose(preds["prediction"].to_numpy(), recomputed, rtol=1e-6, atol=1e-6)
    return len(preds)


def verify_all(root: Path = ROOT) -> tuple[int, int]:
    """Run all offline Stage 4 verification checks."""
    params = load_params(root / "params.yaml")
    findings = yaml.safe_load((root / "reports" / "findings.yaml").read_text(encoding="utf-8"))

    verify_dvc_pointers(root)
    verify_findings_structure(findings, params)
    n_runs = verify_mlflow_db(root / "mlflow.db", findings)
    n_preds = verify_holdout_outputs(root, params)
    return n_runs, n_preds


def main() -> None:
    n_runs, n_preds = verify_all(ROOT)
    print(f"Verified Stage 4 artifacts: {n_runs} MLflow runs, @champion synced, {n_preds} holdout predictions.")


if __name__ == "__main__":
    main()
