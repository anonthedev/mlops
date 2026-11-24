"""Train and register the MP124 models, then predict with the champion."""

import argparse
import sqlite3
from contextlib import chdir, closing
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mp124.config import ROOT, file_sha256, load_csv, load_params, resolve
from mp124.model import build_model


def configure_tracking():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")


def make_artifact_paths_relative(db_path, artifact_location, default_artifact_location):
    """Make the finished SQLite export reloadable from another checkout."""
    # MLflow 3.18 resolves local artifact locations before storing them.
    # Export only this experiment's artifact root and its child paths.
    prefix = artifact_location + "/"
    fields = (
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
        ("model_versions", "storage_location"),
    )
    with closing(sqlite3.connect(db_path)) as database, database:
        database.execute("PRAGMA secure_delete = ON")
        database.execute(
            "UPDATE experiments SET artifact_location = 'mlruns/0' "
            "WHERE experiment_id = 0 AND artifact_location = ?",
            (default_artifact_location,),
        )
        for table, field in fields:
            database.execute(
                f"UPDATE {table} SET {field} = 'mlartifacts' || substr({field}, ?) "
                f"WHERE {field} = ? OR substr({field}, 1, ?) = ?",
                (len(artifact_location) + 1, artifact_location, len(prefix), prefix),
            )

    for manifest_path in Path("mlartifacts").glob("models/*/artifacts/MLmodel"):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_path"] = manifest_path.parent.as_posix()
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def chronological_split(frame, date_column, first_fraction):
    ordered = frame.sort_values(date_column, kind="stable")
    cutoff = int(len(ordered) * first_fraction)
    if cutoff <= 0 or cutoff >= len(ordered):
        raise ValueError("The split must leave rows on both sides of the cutoff.")
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def log_candidate(candidate, training, data_sha256, fitted_rows):
    mlflow.set_tag("mlflow.source.name", "src/mp124/train.py")
    mlflow.log_params({
        "data_sha256": data_sha256,
        "estimator": candidate["estimator"],
        "fit_rows": fitted_rows,
        "random_state": training["random_state"],
        **candidate["settings"],
    })


def log_model(model, name, features):
    pip_requirements = [
        f"{package}=={version(package)}"
        for package in ("mlflow", "cloudpickle", "numpy", "pandas", "scikit-learn", "scipy")
    ]
    # Numeric inputs can be missing at inference, which uses float NaN.
    example = features.head(5).astype({col: float for col in features.select_dtypes("number")})
    return mlflow.sklearn.log_model(
        model,
        name=name,
        serialization_format="cloudpickle",
        signature=infer_signature(example, model.predict(example)),
        code_paths=[str(ROOT / "src" / "mp124")],
        pip_requirements=pip_requirements,
    )


def run_training(params, train, later, holdout, data_sha256):
    columns, training = params["columns"], params["training"]
    feature_columns = columns["numeric"] + columns["categorical"]
    target = columns["target"]
    train_fit, train_validation = chronological_split(
        train, columns["date"], 1 - training["train_validation_fraction"]
    )
    later_fit, later_check = chronological_split(
        later, columns["date"], training["later_training_fraction"]
    )

    results = []
    for candidate in training["candidates"]:
        estimator = build_model(columns, training, candidate)
        estimator.fit(train_fit[feature_columns], train_fit[target])
        mae = mean_absolute_error(
            train_validation[target], estimator.predict(train_validation[feature_columns])
        )
        results.append((candidate, estimator, mae))
    winner, _, _ = min(results, key=lambda result: result[2])

    configure_tracking()
    client = MlflowClient()
    experiment_name = training["experiment_name"]
    experiment_id = client.create_experiment(experiment_name, artifact_location="mlartifacts")
    registered_name = training["registered_model_name"]
    check_features = later_check[feature_columns]

    for candidate, estimator, mae in results:
        selected = candidate == winner
        with mlflow.start_run(experiment_id=experiment_id, run_name=candidate["name"]) as run:
            log_candidate(candidate, training, data_sha256, len(train) if selected else len(train_fit))
            mlflow.log_param("validation_fit_rows", len(train_fit))
            mlflow.log_metric("mae_validation", mae)
            if selected:
                v1 = clone(estimator).fit(train[feature_columns], train[target])
                mae_v1 = mean_absolute_error(later_check[target], v1.predict(check_features))
                mlflow.log_metric("mae_later_check", mae_v1)
                v1_model = log_model(v1, "v1", train[feature_columns])
                best_run_id = run.info.run_id
        print(f"{candidate['name']} validation MAE = {mae:.4f} kg")

    v1_version = mlflow.register_model(v1_model.model_uri, registered_name)
    client.set_registered_model_alias(registered_name, "champion", v1_version.version)

    combined = pd.concat([train, later_fit], ignore_index=True)
    with mlflow.start_run(experiment_id=experiment_id, run_name="retrain-v2"):
        log_candidate(winner, training, data_sha256, len(combined))
        mlflow.log_params({
            "later_fit_rows": len(later_fit),
            "later_check_rows": len(later_check),
            "minimum_mae_reduction": training["minimum_mae_reduction"],
        })
        v2 = clone(v1).fit(combined[feature_columns], combined[target])
        mae_v2 = mean_absolute_error(later_check[target], v2.predict(check_features))
        promote = mae_v1 > 0 and mae_v2 <= mae_v1 * (1 - training["minimum_mae_reduction"])
        decision = "promote" if promote else "keep"
        mlflow.set_tag("promotion_decision", decision)
        mlflow.log_metrics({"mae_later_check": mae_v2, "mae_v1_later_check": mae_v1})
        v2_model = log_model(v2, "v2", combined[feature_columns])

    v2_version = mlflow.register_model(v2_model.model_uri, registered_name)
    if promote:
        client.set_registered_model_alias(registered_name, "champion", v2_version.version)

    champion = client.get_model_version_by_alias(registered_name, "champion")
    champion_model = mlflow.sklearn.load_model(f"models:/{registered_name}@champion")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(champion_model, "models/model.joblib", compress=3)
    Path("predictions").mkdir(exist_ok=True)
    pd.DataFrame({
        "record_id": holdout[columns["id"]],
        "prediction": champion_model.predict(holdout[feature_columns]),
    }).to_csv("predictions/holdout_predictions.csv", index=False)
    make_artifact_paths_relative(
        "mlflow.db",
        client.get_experiment(experiment_id).artifact_location,
        client.get_experiment("0").artifact_location,
    )

    return {
        "tracking": {
            "experiment_name": experiment_name,
            "best_run_id": best_run_id,
            "registered_model_name": registered_name,
            "champion_version": int(champion.version),
        },
        "monitoring": {
            "mae_v1_on_check": float(mae_v1),
            "mae_v2_on_check": float(mae_v2),
            "decision": decision,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="directory for a new MLflow database, artifacts and predictions")
    args = parser.parse_args()

    params = load_params()
    train = load_csv("train", params).drop_duplicates()
    later, holdout = (load_csv(name, params) for name in ("later", "holdout"))
    data_sha256 = file_sha256(resolve(params["data"]["train"]))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with chdir(output):
        db = Path("mlflow.db")
        try:
            db.touch(exist_ok=False)
        except FileExistsError:
            parser.error("mlflow.db already exists in the output directory. Choose a fresh --output-dir.")
        findings = run_training(params, train, later, holdout, data_sha256)
        print(yaml.safe_dump(findings, sort_keys=False))


if __name__ == "__main__":
    main()
