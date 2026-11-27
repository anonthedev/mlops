# MP124 - Ration Shop Distribution (Bhopal)

SCSE3040 MLOps mini project.

- **Group:** G013
- **Problem:** MP124 (predict `grain_kg`, the grain a ration shop hands out in a day; regression, scored by MAE)
- **Faculty:** Dr. Amit Soni, Batch 35

| Enrolment no. | Name |
|---|---|
| S24CSEU1564 | Rishabh Choudhary |
| S24CSEU2137 | Akshaj Pandey |
| S24CSEU1549 | Pranav Kumar |
| S24CSEU1581 | Ayush Yadav |

## Status

All four stages are implemented and verified:
1. **Stage 1 (Data & DVC):** Repository layout, DVC tracking for `train.csv` and `later.csv`, the seven data checks in `src/mp124/checks.py`, and `data_issues` in `reports/findings.yaml`.
2. **Stage 2 (Training & MLflow):** Cleaning transformer and candidate models (`src/mp124/model.py`), chronological validation across 3 candidates logged to `mlflow.db`, and registered model `mp124-grain-regressor`.
3. **Stage 3 (Service, Docker, CI, K8s & Logging):** FastAPI prediction service (`app/main.py`) with Pydantic boundary rules (`cards_attached <= 3500`, `biometric_failures >= 0`), structured stdout JSON logging (`app/logging_config.py`), `Dockerfile` + `.dockerignore`, GitHub Actions workflow (`.github/workflows/ci.yml`), Kubernetes manifests (`k8s/`), and pytest suite (`tests/`).
4. **Stage 4 (Monitoring, Retraining, Holdout & Audit):** PSI per numeric column and per-block MAE (`src/mp124/monitor.py`), v2 retraining on cleaned `train.csv` + first 70% of `later.csv` (`src/mp124/train.py`), promotion check (27.9% MAE reduction -> `promote` v2 as `@champion`), `predictions/holdout_predictions.csv`, completed `reports/findings.yaml`, and end-to-end artifact verification (`src/mp124/verify.py`).

## Layout

```
README.md                        project overview and run commands
requirements.txt                 pinned Python dependencies
params.yaml                      paths, columns, thresholds, training candidates
Dockerfile                       python:3.12-slim API container (port 8000)
.dockerignore                    container build exclusions
.github/workflows/ci.yml         runs pytest on every push
app/
  __init__.py
  logging_config.py              stdout JSON formatter for predictions
  main.py                        FastAPI app: GET /health and POST /predict
k8s/
  deployment.yaml                Deployment mp124-api (2 replicas, probes on /health)
  service.yaml                   Service mp124-svc (port 80 -> 8000)
src/mp124/
  __init__.py
  config.py                      loads params.yaml, file hashing, CSV loading
  checks.py                      the seven data-problem checks
  find_issues.py                 populates data_issues in reports/findings.yaml
  model.py                       feature cleaning and scikit-learn pipelines
  train.py                       MLflow experiment tracking, registry v1/v2, holdout CSV
  monitor.py                     PSI drift, per-block MAE, promotion helper
  verify.py                      Stage 4 audit of findings, DVC, MLflow, model, holdout
tests/
  conftest.py                    adds src/ to sys.path for pytest
  test_api.py                    FastAPI TestClient tests (health, bounds, JSON logs)
  test_pipeline.py               cleaning, data checks, model fit, Stage 4 & K8s verification
  test_stage4_monitoring.py      PSI, group MAE, chronological split, 10% promotion rule
data/raw/
  train.csv.dvc                  DVC pointer (CSV excluded from git)
  later.csv.dvc                  DVC pointer (CSV excluded from git)
  holdout_features.csv           holdout feature rows (250 records)
.dvc/config                      DVC configuration
mlflow.db                        SQLite tracking store and model registry
mlartifacts/                     logged MLflow model artifacts
models/model.joblib              exported @champion pipeline
predictions/holdout_predictions.csv
reports/findings.yaml            completed findings form
docs/                            PROBLEM.md, GROUP.md, STUDENT_HANDBOOK.md
```

## Run & Verify

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the test suite (works without raw train/later CSVs)
python -m pytest -v

# Run Stage 4 artifact & findings audit
PYTHONPATH=src python -m mp124.verify

# Start the FastAPI prediction service locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Data Checks, Training & Monitoring (requires original `train.csv` and `later.csv` in `data/raw/`)

```bash
dvc status
PYTHONPATH=src python -m mp124.find_issues --dry-run
PYTHONPATH=src python -m mp124.train --output-dir .venv/mp124-reproduction
PYTHONPATH=src python -m mp124.monitor
```

### Container & Kubernetes

```bash
docker build -t mp124-api:1.0 .
docker run --rm -p 8000:8000 mp124-api:1.0
kubectl apply -f k8s/
```

## Stage 1 & Stage 4 Summary

- `train.csv` SHA-256: `1f6f6cad6e2959cb7840d884728b614aa1360d629ec8c41a8330985ef8f1ffb3`
- **Data issues found:** `sentinel_value` (`biometric_failures = 9999` on 44 rows), `duplicate_rows` (38 extra copies), `spelling_variants` (`shop_area` -> 3 levels: `rural`, `tribal`, `urban`).
- **Drift & group MAE (v1 on `later.csv`):** most drifted numeric feature `cards_attached` (`PSI = 5.496764`); worst-served group `Block South` (`MAE = 557.140678` kg).
- **Retraining & promotion (last 30% of `later.csv`):** v1 MAE `167.7568` kg -> v2 MAE `120.9564` kg (`promote`, `@champion` = version 2).
