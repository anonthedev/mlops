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

Stage 1 is done: repo layout, DVC tracking of `train.csv` and `later.csv`, the seven data checks in `src/`, and the `data_issues` part of `reports/findings.yaml`.
The model workflow tracks three initial settings in MLflow, registers v1 and v2, and exports the registry champion with its holdout predictions.

## Layout (so far)

```
README.md
requirements.txt              pinned libraries
params.yaml                   paths, column names, check thresholds (read by src/mp124/config.py)
src/mp124/__init__.py
src/mp124/config.py           loads params.yaml, file hashing, CSV loading
src/mp124/checks.py           the seven data-problem checks (handbook section 7)
src/mp124/find_issues.py      runs the checks, fills reports/findings.yaml
src/mp124/model.py            preprocessing and model definitions
src/mp124/train.py            training, experiment tracking, registry and holdout prediction
data/raw/train.csv.dvc        DVC pointer (the CSV itself is not committed)
data/raw/later.csv.dvc        DVC pointer
data/raw/holdout_features.csv committed as is
.dvc/config                   from `dvc init`
mlflow.db                     SQLite tracking and model registry
mlartifacts/                  MLflow files for registered models
models/model.joblib           champion pipeline for the prediction service
predictions/holdout_predictions.csv
reports/findings.yaml         data issues, tracking and model comparison
docs/                         PROBLEM.md, GROUP.md, STUDENT_HANDBOOK.md from the zip
```

## Put it on GitHub (Stage 1 "commit" step)

1. One member creates a public GitHub repo and adds the other three as collaborators.
2. Everyone clones it. Copy this folder's contents into the clone, including the hidden `.dvc/`, `.dvcignore` and `.gitignore`.
3. Then run:

```bash
git add .
git status        # train.csv and later.csv must NOT be listed; their .dvc files must be
git commit -m "Stage 1: repo layout, DVC tracking, seven data checks, findings data part"
git push
```

Each member should commit their own part from their own GitHub account. The marking tool counts commits per person.

## Run it

Use Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# The CSVs are not in git. Copy the ORIGINAL train.csv and later.csv from the
# problem zip into data/raw/ (never open and save them in Excel).
dvc status                          # "up to date" = your files match the .dvc hashes

# Hunt the seven data problems and write them into reports/findings.yaml
PYTHONPATH=src python -m mp124.find_issues
# print only, don't touch the file:
PYTHONPATH=src python -m mp124.find_issues --dry-run
```

On Windows PowerShell, use `$env:PYTHONPATH="src"; python -m mp124.find_issues`.

## Reproduce the MLflow runs

After copying the original CSVs into `data/raw/`, run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m mp124.train --output-dir .venv/mp124-reproduction
```

Use a fresh output directory for each reproduction. The command creates a new tracking
database, logs the SHA-256 of the original `train.csv` in every run, and prints
the tracking fields and promotion decision for the findings report.

The last 20% of the deduplicated training rows by date select the v1 settings.
The winning settings are refit on all deduplicated training rows. v2 adds the
first 70% of `later.csv` by date and receives the `champion` alias only if
its MAE on the remaining 30% improves by at least 10%.

Open the committed experiment from the repository root with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. The registered model is
`mp124-grain-regressor`. The artifact paths are relative to the directory
that contains the tracking database, so start MLflow from that directory.

## Stage 1 results (train.csv as given, 1038 rows)

train.csv SHA-256: `1f6f6cad6e2959cb7840d884728b614aa1360d629ec8c41a8330985ef8f1ffb3`

| Problem | Present | Details |
|---|---|---|
| leaking_column | no | holdout has every feature column; no feature has a correlation with `grain_kg` above 0.68 |
| unit_mixup | no | every column has nearly the same mean in all three blocks (largest gap 7%) |
| sentinel_value | **yes** | `biometric_failures` = 9999 on 44 rows (the real values are 0-80) |
| duplicate_rows | **yes** | 38 extra copies |
| stuck_sensor | no | the longest repeated-value run lasts 2 days |
| target_typos | no | max `grain_kg` is 2022.3, only 1.8x the median; no gap |
| spelling_variants | **yes** | `shop_area` has 18 spellings that clean up to 3 levels (rural, tribal, urban) |
