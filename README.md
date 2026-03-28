# mlflow-model-registry-pipeline

![CI](https://github.com/himabindu-peramala/mlflow-model-registry-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![MLflow](https://img.shields.io/badge/mlflow-2.12.2-orange)
![XGBoost](https://img.shields.io/badge/xgboost-2.0.3-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An enhanced MLflow experiment tracking and model registry pipeline for wine quality prediction. This project implements the full ML lifecycle — from data validation and hyperparameter tuning to model explainability, registration, and automated champion/challenger promotion.

---

## Key Enhancements

- **Combined dataset** — Red and white wine datasets merged into a single training set with a `wine_type` binary feature, increasing diversity and model generalizability
- **Data validation gate** — Pre-training quality checks (null detection, physicochemical range validation, class balance, duplicate detection) logged as MLflow run tags
- **Extended Hyperopt search** — 9-parameter Bayesian optimization (TPE) with 5-fold stratified cross-validation, covering depth, regularization, sampling, and learning rate
- **SHAP explainability** — TreeExplainer generates a feature importance summary plot automatically logged as an MLflow artifact after every run
- **Custom PyFunc model** — `StandardScaler` and `XGBClassifier` bundled into a single `mlflow.pyfunc.PythonModel`, making the registered model fully self-contained at serving time
- **Automated champion/challenger promotion** — New model versions are tagged `@challenger`; promotion to `@champion` happens automatically only if test AUC strictly improves over the current champion
- **GitHub Actions CI** — Linting with `flake8` and an MLflow smoke test run on every push to `main`

---

## Project Structure

```
mlflow-model-registry-pipeline/
├── train.py                  # Main training pipeline
├── data_validation.py        # Pre-training data quality checks
├── data/
│   ├── winequality-red.csv   # UCI red wine dataset (1,599 rows)
│   └── winequality-white.csv # UCI white wine dataset (4,898 rows)
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## MLflow Concepts Covered

| Concept | Usage |
|---|---|
| Experiment Tracking | Named experiments, run logging with params, metrics, and tags |
| Artifact Logging | SHAP summary plot stored per run |
| Model Registry | Model versioned and registered on every training run |
| PyFunc Model | Custom wrapper with preprocessing baked in |
| Model Aliases | `@champion` and `@challenger` for deployment lifecycle management |
| MlflowClient | Programmatic registry access for champion comparison and promotion |

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/himabindu-peramala/mlflow-model-registry-pipeline.git
cd mlflow-model-registry-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Validate the Dataset
```bash
python data_validation.py
```
Runs all data quality checks and prints a validation report. Exits with code `1` if critical issues are found.

### 2. Run the Training Pipeline
```bash
python train.py
```

With optional arguments:
```bash
python train.py --experiment-name my-experiment --max-evals 30
```

| Argument | Default | Description |
|---|---|---|
| `--experiment-name` | `wine-quality-pipeline` | MLflow experiment name |
| `--max-evals` | `20` | Number of Hyperopt evaluations |

### 3. Explore Results in MLflow UI
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` to inspect runs, compare metrics, view SHAP artifacts, and browse the model registry.

---

## Tech Stack

- **MLflow 2.12.2** — Experiment tracking, artifact logging, model registry, PyFunc, model aliases
- **XGBoost 2.0.3** — Gradient boosted classifier
- **Hyperopt 0.2.7** — Bayesian hyperparameter search (TPE algorithm)
- **SHAP 0.45.0** — Model explainability via TreeExplainer
- **scikit-learn 1.4.2** — Preprocessing and evaluation metrics
- **pandas / numpy** — Data manipulation
- **GitHub Actions** — CI/CD pipeline

---

## Dataset

UCI Wine Quality Dataset  
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, Elsevier, 47(4):547–553, 2009.  
[UCI Repository](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)

---

> Built upon [MLOps Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs) by Ramin Mohammadi with significant enhancements.
