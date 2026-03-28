# mlflow-model-registry-pipeline

![CI](https://github.com/himabindu-peramala/mlflow-model-registry-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![MLflow](https://img.shields.io/badge/mlflow-2.12.2-orange)

An enhanced MLflow experiment tracking and model registry pipeline for wine quality prediction.  
Built upon the [MLOps Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs) by Ramin Mohammadi with additional enhancements.

---

## Key Enhancements

| Feature | Original Lab | This Project |
|---|---|---|
| Dataset | Red wine only | Red + White wine combined with `wine_type` feature |
| Hyperparameter search | 3 params | 9 params with extended Hyperopt search space |
| Data quality | None | Automated validation with MLflow tag logging |
| Explainability | None | SHAP summary plot logged as MLflow artifact |
| Model packaging | Raw sklearn model | Custom PyFunc wrapper (scaler baked in) |
| Model promotion | Manual staging | Automated champion/challenger via MLflow aliases |
| CI/CD | None | GitHub Actions — lint + MLflow smoke test |

---

## Project Structure

```
├── train.py               # Enhanced training pipeline
├── data_validation.py     # Data quality checks
├── data/                  # Red + white wine datasets (UCI)
├── .github/workflows/     # CI/CD
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Validate data
python data_validation.py

# Train + register model
python train.py --experiment-name wine-quality-pipeline --max-evals 20

# View results
mlflow ui
```

---

## MLflow Concepts Covered

`Experiment Tracking` · `Run Logging` · `Artifact Logging` · `Model Registry` · `PyFunc Model` · `Model Aliases (@champion/@challenger)` · `MlflowClient`

---

## Dataset

UCI Wine Quality Dataset — P. Cortez et al., 2009.
