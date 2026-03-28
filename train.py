"""
train.py
--------
Enhanced MLflow Model Registry Pipeline — Wine Quality Prediction
Based on the MLOps course lab by Ramin Mohammadi (Lab 2).

Key Enhancements over the original:
  1. Combined red + white wine dataset with wine_type feature
  2. Data validation gate (via data_validation.py)
  3. Extended Hyperopt search space (XGBoost + more hyperparams)
  4. SHAP explainability — summary plot logged as MLflow artifact
  5. Custom PyFunc wrapper — StandardScaler baked into the model
  6. Automated champion/challenger promotion — only promotes if AUC improves
  7. Model aliases (@champion, @challenger) using the modern MLflow Registry API

Usage:
    python train.py [--experiment-name NAME] [--max-evals N]

    --experiment-name : MLflow experiment name (default: wine-quality-pipeline)
    --max-evals       : Number of Hyperopt evaluations (default: 20)
"""

import argparse
import logging
import os
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from xgboost import XGBClassifier

from data_validation import run_validation

matplotlib.use("Agg")  # Non-interactive backend for artifact saving
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "wine-quality-classifier"
RED_DATA_PATH = "data/winequality-red.csv"
WHITE_DATA_PATH = "data/winequality-white.csv"
TARGET_COL = "quality"
RANDOM_STATE = 42


# ── 1. Data Loading ────────────────────────────────────────────────────────────
def load_data(red_path: str, white_path: str) -> tuple:
    """Load, combine, and split the wine quality dataset."""
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red["wine_type"] = 0       # red wine
    white["wine_type"] = 1     # white wine
    df = pd.concat([red, white], ignore_index=True)

    # Binarize quality: good (1) if quality >= 7, bad (0) otherwise
    df["quality"] = (df["quality"] >= 7).astype(int)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Dataset: %d train / %d test | Positive class: %.1f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ── 2. Custom PyFunc Model (scaler baked in) ───────────────────────────────────
class WineQualityModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc wrapper that bundles StandardScaler + XGBClassifier.
    Preprocessing is embedded so the model is fully self-contained at serving time.
    """

    def __init__(self, scaler: StandardScaler, classifier: XGBClassifier):
        self.scaler = scaler
        self.classifier = classifier

    def predict(self, context, model_input):
        scaled = self.scaler.transform(model_input)
        return self.classifier.predict(scaled)

    def predict_proba(self, model_input):
        scaled = self.scaler.transform(model_input)
        return self.classifier.predict_proba(scaled)


# ── 3. SHAP Explainability ─────────────────────────────────────────────────────
def log_shap_summary(model: XGBClassifier, X_test_scaled: np.ndarray, feature_names: list):
    """Generate and log a SHAP summary plot as an MLflow artifact."""
    logger.info("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_scaled,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Wine Quality Classifier")
    plt.tight_layout()

    shap_path = "shap_summary.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(shap_path, artifact_path="explainability")
    os.remove(shap_path)
    logger.info("SHAP summary plot logged to MLflow artifacts.")


# ── 4. Hyperopt Objective ──────────────────────────────────────────────────────
def build_objective(X_train_scaled, y_train):
    """
    Returns a Hyperopt objective function.
    Extended search space vs original: adds subsample, colsample_bytree,
    min_child_weight, gamma for richer exploration.
    """

    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])

        clf = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
        mean_auc = auc_scores.mean()

        return {"loss": -mean_auc, "status": STATUS_OK, "auc": mean_auc}

    return objective


SEARCH_SPACE = {
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "n_estimators": hp.quniform("n_estimators", 50, 300, 50),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    "gamma": hp.uniform("gamma", 0, 0.5),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(1.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(1.0)),
}


# ── 5. Champion/Challenger Promotion ──────────────────────────────────────────
def get_champion_auc(client: MlflowClient, model_name: str) -> float:
    """Return the AUC of the current @champion model, or 0.0 if none exists."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(champion.run_id)
        return float(run.data.metrics.get("test_auc", 0.0))
    except Exception:
        return 0.0


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    new_auc: float,
) -> str:
    """
    Compare new model AUC to the current champion.
    Assign @challenger to new model. Promote to @champion only if AUC improves.
    Returns 'promoted' or 'challenger'.
    """
    champion_auc = get_champion_auc(client, model_name)
    logger.info("Champion AUC: %.4f | Challenger AUC: %.4f", champion_auc, new_auc)

    # Always tag new model as challenger first
    client.set_registered_model_alias(model_name, "challenger", new_version)
    logger.info("Model v%s tagged as @challenger.", new_version)

    if new_auc > champion_auc:
        client.set_registered_model_alias(model_name, "champion", new_version)
        logger.info("New model PROMOTED to @champion (AUC %.4f > %.4f).", new_auc, champion_auc)
        return "promoted"
    else:
        logger.info("Champion retained — new model stays as @challenger.")
        return "challenger"


# ── 6. Main Training Pipeline ──────────────────────────────────────────────────
def train(experiment_name: str, max_evals: int):
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # ── Step 1: Data Validation ──────────────────────────────────────────────
    logger.info("Running data validation...")
    with mlflow.start_run(run_name="data-validation", nested=False) as val_run:
        mlflow.set_tag("stage", "data_validation")
        passed = run_validation(RED_DATA_PATH, WHITE_DATA_PATH)
        if not passed:
            logger.error("Data validation FAILED. Aborting training.")
            return

    # ── Step 2: Load & Scale Data ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data(RED_DATA_PATH, WHITE_DATA_PATH)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X_train.columns.tolist()

    # ── Step 3: Hyperopt Tuning ──────────────────────────────────────────────
    logger.info("Starting Hyperopt search (max_evals=%d)...", max_evals)
    with mlflow.start_run(run_name="hyperopt-tuning") as parent_run:
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("dataset", "red+white wine (combined)")
        mlflow.set_tag("author", "mlflow-model-registry-pipeline")

        trials = Trials()
        objective = build_objective(X_train_scaled, y_train)
        best_params = fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_STATE),
        )

        best_auc = -min(t["result"]["loss"] for t in trials.trials)
        logger.info("Best CV AUC: %.4f | Best params: %s", best_auc, best_params)

        # ── Step 4: Train Final Model ────────────────────────────────────────
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["min_child_weight"] = int(best_params["min_child_weight"])

        final_model = XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        final_model.fit(X_train_scaled, y_train)

        # ── Step 5: Evaluate ─────────────────────────────────────────────────
        y_pred = final_model.predict(X_test_scaled)
        y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        test_acc = accuracy_score(y_test, y_pred)

        logger.info("Test AUC: %.4f | Test Accuracy: %.4f", test_auc, test_acc)

        # ── Step 6: Log Params, Metrics, Tags ───────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_auc", best_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_accuracy", test_acc)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # ── Step 7: SHAP Explainability ──────────────────────────────────────
        log_shap_summary(final_model, X_test_scaled, feature_names)

        # ── Step 8: Register Custom PyFunc Model ─────────────────────────────
        pyfunc_model = WineQualityModel(scaler=scaler, classifier=final_model)
        mlflow.pyfunc.log_model(
            python_model=pyfunc_model,
            artifact_path="wine_quality_pyfunc",
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:5],
        )

        # Wait briefly for registration to complete
        time.sleep(5)

        # ── Step 9: Champion/Challenger Promotion ────────────────────────────
        latest = client.get_latest_versions(MODEL_NAME)
        latest_version = max(v.version for v in latest)
        outcome = promote_if_better(client, MODEL_NAME, latest_version, test_auc)
        mlflow.set_tag("promotion_outcome", outcome)

        logger.info("Pipeline complete. Model v%s → %s", latest_version, outcome.upper())
        logger.info("Run ID: %s", parent_run.info.run_id)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Wine Quality MLflow Pipeline")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="wine-quality-pipeline",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=20,
        help="Number of Hyperopt evaluations (default: 20)",
    )
    args = parser.parse_args()
    train(args.experiment_name, args.max_evals)
