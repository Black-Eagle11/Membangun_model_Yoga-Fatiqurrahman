#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelling_tuning.py — FIXED VERSION
Author  : Yoga Fatiqurrahman
Tujuan  : Hyperparameter Tuning TANPA preprocessing ulang
"""

import os, json, argparse, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.models.signature import infer_signature

from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss
)

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ========================
# PATH — Hasil Preprocessing
# ========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "namadataset_preprocessing"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ========================
# Load dataset yang SUDAH PREPROCESSING
# ========================
def load_splits():
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    val_csv = DATA_DIR / "val.csv"

    target = "target" if "target" in tr.columns else "condition"
    feats = [c for c in tr.columns if c != target]

    return tr, te, feats, target


# ========================
# Plot functions
# ========================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_fig(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close()

def plot_cm(cm, out):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    save_fig(out)

def plot_roc(y, prob, out):
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    save_fig(out)

def plot_pr(y, prob, out):
    p, r, _ = precision_recall_curve(y, prob)
    plt.figure(figsize=(5,4))
    plt.plot(r, p)
    save_fig(out)


# ========================
# Model Candidates (Tanpa Preprocess)
# ========================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def candidates():
    return {
        "LogReg": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
            {"C": [0.5, 1.0, 3.0]}
        ),
        "SVC": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            {"C": [0.5, 1, 2], "kernel": ["rbf", "linear"]}
        ),
        "RF": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
            {"n_estimators": [150, 250], "max_depth": [None, 10, 20]}
        ),
        "GB": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
        ),
    }


# ========================
# Evaluasi + Log
# ========================
def eval_and_log(model, X, y, phase, out_dir):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:,1]
    else:
        scores = model.decision_function(X)
        prob = (scores - scores.min())/(scores.max() - scores.min() + 1e-8)

    pred = model.predict(X)

    metrics = {
        f"{phase}_acc": accuracy_score(y, pred),
        f"{phase}_prec": precision_score(y, pred),
        f"{phase}_rec": recall_score(y, pred),
        f"{phase}_f1": f1_score(y, pred),
        f"{phase}_roc_auc": roc_auc_score(y, prob),
        f"{phase}_pr_auc": average_precision_score(y, prob),
        f"{phase}_balanced_acc": balanced_accuracy_score(y, pred),
        f"{phase}_mcc": matthews_corrcoef(y, pred),
        f"{phase}_brier": brier_score_loss(y, prob),
    }

    mlflow.log_metrics({k: float(v) for k,v in metrics.items()})

    # confusion matrix
    cm = confusion_matrix(y, pred)
    cm_path = out_dir / f"{phase}_cm.png"
    plot_cm(cm, cm_path)
    mlflow.log_artifact(str(cm_path))

    roc_path = out_dir / f"{phase}_roc.png"
    pr_path = out_dir / f"{phase}_pr.png"
    plot_roc(y, prob, roc_path)
    plot_pr(y, prob, pr_path)
    mlflow.log_artifact(str(roc_path))
    mlflow.log_artifact(str(pr_path))

    return metrics


# ========================
# MAIN
# ========================
def main():
    mlflow.set_experiment("Heart Disease — Tuning")

    tr, te, FEATS, TARGET = load_splits()
    Xtr, ytr = tr[FEATS], tr[TARGET].astype(int)
    Xte, yte = te[FEATS], te[TARGET].astype(int)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", default="f1")
    args = parser.parse_args()

    all_cand = candidates()
    selected = all_cand if args.model == "all" else {args.model: all_cand[args.model]}

    results = {}

    for name, (est, grid) in selected.items():
        with mlflow.start_run(run_name=f"Tuning_{name}"):

            grid_params = {f"{k}": v for k, v in grid.items()}

            gs = GridSearchCV(
                est, grid_params,
                cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED),
                scoring=args.scoring,
                n_jobs=-1
            )

            t0 = time.time()
            gs.fit(Xtr, ytr)
            mlflow.log_metric("cv_time_sec", time.time() - t0)

            best = gs.best_estimator_
            mlflow.log_params({"best_" + k: v for k, v in gs.best_params_.items()})

            out_dir = ARTIFACT_DIR / name
            out_dir.mkdir(exist_ok=True)

            test_metrics = eval_and_log(best, Xte, yte, "test", out_dir)

            # save model
            pkl = out_dir / f"{name}_best.pkl"
            joblib.dump(best, pkl)
            mlflow.log_artifact(str(pkl))

            sig = infer_signature(Xtr.head(3), best.predict(Xtr.head(3)))
            mlflow.sklearn.log_model(best, "model", signature=sig)

            results[name] = test_metrics

    # summary
    summ_sorted = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
    summ_path = REPORT_DIR / "tuning_summary.json"
    json.dump(summ_sorted, open(summ_path, "w"), indent=2)
    mlflow.log_artifact(str(summ_path))

    print("\n>>> BEST MODEL:", summ_sorted[0][0])
    print("Saved summary:", summ_path)


if __name__ == "__main__":
    main()