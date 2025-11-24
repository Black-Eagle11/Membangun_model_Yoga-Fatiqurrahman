#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modelling — Heart Disease (tabular, binary)
Author  : Yoga Fatiqurrahman
Level   : Advanced (Dicoding) — autolog + manual logging (>= 2 metrik tambahan)
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
import platform
import socket
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import shap
import psutil

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.inspection import permutation_importance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
PREP_DIR = ROOT / "preprocessing" / "namadataset_preprocessing"
REPORT_DIR = ROOT / "preprocessing" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MLRUNS_URI = f"file:///{(ROOT / 'mlruns').as_posix()}"


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    _save_fig(out_path)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label="ROC")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, ls=":")
    plt.legend()
    _save_fig(out_path)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, lw=2, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, ls=":")
    _save_fig(out_path)


def _target_col(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "condition" in df.columns:
        return "condition"
    raise ValueError("Kolom target tidak ditemukan ('target' / 'condition').")


def load_preprocessed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    train_csv = PREP_DIR / "train.csv"
    val_csv = PREP_DIR / "val.csv"
    test_csv = PREP_DIR / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Tidak menemukan train/test.csv di {PREP_DIR}. "
            f"Jalankan otomatisasi preprocessing terlebih dahulu."
        )

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        target = _target_col(df_val)
        df_train = pd.concat([df_train, df_val], ignore_index=True)
    else:
        target = _target_col(df_train)

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target].astype(int)
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target].astype(int)

    return X_train, X_test, y_train, y_test, target

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def candidate_models() -> Dict[str, object]:
    return {
        "logreg_l2": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=SEED,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
        "svc_rbf": SVC(
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        ),
    }


def evaluate_and_log(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    classes: Tuple[str, str] = ("0", "1"),
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
        if np.unique(y_true).size == 2
        else float("nan"),
        "avg_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])

    mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()})

    rep_txt = classification_report(y_true, y_pred, digits=4)
    rep_path = REPORT_DIR / f"{name}_classification_report.txt"
    rep_path.write_text(rep_txt)
    mlflow.log_artifact(str(rep_path))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = REPORT_DIR / f"{name}_confusion_matrix.png"
    plot_confusion_matrix(cm, classes=list(classes), out_path=cm_path)
    mlflow.log_artifact(str(cm_path))

    roc_path = REPORT_DIR / f"{name}_roc.png"
    pr_path = REPORT_DIR / f"{name}_pr.png"
    plot_roc(y_true, y_prob, roc_path)
    plot_pr(y_true, y_prob, pr_path)
    mlflow.log_artifact(str(roc_path))
    mlflow.log_artifact(str(pr_path))

    return metrics


def try_feature_importance(pipe: Pipeline, X: pd.DataFrame, name: str) -> None:
    try:
        clf = pipe.named_steps["clf"]
        pre = pipe.named_steps["preprocess"]

        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            transformed_sample = pre.transform(X.iloc[:1])
            feat_names = [f"f{i}" for i in range(transformed_sample.shape[1])]

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(np.ravel(clf.coef_))
        else:
            sample = min(200, len(X))
            idx = np.random.choice(len(X), sample, replace=False)
            pi = permutation_importance(
                pipe,
                X.iloc[idx],
                pipe.predict(X.iloc[idx]),
                n_repeats=5,
                random_state=SEED,
                n_jobs=-1,
            )
            imp = pi.importances_mean

        df_imp = (
            pd.DataFrame({"feature": feat_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        out_csv = REPORT_DIR / f"{name}_feature_importance.csv"
        df_imp.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))

        top = df_imp.head(20)
        plt.figure(figsize=(7, 6))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title(f"Feature Importance — {name}")
        plt.xlabel("Importance")
        fi_png = REPORT_DIR / f"{name}_feature_importance.png"
        _save_fig(fi_png)
        mlflow.log_artifact(str(fi_png))

    except Exception as e:
        print(f"[WARN] Feature importance skipped: {e}")


def try_shap_explain(
    pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, name: str
) -> None:
    try:
        n_bg = min(200, len(X_train))
        n_explain = min(200, len(X_test))
        bg = X_train.sample(n_bg, random_state=SEED)
        ex = X_test.sample(n_explain, random_state=SEED)

        explainer = shap.Explainer(pipe, bg)
        shap_values = explainer(ex)

        shap.summary_plot(shap_values, ex, show=False)
        shap_path = REPORT_DIR / f"{name}_shap_summary.png"
        _save_fig(shap_path)
        mlflow.log_artifact(str(shap_path))
    except Exception as e:
        print(f"[WARN] SHAP explainability skipped for {name}: {e}")


def main():
    try:
        mlflow.end_run()
    except Exception:
        pass
    time.sleep(0.5)

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        print(f"[INFO] Menggunakan MLflow tracking URI dari environment → {env_uri}")
        mlflow.set_tracking_uri(env_uri)
    else:
        mlflow.set_tracking_uri(MLRUNS_URI)

    ap = argparse.ArgumentParser(
        description="Baseline modelling — Heart Disease (tabular, binary)."
    )
    ap.add_argument(
        "--experiment",
        type=str,
        default="Heart Disease — Baseline Models",
        help="Nama eksperimen MLflow.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "logreg_l2", "random_forest", "gradient_boosting", "svc_rbf"],
        help="Model spesifik yang ingin dilatih (default: all).",
    )
    args = ap.parse_args()

    mlflow.set_experiment(args.experiment)
    print(f"\n[INFO] MLflow Tracking URI aktif: {mlflow.get_tracking_uri()}")

    X_train, X_test, y_train, y_test, target_col = load_preprocessed()
    print(
        f"[INFO] Dataset loaded → Train: {X_train.shape}, "
        f"Test: {X_test.shape}, Target: {target_col}"
    )

    preprocessor = build_preprocessor(X_train)

    try:
        mlflow.sklearn.autolog(log_models=True)
        print("[INFO] MLflow sklearn autolog aktif.")
    except Exception as e:
        print(f"[WARN] Autolog gagal diaktifkan: {e}")

    all_models = candidate_models()
    if args.model == "all":
        models = all_models
    else:
        models = {args.model: all_models[args.model]}

    summary: Dict[str, Dict[str, float]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    print(f"\n[INFO] Training {len(models)} model(s): {', '.join(models.keys())}")

    with mlflow.start_run(run_name="baseline_all_models") as parent_run:
        mlflow.set_tag("author", "Yoga Fatiqurrahman")
        mlflow.set_tag("project", "Heart Disease — Baseline Models")
        mlflow.set_tag("hostname", socket.gethostname())
        mlflow.set_tag("os", platform.system())
        mlflow.set_tag("python_version", platform.python_version())

        mlflow.log_param("seed_global", SEED)
        mlflow.log_param("sklearn_version", __import__("sklearn").__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("run_env", os.getenv("GITHUB_WORKFLOW", "local"))

        np.save(REPORT_DIR / "train_feature_shape.npy", np.array(X_train.shape))
        np.save(REPORT_DIR / "test_feature_shape.npy", np.array(X_test.shape))
        mlflow.log_artifact(str(REPORT_DIR / "train_feature_shape.npy"))
        mlflow.log_artifact(str(REPORT_DIR / "test_feature_shape.npy"))

        for name, clf in models.items():
            print(f"\n[INFO] Fitting model: {name} ...")
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("model_name", name)
                mlflow.set_tag("author", "Yoga Fatiqurrahman")

                pipe = Pipeline(
                    [
                        ("preprocess", preprocessor),
                        ("clf", clf),
                    ]
                )

                t0 = time.time()
                pipe.fit(X_train, y_train)
                train_time = time.time() - t0
                mlflow.log_metric("train_time_sec", float(train_time))

                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_test)[:, 1]
                else:
                    try:
                        scores = pipe.decision_function(X_test)
                        y_prob = (scores - scores.min()) / (
                            scores.max() - scores.min() + 1e-8
                        )
                    except Exception:
                        y_prob = np.zeros_like(y_test, dtype=float)

                y_pred = pipe.predict(X_test)

                metrics = evaluate_and_log(
                    y_true=y_test,
                    y_prob=y_prob,
                    y_pred=y_pred,
                    name=name,
                    classes=("0", "1"),
                )
                summary[name] = metrics
                trained_pipelines[name] = pipe

                try:
                    input_example = X_train.head(1)
                    mlflow.sklearn.log_model(
                        sk_model=pipe,
                        artifact_path="model_manual",
                        input_example=input_example,
                    )
                except Exception as e:
                    print(f"[WARN] log_model manual gagal untuk {name}: {e}")

                try_feature_importance(pipe, X_train, name)

                try_shap_explain(pipe, X_train, X_test, name)

        summ_path = REPORT_DIR / "baseline_summary.json"
        summ_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summ_path))

        def _score(m: Dict[str, float]) -> Tuple[float, float]:
            return (m.get("roc_auc", 0.0), m.get("f1", 0.0))

        best_model_name, _ = max(summary.items(), key=lambda kv: _score(kv[1]))
        mlflow.set_tag("best_model", best_model_name)

        best_pipe = trained_pipelines[best_model_name]
        best_checkpoint_path = REPORT_DIR / f"{best_model_name}_pipeline_best.pkl"
        joblib.dump(best_pipe, best_checkpoint_path)
        mlflow.log_artifact(str(best_checkpoint_path))

        (REPORT_DIR / "best_model.txt").write_text(best_model_name)
        mlflow.log_artifact(str(REPORT_DIR / "best_model.txt"))

        mlflow.log_metric("total_models_trained", len(models))

        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        mlflow.log_metric("cpu_usage_percent", cpu_usage)
        mlflow.log_metric("mem_usage_percent", mem_usage)

        total_train_time = sum(m.get("train_time_sec", 0.0) for m in summary.values())

    print("\n[INFO] Training selesai tanpa error.")
    print(f"[INFO] Total models trained: {len(models)}")
    print(f"[INFO] Total training time (approx): {total_train_time:.2f} seconds")
    print(f"[INFO] Best model: {best_model_name}")
    print(f"[INFO] Summary saved → {summ_path}")
    print(f"[INFO] Tracking URI → {mlflow.get_tracking_uri()}")
    print(f"[INFO] CPU Usage (last logged): {cpu_usage:.1f}%")
    print(f"[INFO] Memory Usage (last logged): {mem_usage:.1f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass
