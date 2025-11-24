#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, time, warnings, platform, socket, getpass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.models.signature import infer_signature
import dagshub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "namadataset_preprocessing"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"

for d in (REPORT_DIR, ARTIFACT_DIR):
    d.mkdir(exist_ok=True, parents=True)

LOCAL_URI = (ROOT.parent / "mlruns").resolve().as_uri()


def setup_mlflow():
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
        print(f"[INFO] MLflow menggunakan environment URI → {env_uri}")
        return
    try:
        dagshub.init(
            repo_owner="Black-Eagle11", repo_name="Heart_Disease_MLflow", mlflow=True
        )
        mlflow.set_tracking_uri(
            "https://dagshub.com/Black-Eagle11/Heart_Disease_MLflow.mlflow"
        )
        print("[INFO] Terkoneksi ke DagsHub MLflow Tracking.")
    except Exception:
        print("[WARN] Tidak dapat konek DagsHub. Fallback ke MLflow lokal.")
        mlflow.set_tracking_uri(LOCAL_URI)


def save_figure(path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_cm(cm, labels, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    save_figure(out_path)


def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    save_figure(out_path)


def plot_pr(y_true, y_prob, out_path):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    save_figure(out_path)


def load_splits():
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    val = pd.read_csv(DATA_DIR / "val.csv") if (DATA_DIR / "val.csv").exists() else None
    target = "target" if "target" in tr.columns else "condition"
    feats = [c for c in tr.columns if c != target]
    return tr, val, te, feats, target


def make_preprocessor(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    num = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat = (
        Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        if cat_cols
        else None
    )
    transformers = [("num", num, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat, cat_cols))
    return ColumnTransformer(transformers)


def candidates():
    return {
        "LogReg": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
            {"C": [0.1, 1.0, 3.0]},
        ),
        "SVC": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            {"C": [0.5, 1.0, 2.0], "kernel": ["rbf", "linear"]},
        ),
        "RF": (
            RandomForestClassifier(
                random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"
            ),
            {"n_estimators": [150, 300], "max_depth": [None, 10, 20]},
        ),
        "GB": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
    }


def eval_and_log(phase, model, X, y, out_dir):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    y_pred = model.predict(X)
    metrics = {
        f"{phase}_accuracy": accuracy_score(y, y_pred),
        f"{phase}_precision": precision_score(y, y_pred),
        f"{phase}_recall": recall_score(y, y_pred),
        f"{phase}_f1": f1_score(y, y_pred),
        f"{phase}_balanced_acc": balanced_accuracy_score(y, y_pred),
        f"{phase}_roc_auc": roc_auc_score(y, y_prob),
        f"{phase}_pr_auc": average_precision_score(y, y_prob),
        f"{phase}_brier": brier_score_loss(y, y_prob),
        f"{phase}_mcc": matthews_corrcoef(y, y_pred),
    }
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    cm = confusion_matrix(y, y_pred)
    cm_path = out_dir / f"{phase}_cm.png"
    plot_cm(cm, ["0", "1"], cm_path)
    mlflow.log_artifact(str(cm_path))
    roc_path = out_dir / f"{phase}_roc.png"
    pr_path = out_dir / f"{phase}_pr.png"
    plot_roc(y, y_prob, roc_path)
    plot_pr(y, y_prob, pr_path)
    mlflow.log_artifact(str(roc_path))
    mlflow.log_artifact(str(pr_path))
    return metrics


def shap_explain(best_model, X, out_dir):
    try:
        import shap

        sample = X.sample(min(80, len(X)), random_state=SEED)
        X_t = best_model.named_steps["preprocess"].transform(sample)
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        explainer = shap.Explainer(best_model.named_steps["clf"])
        vals = explainer(X_t)
        plt.figure()
        shap.summary_plot(vals, feature_names=feature_names, show=False)
        out_png = out_dir / "shap_summary.png"
        save_figure(out_png)
        mlflow.log_artifact(str(out_png))
    except Exception as e:
        print(f"[WARN] SHAP skipped: {e}")


def main():
    setup_mlflow()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--scoring", default="f1")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    mlflow.set_experiment("Heart Disease — Tuning")
    tr, val, te, FEATS, TARGET = load_splits()
    Xtr, ytr = tr[FEATS], tr[TARGET].astype(int)
    Xte, yte = te[FEATS], te[TARGET].astype(int)
    pre = make_preprocessor(Xtr)
    cand = candidates()
    selected = cand if args.model == "all" else {args.model: cand[args.model]}
    results = {}
    for name, (est, grid) in selected.items():
        with mlflow.start_run(run_name=f"Tuning_{name}"):
            mlflow.log_param("model", name)
            pipe = Pipeline([("preprocess", pre), ("clf", est)])
            gs = GridSearchCV(
                pipe,
                {f"clf__{k}": v for k, v in grid.items()},
                scoring=args.scoring,
                cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED),
                n_jobs=-1,
            )
            t0 = time.time()
            gs.fit(Xtr, ytr)
            mlflow.log_metric("cv_duration_sec", round(time.time() - t0, 3))
            best = gs.best_estimator_
            mlflow.log_params(
                {
                    f"best_{k.replace('clf__', '')}": v
                    for k, v in gs.best_params_.items()
                }
            )
            out_dir = ARTIFACT_DIR / name
            out_dir.mkdir(exist_ok=True, parents=True)
            test_metrics = eval_and_log("test", best, Xte, yte, out_dir)
            model_path = out_dir / f"{name}_best.pkl"
            joblib.dump(best, model_path)
            mlflow.log_artifact(str(model_path))
            sig = infer_signature(Xtr.head(3), best.predict(Xtr.head(3)))
            mlflow.sklearn.log_model(best, "model", signature=sig)
            shap_explain(best, Xtr, out_dir)
            results[name] = test_metrics
    summ = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
    json.dump(summ, open(REPORT_DIR / "tuning_summary.json", "w"), indent=2)
    mlflow.log_artifact(str(REPORT_DIR / "tuning_summary.json"))
    best_model_name = summ[0][0]
    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print("Summary saved:", REPORT_DIR / "tuning_summary.json")


if __name__ == "__main__":
    main()
