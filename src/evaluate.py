"""
Evaluate models on the UNSEEN test set and choose the decision threshold.
Run from project root:  python -m src.evaluate

Threshold rule (decided on TRAIN data only, via out-of-fold scores):
    flag the top MAX_REVIEW_RATE of transactions for manual review.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.preprocessing import load_and_split

SEED = 42
MAX_REVIEW_RATE = 0.10   # business assumption: analysts can manually review at most 10% of transactions
MEDIUM_RATE = 0.30       # top 30% of scores = MEDIUM or higher
CANDIDATES = ["logistic_regression", "random_forest"]


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    os.makedirs("reports", exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_split(seed=SEED)
    selected = json.load(open("models/training_summary.json"))["selected_model"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, test_scores, thresholds, oof_scores = {}, {}, {}, {}
    for name in CANDIDATES:
        model = joblib.load(f"models/{name}.joblib")
        # threshold from out-of-fold scores on TRAIN -> test set never used to pick it
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = float(np.quantile(oof, 1 - MAX_REVIEW_RATE))
        proba = model.predict_proba(X_test)[:, 1]
        rows[name] = metrics_at(y_test, proba, thr)
        test_scores[name], thresholds[name], oof_scores[name] = proba, thr, oof

    # baseline: predicts "legit" for everything
    zeros = np.zeros(len(y_test))
    rows["baseline_dummy"] = metrics_at(y_test, zeros, 0.5)

    table = pd.DataFrame(rows).T
    table.to_csv("reports/metrics.csv")
    show = table[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TP", "FP", "FN", "TN"]]
    print(show.round(4).to_string())
    print(f"\nSelected model (chosen by CV before seeing test): {selected}")

    sel_thr = thresholds[selected]
    sel_med = float(np.quantile(oof_scores[selected], 1 - MEDIUM_RATE))
    sel = rows[selected]

    # confusion matrix plot
    cm = np.array([[sel["TN"], sel["FP"]], [sel["FN"], sel["TP"]]])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Legit", "Pred Fraud"], yticklabels=["Actual Legit", "Actual Fraud"])
    plt.title(f"Confusion matrix: {selected} (threshold {sel_thr:.3f})")
    plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

    # precision-recall curves
    plt.figure(figsize=(6, 4.5))
    for name in CANDIDATES:
        p, r, _ = precision_recall_curve(y_test, test_scores[name])
        plt.plot(r, p, label=f"{name} (PR-AUC {rows[name]['pr_auc']:.3f})")
    plt.axhline(y_test.mean(), ls="--", c="gray", label=f"baseline ({y_test.mean():.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve (test set)")
    plt.legend(); plt.tight_layout(); plt.savefig("reports/pr_curve.png", dpi=150); plt.close()

    # metadata used by the API
    meta = {
        "model_name": selected,
        "decision_threshold": round(sel_thr, 4),     # score >= this -> HIGH risk -> REVIEW
        "medium_threshold": round(sel_med, 4),       # score >= this -> MEDIUM risk
        "max_review_rate": MAX_REVIEW_RATE,
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sel.items()},
    }
    json.dump(meta, open("models/metadata.json", "w"), indent=2)

    base = rows["baseline_dummy"]
    report = f"""# Evaluation Report

All numbers below are produced by `python -m src.evaluate` on the held-out test set
({len(y_test)} transactions, {int(y_test.sum())} fraud). Nothing here is hand-written.

## Results (same test set for every model)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {n} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |"
        for n, r in rows.items()) + f"""

## Selected model: `{selected}`
Selected using cross-validated PR-AUC on the training split, **before** looking at the test set.
Test results are reported once and were not used to tune anything.

Confusion matrix (threshold {sel_thr:.3f}): TN={sel['TN']}, FP={sel['FP']}, FN={sel['FN']}, TP={sel['TP']}
(see `reports/confusion_matrix.png`; PR curves in `reports/pr_curve.png`).

## Decision threshold
Rule: flag the top {MAX_REVIEW_RATE:.0%} highest-scoring transactions for manual review (assumed analyst capacity).
The threshold ({sel_thr:.3f}) is the {100 - MAX_REVIEW_RATE * 100:.0f}th percentile of out-of-fold scores on the
TRAINING data, so the test set played no role in choosing it. Risk bands: HIGH >= {sel_thr:.3f} (REVIEW),
MEDIUM >= {sel_med:.3f}, LOW below that (APPROVE).
At this threshold the model catches {sel['recall']:.1%} of fraud with {sel['precision']:.1%} precision.

## False positives vs false negatives
- **False positive:** a legitimate transaction flagged. Costs analyst time and annoys the customer.
- **False negative:** a fraud that is approved. Direct money loss and chargebacks; usually costlier than a false positive, which is why we favour recall within the review capacity.

## Why accuracy is misleading
Fraud is only {y_test.mean():.2%} of transactions. The baseline that approves everything scores
{base['accuracy']:.2%} accuracy yet catches 0 fraud (recall {base['recall']:.0%}). So models are compared on PR-AUC, recall and precision instead.

## Limitations and failure cases
- Data is synthetic with label noise, so scores are moderate and real-world performance will differ.
- `class_weight` shifts scores upward: `fraud_probability` is a risk score, not a calibrated probability.
- Low precision means many false alarms at this threshold; capacity or threshold must be tuned with real costs.
- Fraud patterns outside the training rules (new attack types) will be missed; drift monitoring is needed.
- No customer/device history, so repeat-fraud behaviour cannot be detected.
"""
    open("reports/evaluation_report.md", "w", encoding="utf-8").write(report)
    print("\nSaved: reports/metrics.csv, confusion_matrix.png, pr_curve.png, evaluation_report.md, models/metadata.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()