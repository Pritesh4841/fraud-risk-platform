"""
Train + compare models. Run from project root:  python -m src.train
Selection uses cross-validation on the TRAIN split only. The test set is not touched here.
"""
import json
import time

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, load_and_split

SEED = 42
SCORING = "average_precision"   # PR-AUC: the right metric when fraud is ~2.5% of data


def make_pipeline(model):
    # preprocessing lives INSIDE the pipeline -> re-fitted on the training part of every CV fold
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def main():
    X_train, X_test, y_train, y_test = load_and_split(seed=SEED)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {}

    # 1) Baseline: always predicts the majority class. Sets the "do-nothing" score to beat.
    baseline = make_pipeline(DummyClassifier(strategy="prior"))
    base_score = cross_val_score(baseline, X_train, y_train, cv=cv, scoring=SCORING).mean()
    results["baseline_dummy"] = {"cv_average_precision": float(base_score), "best_params": {}}

    # 2) Logistic Regression: tune C (regularisation strength); class_weight handles imbalance
    logreg = GridSearchCV(
        make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        scoring=SCORING, cv=cv, n_jobs=-1, refit=True,
    )

    # 3) Random Forest: tune max_depth / min_samples_leaf (controls overfitting)
    forest = GridSearchCV(
        make_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        param_grid={"model__max_depth": [6, 10, 16], "model__min_samples_leaf": [5, 20]},
        scoring=SCORING, cv=cv, n_jobs=1, refit=True,
    )

    candidates = {"logistic_regression": logreg, "random_forest": forest}
    fitted = {}
    for name, search in candidates.items():
        t0 = time.time()
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results[name] = {
            "cv_average_precision": float(search.best_score_),
            "best_params": search.best_params_,
        }
        joblib.dump(search.best_estimator_, f"models/{name}.joblib")
        print(f"{name:20s} CV PR-AUC = {search.best_score_:.4f}  params = {search.best_params_}  ({time.time()-t0:.0f}s)")

    print(f"{'baseline_dummy':20s} CV PR-AUC = {base_score:.4f}  (= fraud rate, i.e. random guessing)")

    # Select best by CV PR-AUC (train only), save as the production model
    best_name = max(candidates, key=lambda n: results[n]["cv_average_precision"])
    joblib.dump(fitted[best_name], "models/fraud_model.joblib")
    results["selected_model"] = best_name
    with open("models/training_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSelected: {best_name}  -> saved models/fraud_model.joblib")


if __name__ == "__main__":
    main()