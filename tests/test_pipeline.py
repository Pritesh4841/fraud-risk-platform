"""
Data / preprocessing tests (no web framework needed).  Run from the project root:   pytest -v
They check the things the assignment cares about: duplicate handling, correct train/test split,
and that preprocessing learns ONLY from the training data (no leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (BINARY_FEATURES, CATEGORICAL_FEATURES, ID_COL, NUMERIC_FEATURES, TARGET,
                               build_features, build_preprocessor, clean_raw, load_and_split)

DATA = str(Path(__file__).resolve().parents[1] / "data" / "transactions.csv")


def test_clean_raw_removes_duplicate_transaction_ids():
    raw = pd.read_csv(DATA, parse_dates=["TransactionTime"])
    assert raw[ID_COL].duplicated().sum() > 0            # the dataset really contains duplicates
    cleaned = clean_raw(raw)
    assert cleaned[ID_COL].is_unique
    assert len(cleaned) == raw[ID_COL].nunique()


def test_split_is_stratified_and_train_test_do_not_overlap():
    X_train, X_test, y_train, y_test = load_and_split(path=DATA)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.002   # same fraud rate in both parts
    assert TARGET not in X_train.columns and ID_COL not in X_train.columns


def test_target_and_id_are_never_model_features():
    used = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    assert TARGET not in used and ID_COL not in used


def test_build_features_turns_invalid_amounts_into_nan_and_returns_expected_columns():
    df = pd.DataFrame({
        "Amount": [-5.0, "abc", np.nan, 100.0],
        "TransactionTime": pd.to_datetime(["2025-01-01 02:00"] * 4),
        "MerchantCategory": ["Fuel"] * 4,
        "CustomerAge": [30] * 4,
        "PreviousTransactions": [10] * 4,
        "FailedAttempts": [0] * 4,
        "Location": ["Mumbai"] * 4,
    })
    X = build_features(df)
    assert list(X.columns) == NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    assert X["LogAmount"].isna().tolist() == [True, True, True, False]


def test_hour_from_timestamp_and_hour_field_give_identical_features():
    """Training uses TransactionTime, the API sends Hour - both must produce the same features."""
    base = {"Amount": [250.0], "MerchantCategory": ["Electronics"], "CustomerAge": [25],
            "PreviousTransactions": [2], "FailedAttempts": [1], "Location": ["Delhi"]}
    from_time = build_features(pd.DataFrame({**base, "TransactionTime": pd.to_datetime(["2025-06-01 03:30"])}))
    from_hour = build_features(pd.DataFrame({**base, "Hour": [3]}))
    pd.testing.assert_frame_equal(from_time, from_hour)
    assert from_hour["IsNight"].iloc[0] == 1


def test_preprocessor_learns_only_from_training_data():
    X_train, X_test, _, _ = load_and_split(path=DATA)
    prep = build_preprocessor().fit(X_train)
    imputer = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    expected = build_features(X_train)[NUMERIC_FEATURES].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)     # medians come from TRAIN rows only
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    assert scaler.n_samples_seen_ == len(X_train)         # scaler never saw the test rows

    Xt = prep.transform(X_test)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    assert not np.isnan(dense).any()                      # test NaNs are imputed with train medians


def test_unseen_category_does_not_break_the_pipeline():
    X_train, _, _, _ = load_and_split(path=DATA)
    prep = build_preprocessor().fit(X_train)
    row = X_train.head(1).copy()
    row["Location"] = "Atlantis"                          # city never seen in training
    row["CustomerAge"] = np.nan
    out = prep.transform(row)
    assert out.shape[0] == 1


if __name__ == "__main__":      # lets the file also run without pytest:  python -m tests.test_pipeline
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
