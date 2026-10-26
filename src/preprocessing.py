"""
Reusable preprocessing for the fraud model.
Everything that LEARNS from data (imputer medians, scaler mean/std, one-hot categories)
lives inside an sklearn Pipeline, so it is fitted on TRAIN data only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "IsFraud"
ID_COL = "TransactionID"

NUMERIC_FEATURES = ["LogAmount", "CustomerAge", "PreviousTransactions", "FailedAttempts"]
BINARY_FEATURES = ["IsNight", "IsNewCustomer"]
CATEGORICAL_FEATURES = ["MerchantCategory", "Location"]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning, safe to run BEFORE the split (learns nothing from data)."""
    df = df.drop_duplicates(subset=ID_COL).copy()   # same row must not land in both train and test
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering. Used in training AND in the API, so both stay consistent."""
    X = df.copy()

    # invalid values -> NaN (imputed later, inside the pipeline)
    X["Amount"] = pd.to_numeric(X["Amount"], errors="coerce")
    X["Amount"] = X["Amount"].where(X["Amount"] >= 0)

    # hour: from TransactionTime if given, else use Hour directly (API can send either)
    if "Hour" not in X.columns:
        X["Hour"] = pd.to_datetime(X["TransactionTime"]).dt.hour

    # derived features
    X["LogAmount"] = np.log1p(X["Amount"])                                  # fixes right-skew
    X["IsNight"] = X["Hour"].between(0, 5).astype(int)                       # 00:00-05:59
    X["IsNewCustomer"] = (X["PreviousTransactions"] < 5).astype(int)         # little history

    return X[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> Pipeline:
    """Unfitted preprocessing pipeline. Call .fit() on TRAIN data only."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    column_transformer = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(build_features, validate=False)),
        ("columns", column_transformer),
    ])


def load_and_split(path="data/transactions.csv", test_size=0.2, seed=42):
    """Clean -> stratified train/test split. Returns raw (unprocessed) X/y splits."""
    df = clean_raw(pd.read_csv(path, parse_dates=["TransactionTime"]))
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print("Train:", X_train.shape, "fraud %:", round(y_train.mean() * 100, 2))
    print("Test :", X_test.shape, "fraud %:", round(y_test.mean() * 100, 2))

    prep = build_preprocessor()
    Xt_train = prep.fit_transform(X_train)   # fit on TRAIN only
    Xt_test = prep.transform(X_test)         # only transform on TEST
    print("Processed train:", Xt_train.shape, "| processed test:", Xt_test.shape)
    print("NaNs left in train:", int(np.isnan(Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train).sum()))