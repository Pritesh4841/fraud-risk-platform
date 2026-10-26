"""
Sanity check: how good could ANY model be on this synthetic data?
Run from project root:  python -m src.oracle_check

The generator (src/generate_data.py) creates fraud labels from a hidden risk rule plus random noise.
Scoring every transaction with that exact rule ("oracle") gives a rough ceiling for ROC-AUC / PR-AUC.
If the trained model is near this ceiling, the modest metrics come from the data's label noise
(fraud is a random draw from a small probability), not from a weak model.
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def oracle_score(df: pd.DataFrame) -> pd.Series:
    hour = df["TransactionTime"].dt.hour
    amount = df["Amount"].where(df["Amount"] >= 0).fillna(df["Amount"].median())
    age = df["CustomerAge"].fillna(df["CustomerAge"].median())
    return (
        0.9 * hour.between(0, 5)
        + 0.9 * df["MerchantCategory"].isin(["Online Shopping", "Electronics", "Jewellery"])
        + 0.8 * (df["PreviousTransactions"] < 5)
        + 0.8 * df["FailedAttempts"].clip(upper=5)
        + 1.0 * (df["Location"] == "Unknown/Foreign")
        + 0.5 * (amount > 2000) + 0.6 * (amount > 6000)
        + 0.3 * ((age < 22) | (age > 65))
    ).astype(float)


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv", parse_dates=["TransactionTime"]).drop_duplicates("TransactionID")
    s = oracle_score(df)
    print(f"Fraud rate: {df['IsFraud'].mean():.2%}")
    print(f"Oracle ceiling (hidden rule, whole dataset): ROC-AUC = {roc_auc_score(df['IsFraud'], s):.3f} | "
          f"PR-AUC = {average_precision_score(df['IsFraud'], s):.3f}")
    print("Compare with the trained model's test metrics in reports/metrics.csv")
