"""
Synthetic fraud dataset generator.
Columns: TransactionID, Amount, TransactionTime, MerchantCategory, CustomerAge,
         PreviousTransactions, FailedAttempts, Location, IsFraud
Usage (from project root): python -m src.generate_data  ->  creates data/transactions.csv
"""
import os

import numpy as np
import pandas as pd

def generate_transactions(n=50_000, fraud_rate=0.02, seed=42):
    rng = np.random.default_rng(seed)

    # --- Base features ---
    start = pd.Timestamp("2025-01-01")
    seconds = rng.integers(0, 365 * 24 * 3600, size=n)
    times = start + pd.to_timedelta(seconds, unit="s")

    df = pd.DataFrame({
        "TransactionID": [f"TXN{100000 + i}" for i in range(n)],
        "Amount": np.round(rng.lognormal(mean=6.0, sigma=1.1, size=n), 2),  # right-skewed (Rs.)
        "TransactionTime": times,
        "MerchantCategory": rng.choice(
            ["Grocery", "Online Shopping", "Travel", "Electronics", "Restaurant", "Fuel", "Jewellery"],
            size=n, p=[0.24, 0.24, 0.08, 0.10, 0.20, 0.10, 0.04]),
        "CustomerAge": np.clip(rng.normal(38, 12, size=n), 18, 85).astype(int),
        "PreviousTransactions": np.where(rng.random(n) < 0.10, rng.poisson(3, n), rng.poisson(35, n)),
        "FailedAttempts": rng.poisson(0.25, size=n),
        "Location": rng.choice(
            ["Ahmedabad", "Mumbai", "Delhi", "Bengaluru", "Pune", "Vadodara", "Surat", "Hyderabad", "Unknown/Foreign"],
            size=n, p=[0.14, 0.18, 0.16, 0.14, 0.10, 0.07, 0.07, 0.10, 0.04]),
    })

    # --- Hidden fraud rule (document this in README) ---
    hour = df["TransactionTime"].dt.hour
    risk = (
        0.9 * hour.between(0, 5).astype(int)                                   # night-time
        + 0.9 * df["MerchantCategory"].isin(["Online Shopping", "Electronics", "Jewellery"]).astype(int)
        + 0.8 * (df["PreviousTransactions"] < 5).astype(int)                   # new / inactive customer
        + 0.8 * df["FailedAttempts"].clip(upper=5)                             # failed attempts
        + 1.0 * (df["Location"] == "Unknown/Foreign").astype(int)
        + 0.5 * (df["Amount"] > 2000).astype(int) + 0.6 * (df["Amount"] > 6000).astype(int)
        + 0.3 * ((df["CustomerAge"] < 22) | (df["CustomerAge"] > 65)).astype(int)
    )
    # find intercept so overall fraud rate ~ fraud_rate
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(mid + risk + rng.normal(0, 0.0, n))))
        lo, hi = (mid, hi) if p.mean() < fraud_rate else (lo, mid)
    noise = rng.normal(0, 0.7, n)  # label noise so model isn't perfect
    prob = 1 / (1 + np.exp(-(mid + risk + noise)))
    df["IsFraud"] = (rng.random(n) < prob).astype(int)

    # --- Realistic mess: missing values, bad values, duplicates ---
    for col, frac in [("CustomerAge", 0.01), ("Location", 0.005), ("Amount", 0.003)]:
        idx = rng.choice(n, size=int(frac * n), replace=False)
        df[col] = df[col].astype("object") if col == "Location" else df[col].astype(float)
        df.loc[idx, col] = np.nan
    bad = rng.choice(n, size=10, replace=False)
    df.loc[bad, "Amount"] = -1 * df.loc[bad, "Amount"].fillna(100)            # negative amounts (invalid)
    df = pd.concat([df, df.sample(60, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_transactions()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print("Saved: data/transactions.csv")
    print("Shape:", df.shape)
    print("\nFraud %:\n", (df["IsFraud"].value_counts(normalize=True) * 100).round(2))
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSample:\n", df.head())
