# Fraud Detection and Risk Scoring Platform

An end-to-end machine-learning project for generating synthetic transaction data, exploring it, training fraud classifiers, evaluating risk thresholds, and serving predictions through a REST API.

## Project status

The core project tasks are implemented:

- [x] Generate a reproducible synthetic transaction dataset with fraud labels and intentional data-quality issues.
- [x] Explore the dataset in `notebooks/eda.ipynb`.
- [x] Clean duplicate transaction IDs and split data using stratification.
- [x] Build preprocessing and feature engineering into a scikit-learn pipeline to avoid train/test leakage.
- [x] Compare a baseline, Logistic Regression, and Random Forest using cross-validated PR-AUC.
- [x] Evaluate the selected model on a held-out test set and save metrics, plots, and risk thresholds.
- [x] Provide health, single-prediction, and batch-prediction API endpoints.
- [x] Add tests for data processing, model/API behavior, validation, and restart consistency.

The checked-in model artifacts and reports are ready to use; retraining is optional.

## Results

The selected model is Logistic Regression. On the held-out test set of 10,000 transactions (252 fraud cases), its results are:

| Metric | Result |
|---|---:|
| Accuracy | 0.8955 |
| Precision | 0.0933 |
| Recall | 0.3611 |
| F1 | 0.1483 |
| ROC-AUC | 0.7357 |
| PR-AUC | 0.0969 |

The high-risk review threshold is 0.6692 and the medium-risk threshold is 0.4898. A score at or above the high threshold is marked for review. Accuracy alone is misleading for this imbalanced dataset: a baseline that approves every transaction achieves 0.9748 accuracy while detecting no fraud.

See [reports/evaluation_report.md](reports/evaluation_report.md) for methodology, additional results, and limitations.

## Requirements

- Python 3.14 (the pinned dependencies and included model files were prepared for this version).
- `pip`.

The scikit-learn version is pinned because the serialized `.joblib` model files depend on the version used to create them. If you use another Python or scikit-learn version and encounter compatibility issues, recreate the models as described below.

## Setup

Run commands from the project root directory.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python directly (for example, `.\venv\Scripts\python.exe -m pip install -r requirements.txt`) or follow your organization's PowerShell execution policy.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


### Regenerate data, train, and evaluate (optional)

These commands rebuild the outputs and overwrite the corresponding files in `data/`, `models/`, and `reports/`:

```bash
python -m src.generate_data
python -m src.train # It Takes time like 2-5 minutes
python -m src.evaluate
```

The data generator creates 50,000 synthetic transactions and adds 60 duplicate rows; preprocessing removes duplicate transaction IDs before splitting. Training compares the candidate models, selects the best by training-only cross-validated PR-AUC, and writes the selected pipeline to `models/fraud_model.joblib`. Evaluation writes test metrics, visualizations, a report, and API threshold metadata.

To run the synthetic-data oracle comparison:

```bash
python -m src.oracle_check
```

### Run the tests

```bash
pytest -v
```

## Run the project

### Start the API using the included artifacts

After installing dependencies, start the service:

```bash
uvicorn app:app --reload
```

The API listens at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation. The model and its thresholds are loaded from `models/` at startup.


## API usage

### `GET /health`

Reports whether the model loaded, the selected model name, and the configured thresholds. A failed model load is reported as a degraded status; prediction requests return HTTP 503 while the model is unavailable.

### `POST /predict`

Example request:

```json
{
  "amount": 249.5,
  "merchant_category": "online",
  "transaction_hour": 2,
  "previous_declines_24h": 3
}
```

Example response shape:

```json
{
  "fraud_probability": 0.9593,
  "risk_band": "HIGH",
  "decision": "REVIEW"
}
```

The numeric score is a model risk score and is not guaranteed to be a calibrated real-world probability. Exact scores can change if the model is retrained.

Required fields are `amount` (0 to 10,000,000), `merchant_category`, and `transaction_hour` (0 to 23). Optional fields are `customer_age` (18 to 100), `previous_transactions`, `failed_attempts` (also accepted as `previous_declines_24h`), `location`, and `customer_age_days`. Merchant categories are checked against the supported categories without case sensitivity; `online` is an alias for `Online Shopping`. `customer_age_days` is accepted for compatibility with the sample request but is not used by the model because there is no corresponding dataset feature. Invalid request data returns HTTP 422.

### `POST /predict-batch`

Accepts a JSON object with a `transactions` array of 1 to 1,000 prediction requests. Returns a `count` and a `predictions` array in the same order as the input.

## Dataset and implementation

The dataset is synthetic and is stored in `data/transactions.csv`. Its columns include transaction amount and time, merchant category, customer age and history, failed attempts, location, and the `IsFraud` target. The generated labels are based on a probabilistic risk rule with noise; this is not a substitute for a real, representative fraud dataset.

The preprocessing pipeline handles invalid amounts, missing values, numeric scaling, categorical encoding, and engineered features (`LogAmount`, `IsNight`, and `IsNewCustomer`). Imputation and scaling are fitted within the training pipeline rather than on the held-out test set. The same saved pipeline is used by the API.

## Repository layout

```text
app.py                       FastAPI application
requirements.txt             Pinned Python dependencies
data/transactions.csv        Synthetic transaction dataset
notebooks/eda.ipynb           Exploratory data analysis
src/generate_data.py          Synthetic data generator
src/preprocessing.py          Cleaning, feature engineering, splitting, preprocessing
src/train.py                  Model comparison, tuning, and selection
src/evaluate.py               Held-out evaluation, thresholds, plots, report
src/oracle_check.py           Synthetic-label-rule comparison
models/                       Saved models and training/API metadata
reports/                      Metrics, evaluation report, and plots
tests/                        API and data/model pipeline tests
```

## Limitations

- The data is synthetic and contains noisy labels, so its metrics do not predict production performance.
- Fraud is rare; the current operating threshold prioritizes review capacity and still produces false positives.
- The returned score is not a calibrated probability.
- The model has no customer or device history and may not identify new fraud patterns or data drift.
- Thresholds and model performance should be recalibrated and validated with representative, appropriately governed production data before operational use.
