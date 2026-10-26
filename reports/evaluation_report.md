# Evaluation Report

All numbers below are produced by `python -m src.evaluate` on the held-out test set
(10000 transactions, 252 fraud). Nothing here is hand-written.

## Results (same test set for every model)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8955 | 0.0933 | 0.3611 | 0.1483 | 0.7357 | 0.0969 |
| random_forest | 0.8773 | 0.0779 | 0.3571 | 0.1279 | 0.7181 | 0.0821 |
| baseline_dummy | 0.9748 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0252 |

## Selected model: `logistic_regression`
Selected using cross-validated PR-AUC on the training split, **before** looking at the test set.
Test results are reported once and were not used to tune anything.

Confusion matrix (threshold 0.669): TN=8864, FP=884, FN=161, TP=91
(see `reports/confusion_matrix.png`; PR curves in `reports/pr_curve.png`).

## Decision threshold
Rule: flag the top 10% highest-scoring transactions for manual review (assumed analyst capacity).
The threshold (0.669) is the 90th percentile of out-of-fold scores on the
TRAINING data, so the test set played no role in choosing it. Risk bands: HIGH >= 0.669 (REVIEW),
MEDIUM >= 0.490, LOW below that (APPROVE).
At this threshold the model catches 36.1% of fraud with 9.3% precision.

## False positives vs false negatives
- **False positive:** a legitimate transaction flagged. Costs analyst time and annoys the customer.
- **False negative:** a fraud that is approved. Direct money loss and chargebacks; usually costlier than a false positive, which is why we favour recall within the review capacity.

## Why accuracy is misleading
Fraud is only 2.52% of transactions. The baseline that approves everything scores
97.48% accuracy yet catches 0 fraud (recall 0%). So models are compared on PR-AUC, recall and precision instead.

## Limitations and failure cases
- Data is synthetic with label noise, so scores are moderate and real-world performance will differ.
- `class_weight` shifts scores upward: `fraud_probability` is a risk score, not a calibrated probability.
- Low precision means many false alarms at this threshold; capacity or threshold must be tuned with real costs.
- Fraud patterns outside the training rules (new attack types) will be missed; drift monitoring is needed.
- No customer/device history, so repeat-fraud behaviour cannot be detected.
