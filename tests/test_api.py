"""
API tests.  Run from the project root:   pytest -v

The model is loaded by the app's lifespan hook, so the client must be used as a context manager
(`with TestClient(app) as client`) - otherwise startup never runs and every call returns 503.
"""
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

# Example request from the assignment (customer_age_days is accepted but unused, see README).
SAMPLE = {
    "amount": 249.50,
    "merchant_category": "online",
    "transaction_hour": 2,
    "customer_age_days": 180,
    "previous_declines_24h": 3,
}

RISKY = {"amount": 9000.0, "merchant_category": "Electronics", "transaction_hour": 3,
         "customer_age": 20, "previous_transactions": 1, "failed_attempts": 4,
         "location": "Unknown/Foreign"}
SAFE = {"amount": 300.0, "merchant_category": "Grocery", "transaction_hour": 14,
        "customer_age": 40, "previous_transactions": 40, "failed_attempts": 0,
        "location": "Mumbai"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def assert_valid_prediction(body):
    assert set(body) == {"fraud_probability", "risk_band", "decision"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["decision"] in {"APPROVE", "REVIEW"}
    # decision rule: only HIGH risk goes to manual review
    assert (body["decision"] == "REVIEW") == (body["risk_band"] == "HIGH")


# ------------------------------------------------------------------ /health
def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["thresholds"]["HIGH"] > body["thresholds"]["MEDIUM"] > 0


def test_health_when_model_missing_is_degraded_not_crashing(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "model", None)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["model_loaded"] is False


# ------------------------------------------------------------------ /predict: valid input
def test_predict_assignment_sample_request(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_predict_full_payload_and_optional_fields_missing(client):
    assert client.post("/predict", json=RISKY).status_code == 200
    minimal = {"amount": 100, "merchant_category": "Fuel", "transaction_hour": 12}
    r = client.post("/predict", json=minimal)          # amount as JSON int + all optionals missing
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_category_and_location_are_case_insensitive(client):
    a = client.post("/predict", json={**SAFE, "merchant_category": "grocery", "location": "mumbai"}).json()
    b = client.post("/predict", json=SAFE).json()
    assert a == b


def test_prediction_comes_from_the_model_risky_scores_above_safe(client):
    risky = client.post("/predict", json=RISKY).json()
    safe = client.post("/predict", json=SAFE).json()
    assert risky["fraud_probability"] > safe["fraud_probability"]
    assert risky["risk_band"] == "HIGH" and risky["decision"] == "REVIEW"
    assert safe["risk_band"] == "LOW" and safe["decision"] == "APPROVE"


def test_unknown_extra_fields_are_ignored(client):
    r = client.post("/predict", json={**SAFE, "something_else": 1})
    assert r.status_code == 200


# ------------------------------------------------------------------ /predict: invalid input
@pytest.mark.parametrize("missing", ["amount", "merchant_category", "transaction_hour"])
def test_missing_required_field_returns_422(client, missing):
    payload = {k: v for k, v in SAMPLE.items() if k != missing}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert any(missing in err["loc"] for err in r.json()["detail"])


@pytest.mark.parametrize("field,bad_value", [
    ("amount", "abc"),
    ("amount", "249.5"),              # numeric string rejected (strict types)
    ("amount", None),
    ("merchant_category", 123),
    ("transaction_hour", "2"),
    ("transaction_hour", 2.5),
    ("customer_age", "forty"),
    ("failed_attempts", "3"),
    ("customer_age_days", 1.5),
])
def test_wrong_data_type_returns_422(client, field, bad_value):
    r = client.post("/predict", json={**SAMPLE, field: bad_value})
    assert r.status_code == 422


@pytest.mark.parametrize("field,bad_value", [
    ("transaction_hour", 24),
    ("transaction_hour", -1),
    ("amount", -5),
    ("customer_age", 10),
    ("failed_attempts", -1),
    ("merchant_category", "Spaceships"),
])
def test_out_of_range_or_unknown_value_returns_422(client, field, bad_value):
    r = client.post("/predict", json={**SAMPLE, field: bad_value})
    assert r.status_code == 422


def test_malformed_json_and_empty_body_return_422(client):
    r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert client.post("/predict").status_code == 422


def test_wrong_http_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_model_not_loaded_returns_503(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "model", None)
    assert client.post("/predict", json=SAMPLE).status_code == 503
    assert client.post("/predict-batch", json={"transactions": [SAMPLE]}).status_code == 503


# ------------------------------------------------------------------ /predict-batch
def test_batch_predictions_match_single_predictions(client):
    txns = [SAMPLE, RISKY, SAFE]
    r = client.post("/predict-batch", json={"transactions": txns})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and len(body["predictions"]) == 3
    for t, p in zip(txns, body["predictions"]):
        assert_valid_prediction(p)
        assert p == client.post("/predict", json=t).json()      # same order, same preprocessing


@pytest.mark.parametrize("payload", [
    {"transactions": []},             # empty batch
    {},                               # key missing
    {"transactions": "nope"},         # wrong type
    {"transactions": {"amount": 1}},  # object instead of list
    [],                               # bare list instead of object
])
def test_empty_or_malformed_batch_returns_422(client, payload):
    assert client.post("/predict-batch", json=payload).status_code == 422


def test_batch_with_one_invalid_item_is_rejected(client):
    bad = {**SAMPLE, "transaction_hour": 99}
    r = client.post("/predict-batch", json={"transactions": [SAMPLE, bad]})
    assert r.status_code == 422
    assert any(1 in err["loc"] for err in r.json()["detail"])    # error points at item index 1


def test_batch_over_limit_returns_422(client):
    r = client.post("/predict-batch", json={"transactions": [SAMPLE] * 1001})
    assert r.status_code == 422


# ------------------------------------------------------------------ restart
def test_api_gives_identical_results_after_restart():
    """Two separate app startups (model re-loaded from disk each time) must agree."""
    with TestClient(app) as first:
        before = first.post("/predict", json=RISKY).json()
        batch_before = first.post("/predict-batch", json={"transactions": [SAMPLE, SAFE]}).json()
    with TestClient(app) as second:                    # simulates stopping + starting the server
        assert second.get("/health").json()["model_loaded"] is True
        assert second.post("/predict", json=RISKY).json() == before
        assert second.post("/predict-batch", json={"transactions": [SAMPLE, SAFE]}).json() == batch_before
