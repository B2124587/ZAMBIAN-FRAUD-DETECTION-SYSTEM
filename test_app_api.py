"""
API tests that don't touch the database — auth failures and validation
errors short-circuit before any DB call, so these are safe to run in CI
without a live MySQL instance. Full /score happy-path tests belong in an
integration suite that runs against a real (test) database.
"""

from fastapi.testclient import TestClient

import app

client = TestClient(app.app)

VALID_KEY = "ci-test-key-123"  # matches FRAUD_API_KEYS set in conftest.py


def test_health_returns_200_even_without_a_live_db():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("healthy", "degraded")


def test_score_without_api_key_is_rejected():
    resp = client.post(
        "/score",
        json={"sender_msisdn": "x", "receiver_msisdn": "y", "amount": 100, "txn_type": "P2P"},
    )
    assert resp.status_code == 401


def test_score_with_wrong_api_key_is_rejected():
    resp = client.post(
        "/score",
        json={"sender_msisdn": "x", "receiver_msisdn": "y", "amount": 100, "txn_type": "P2P"},
        headers={"X-API-Key": "not-the-right-key"},
    )
    assert resp.status_code == 401


def test_score_rejects_negative_amount():
    resp = client.post(
        "/score",
        json={"sender_msisdn": "x", "receiver_msisdn": "y", "amount": -5, "txn_type": "P2P"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert resp.status_code == 422


def test_score_rejects_missing_required_field():
    resp = client.post(
        "/score",
        json={"amount": 100, "txn_type": "P2P"},  # missing sender/receiver
        headers={"X-API-Key": VALID_KEY},
    )
    assert resp.status_code == 422


def test_audit_recent_requires_api_key():
    resp = client.get("/audit/recent")
    assert resp.status_code == 401


def test_audit_recent_rejects_limit_over_100():
    resp = client.get("/audit/recent?limit=500", headers={"X-API-Key": VALID_KEY})
    assert resp.status_code == 422


def test_score_degraded_mode_when_models_unavailable():
    # app.py's models are currently None (not loaded by main.py)
    resp = client.post(
        "/score",
        json={"sender_msisdn": "x", "receiver_msisdn": "y", "amount": 100, "txn_type": "P2P"},
        headers={"X-API-Key": VALID_KEY},
    )
    # The scoring engine should run rules and the request should succeed, but degraded_mode=True
    assert resp.status_code == 200
    data = resp.json()
    assert data["degraded_mode"] is True
    assert data["ml_model_used"] == "rule_only"


def test_smishing_classify_without_api_key_is_rejected():
    resp = client.post("/smishing/classify", json={"sms_text": "Free money!"})
    assert resp.status_code == 401


def test_register_without_api_key_is_rejected():
    resp = client.post(
        "/auth/register",
        json={"username": "testuser", "password": "password123", "role": "analyst", "full_name": "Test User"}
    )
    assert resp.status_code == 401


def test_register_invalid_role():
    resp = client.post(
        "/auth/register",
        json={"username": "testuser", "password": "password123", "role": "invalid_role", "full_name": "Test User"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert resp.status_code == 422
    assert "Invalid role" in resp.json()["detail"]
