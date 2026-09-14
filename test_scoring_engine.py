"""Unit tests for scoring_engine.py — pure logic, no database required."""

import pytest

from scoring_engine import RiskScoringEngine, add_to_blocklist, is_on_blocklist


BASE_FEATURES = {
    "amount_to_average_ratio": 1.0,
    "velocity_1h": 0,
    "velocity_24h": 1,
    "new_beneficiary_flag": 0,
    "off_hours_flag": 0,
    "kyc_limit_exceeded": 0,
    "location_deviation_km": 0.0,
    "sim_swap_72h": 0,
    "new_device_flag": 0,
    "smishing_correlation_30m": 0,
    "agent_complaint_rate": 0.0,
}


def test_no_triggers_scores_zero_and_low_risk():
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    result = engine.score(dict(BASE_FEATURES))
    assert result.risk_score == 0
    assert result.risk_level == "Low"
    assert result.reason_codes == []


def test_sim_swap_alone_scores_25_and_stays_low_risk():
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    features = {**BASE_FEATURES, "sim_swap_72h": 1}
    result = engine.score(features)
    assert result.risk_score == 25
    assert result.risk_level == "Low"  # below the 30-point Medium threshold
    assert any("SIM swap" in rc for rc in result.reason_codes)


def test_multiple_triggers_push_into_high_risk_tier():
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    features = {
        **BASE_FEATURES,
        "sim_swap_72h": 1,              # +25
        "new_device_flag": 1,
        "new_beneficiary_flag": 1,      # +20 (combo)
        "amount_to_average_ratio": 4.0,  # +15
    }
    result = engine.score(features)
    assert result.risk_score == 60
    assert result.risk_level == "High"


def test_score_never_exceeds_100():
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    features = {
        **BASE_FEATURES,
        "sim_swap_72h": 1,
        "new_device_flag": 1,
        "new_beneficiary_flag": 1,
        "amount_to_average_ratio": 10.0,
        "velocity_24h": 20,
        "smishing_correlation_30m": 1,
        "location_deviation_km": 999,
        "off_hours_flag": 1,
        "agent_complaint_rate": 0.9,
        "kyc_limit_exceeded": 1,
    }
    result = engine.score(features, receiver_msisdn="blocked-msisdn-test")
    assert result.risk_score <= 100


def test_ml_blend_moves_score_toward_model_probability():
    engine = RiskScoringEngine(ml_blend_weight=0.5)
    no_ml = engine.score(dict(BASE_FEATURES))
    with_high_ml = engine.score(dict(BASE_FEATURES), ml_fraud_probability=0.9)
    assert with_high_ml.risk_score > no_ml.risk_score


def test_receiver_on_blocklist_adds_reason_code():
    add_to_blocklist("test-blocklisted-receiver")
    assert is_on_blocklist("test-blocklisted-receiver")
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    result = engine.score(dict(BASE_FEATURES), receiver_msisdn="test-blocklisted-receiver")
    assert any("blocklist" in rc.lower() for rc in result.reason_codes)


def test_unknown_receiver_not_on_blocklist():
    engine = RiskScoringEngine(ml_blend_weight=0.0)
    result = engine.score(dict(BASE_FEATURES), receiver_msisdn="never-added-to-list")
    assert not any("blocklist" in rc.lower() for rc in result.reason_codes)


@pytest.mark.parametrize("score,expected_level", [(0, "Low"), (29, "Low"), (30, "Medium"), (59, "Medium"), (60, "High"), (84, "High"), (85, "Critical"), (100, "Critical")])
def test_risk_tier_boundaries(score, expected_level):
    from scoring_engine import _risk_tier
    assert _risk_tier(score) == expected_level
