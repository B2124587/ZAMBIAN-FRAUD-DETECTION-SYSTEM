"""
MODULE 4: Risk Scoring Engine
Mobile Money Fraud Detection System - Zambia
scoring_engine.py

Implements the weighted composite risk scoring framework (max 100 points).
Outputs risk_score, risk_level, reason_codes, and an optional ML probability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from alerts import determine_action


# ─────────────────────────────────────────────
# SCORING WEIGHTS (Table 1)
# ─────────────────────────────────────────────

SCORE_WEIGHTS = {
    "sim_swap_72h":              25,   # SIM swap in past 72 h
    "new_device_and_beneficiary": 20,  # New device + new beneficiary combo
    "amount_anomaly_3x":          15,  # Amount > 3× 30-day average
    "velocity_anomaly_5_24h":     10,  # > 5 transactions in 24 h
    "smishing_correlation_30m":   10,  # Preceded by smishing signal ≤ 30 min
    "location_deviation_200km":   10,  # Location deviation > 200 km
    "receiver_on_blocklist":       5,  # Receiver on fraud blocklist
    "off_hours_flag":              3,  # Off-hours activity
    "agent_complaint_elevated":    2,  # Agent complaint rate elevated (> 0.05)
}

MAX_SCORE = 100


# ─────────────────────────────────────────────
# TIER MAPPING
# ─────────────────────────────────────────────

def _risk_tier(score: float) -> str:
    """Map a composite score to the four-tier classification (Table 3.1)."""
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    elif score < 85:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ScoringResult:
    risk_score: float
    risk_level: str
    reason_codes: list[str]
    fraud_probability: float = 0.0
    feature_contributions: dict[str, float] = field(default_factory=dict)
    recommended_action: str = "ALLOW"

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "reason_codes": self.reason_codes,
            "fraud_probability": self.fraud_probability,
            "feature_contributions": self.feature_contributions,
            "recommended_action": self.recommended_action,
        }


# ─────────────────────────────────────────────
# BLOCKLIST — DB-backed with TTL cache (FR-06, Gap #6)
# Reads from reference_data WHERE ref_type='blocklist' AND is_active=1.
# A 60-second in-process TTL cache avoids a DB round-trip on every /score call.
# Admin operations use add_to_blocklist_db() / remove_from_blocklist_db().
# ─────────────────────────────────────────────

_blocklist_cache: set[str] = set()
_blocklist_loaded_at: float = 0.0
_BLOCKLIST_TTL_SECONDS: float = 60.0


def _refresh_blocklist_if_stale() -> None:
    """Reload the blocklist from the DB if the TTL has expired."""
    global _blocklist_cache, _blocklist_loaded_at
    now = time.monotonic()
    if now - _blocklist_loaded_at < _BLOCKLIST_TTL_SECONDS:
        return
    try:
        from db_config import ReferenceData, get_engine
        from sqlalchemy.orm import Session
        with Session(get_engine()) as s:
            rows = (
                s.query(ReferenceData)
                .filter_by(ref_type="blocklist", is_active=True)
                .all()
            )
            _blocklist_cache = {r.ref_key for r in rows}
            _blocklist_loaded_at = now
    except Exception:
        # DB unavailable — keep existing cache rather than crashing
        pass


def add_to_blocklist_db(hashed_msisdn: str, added_by: str = "system") -> None:
    """Persist a new blocklist entry to reference_data and invalidate the local cache."""
    global _blocklist_cache, _blocklist_loaded_at
    try:
        from db_config import ReferenceData, get_engine
        from sqlalchemy.orm import Session
        import uuid
        import datetime
        with Session(get_engine()) as s:
            existing = s.query(ReferenceData).filter_by(
                ref_type="blocklist", ref_key=hashed_msisdn, is_active=True
            ).first()
            if not existing:
                s.add(ReferenceData(
                    ref_type="blocklist",
                    ref_key=hashed_msisdn,
                    ref_value="1",
                    description="Fraud blocklist entry",
                    is_active=True,
                    updated_by=added_by,
                    updated_at=datetime.datetime.utcnow(),
                ))
                s.commit()
        _blocklist_loaded_at = 0.0  # Force cache refresh on next check
    except Exception:
        # DB unavailable — add to in-memory cache as fallback
        _blocklist_cache.add(hashed_msisdn)


def remove_from_blocklist_db(hashed_msisdn: str, removed_by: str = "system") -> bool:
    """Soft-delete a blocklist entry (set is_active=False) and invalidate the cache."""
    global _blocklist_cache, _blocklist_loaded_at
    try:
        from db_config import ReferenceData, get_engine
        from sqlalchemy.orm import Session
        import datetime
        with Session(get_engine()) as s:
            row = s.query(ReferenceData).filter_by(
                ref_type="blocklist", ref_key=hashed_msisdn, is_active=True
            ).first()
            if not row:
                return False
            row.is_active = False
            row.updated_by = removed_by
            row.updated_at = datetime.datetime.utcnow()
            s.commit()
        _blocklist_loaded_at = 0.0
        return True
    except Exception:
        # DB unavailable — remove from in-memory cache as fallback
        if hashed_msisdn in _blocklist_cache:
            _blocklist_cache.remove(hashed_msisdn)
            return True
        return False


def add_to_blocklist(hashed_msisdn: str) -> None:
    """Backwards-compatible shim — delegates to the DB-backed version."""
    add_to_blocklist_db(hashed_msisdn)


def is_on_blocklist(hashed_msisdn: str) -> bool:
    _refresh_blocklist_if_stale()
    return hashed_msisdn in _blocklist_cache


# ─────────────────────────────────────────────
# RISK SCORING ENGINE
# ─────────────────────────────────────────────

class RiskScoringEngine:
    """
    Evaluates an engineered feature vector against the 9 weighted risk rules
    and returns a ScoringResult.

    Optionally accepts an ML model probability to blend with rule score.
    """

    def __init__(self, ml_blend_weight: float = 0.40):
        """
        ml_blend_weight: proportion of final score derived from ML model (0–1).
        The remaining proportion comes from rule-based scoring.
        Report formula: 0.60 × rule_score + 0.40 × ml_score (NFR-03).
        Set to 0.0 for pure rule-based scoring.
        """
        self.ml_blend_weight = ml_blend_weight

    def score(
        self,
        features: dict[str, Any],
        receiver_msisdn: str | None = None,
        ml_fraud_probability: float | None = None,
    ) -> ScoringResult:
        """
        Parameters
        ----------
        features : dict produced by FeatureEngineer.engineer()
        receiver_msisdn : hashed receiver MSISDN for blocklist check
        ml_fraud_probability : optional float [0, 1] from ML model
        """
        score = 0.0
        reasons: list[str] = []
        contributions: dict[str, float] = {}

        # ── RULE 1: SIM swap in past 72h (weight 25) ──
        if features.get("sim_swap_72h", 0):
            score += SCORE_WEIGHTS["sim_swap_72h"]
            reasons.append("R01: SIM swap detected within 72 hours")
            contributions["sim_swap_72h"] = SCORE_WEIGHTS["sim_swap_72h"]

        # ── RULE 2: New device AND new beneficiary combo (weight 20) ──
        if features.get("new_device_flag", 0) and features.get("new_beneficiary_flag", 0):
            score += SCORE_WEIGHTS["new_device_and_beneficiary"]
            reasons.append("R02: Unrecognised device sending to new beneficiary")
            contributions["new_device_and_beneficiary"] = SCORE_WEIGHTS["new_device_and_beneficiary"]

        # ── RULE 3: Amount > 3× 30-day average (weight 15) ──
        if features.get("amount_to_average_ratio", 0) > 3.0:
            score += SCORE_WEIGHTS["amount_anomaly_3x"]
            reasons.append(
                f"R03: Amount {features['amount_to_average_ratio']:.1f}× above 30-day average"
            )
            contributions["amount_anomaly_3x"] = SCORE_WEIGHTS["amount_anomaly_3x"]

        # ── RULE 4: Velocity > 5 transactions / 24h (weight 10) ──
        if features.get("velocity_24h", 0) > 5:
            score += SCORE_WEIGHTS["velocity_anomaly_5_24h"]
            reasons.append(
                f"R04: High transaction velocity ({int(features['velocity_24h'])} txns/24h)"
            )
            contributions["velocity_anomaly_5_24h"] = SCORE_WEIGHTS["velocity_anomaly_5_24h"]

        # ── RULE 5: Smishing signal within 30 min (weight 10) ──
        if features.get("smishing_correlation_30m", 0):
            score += SCORE_WEIGHTS["smishing_correlation_30m"]
            reasons.append("R05: Smishing/vishing signal correlated within 30 minutes")
            contributions["smishing_correlation_30m"] = SCORE_WEIGHTS["smishing_correlation_30m"]

        # ── RULE 6: Location deviation > 200 km (weight 10) ──
        if features.get("location_deviation_km", 0) > 200:
            score += SCORE_WEIGHTS["location_deviation_200km"]
            reasons.append(
                f"R06: Geographic anomaly ({features['location_deviation_km']:.0f} km from usual area)"
            )
            contributions["location_deviation_200km"] = SCORE_WEIGHTS["location_deviation_200km"]

        # ── RULE 7: Receiver on fraud blocklist (weight 5) ──
        if receiver_msisdn and is_on_blocklist(receiver_msisdn):
            score += SCORE_WEIGHTS["receiver_on_blocklist"]
            reasons.append("R07: Receiver MSISDN on fraud blocklist")
            contributions["receiver_on_blocklist"] = SCORE_WEIGHTS["receiver_on_blocklist"]

        # ── RULE 8: Off-hours flag (weight 3) ──
        if features.get("off_hours_flag", 0):
            score += SCORE_WEIGHTS["off_hours_flag"]
            reasons.append("R08: Transaction outside sender's typical active hours")
            contributions["off_hours_flag"] = SCORE_WEIGHTS["off_hours_flag"]

        # ── RULE 9: Elevated agent complaint rate (weight 2) ──
        if features.get("agent_complaint_rate", 0.0) > 0.05:
            score += SCORE_WEIGHTS["agent_complaint_elevated"]
            reasons.append(
                f"R09: Agent complaint rate elevated ({features['agent_complaint_rate']:.2%})"
            )
            contributions["agent_complaint_elevated"] = SCORE_WEIGHTS["agent_complaint_elevated"]

        # ── RULE 10 (bonus): KYC limit exceeded (informational, +2 cap) ──
        if features.get("kyc_limit_exceeded", 0):
            bonus = min(2.0, MAX_SCORE - score)
            score += bonus
            reasons.append("R10: Amount exceeds BoZ KYC tier limit")
            contributions["kyc_limit_exceeded"] = bonus

        # Clamp pure rule score to [0, 100]
        rule_score = min(float(score), float(MAX_SCORE))

        # ── ML BLEND ──
        if ml_fraud_probability is not None and self.ml_blend_weight > 0:
            ml_score = ml_fraud_probability * MAX_SCORE
            final_score = (
                (1 - self.ml_blend_weight) * rule_score
                + self.ml_blend_weight * ml_score
            )
        else:
            final_score = rule_score

        final_score = round(min(final_score, float(MAX_SCORE)), 2)
        risk_level = _risk_tier(final_score)
        fraud_prob = ml_fraud_probability if ml_fraud_probability is not None else final_score / MAX_SCORE

        return ScoringResult(
            risk_score=final_score,
            risk_level=risk_level,
            reason_codes=reasons,
            fraud_probability=round(fraud_prob, 4),
            feature_contributions=contributions,
            recommended_action=determine_action(final_score, risk_level),
        )


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = RiskScoringEngine(ml_blend_weight=0.40)

    test_features = {
        "amount_to_average_ratio": 4.2,
        "velocity_1h": 2,
        "velocity_24h": 7,
        "new_beneficiary_flag": 1,
        "off_hours_flag": 1,
        "kyc_limit_exceeded": 1,
        "location_deviation_km": 350.0,
        "sim_swap_72h": 1,
        "new_device_flag": 1,
        "smishing_correlation_30m": 1,
        "agent_complaint_rate": 0.12,
    }

    result = engine.score(test_features, ml_fraud_probability=0.91)
    print("\n=== RISK SCORING RESULT ===")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
