"""
UTILITY: Backfill risk_scores_audit
Mobile Money Fraud Detection System - Zambia
backfill_audit_log.py

The dashboard (app_dashboard.py) reads exclusively from risk_scores_audit.
That table is normally only written to when the live /score API endpoint
is called. Right after seeding the database (db_config.py / main.py),
risk_scores_audit is empty or has just one test record — so this script
scores every existing transaction directly against the rule engine
(and an ML model, if one has been trained and saved) and writes one
audit row per transaction. Run it once after seeding so the dashboard
has real data to show.

Usage:
    python backfill_audit_log.py                # rule + ML blend, wipes table first
    python backfill_audit_log.py --no-model      # rule-based scoring only
    python backfill_audit_log.py --keep-existing # append instead of wiping
    python backfill_audit_log.py --limit 1000    # only score the first N transactions
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from db_config import get_engine, create_schema, Transaction, RiskScoreAudit
from data_pipeline import FeatureEngineer
from scoring_engine import RiskScoringEngine


def _load_saved_model():
    """Best-effort load of a previously trained model from saved_models/."""
    from models import MLModel

    for filename in ("randomforest.pkl", "xgboost.pkl", "logisticregression.pkl"):
        path = Path("saved_models") / filename
        if path.exists():
            try:
                model = MLModel.load(path)
                print(f"[backfill] Loaded saved model: {model.name} ({path})")
                return model
            except Exception as exc:
                print(f"[backfill] Could not load {path} ({exc}); trying next.")
    print("[backfill] No saved model found — scoring with rules only.")
    return None


def backfill(limit: int | None = None, use_model: bool = True, wipe: bool = True):
    engine = get_engine()
    create_schema()

    fe = FeatureEngineer()
    model = _load_saved_model() if use_model else None
    if model is not None and hasattr(model, "_model"):
        if hasattr(model._model, "n_jobs"):
            model._model.n_jobs = 1  # Disable parallel overhead for row-by-row inference

    risk_engine = RiskScoringEngine(ml_blend_weight=0.30 if model is not None else 0.0)

    with Session(engine) as session:
        if wipe:
            deleted = session.query(RiskScoreAudit).delete()
            session.commit()
            print(f"[backfill] Cleared {deleted} existing audit record(s).")

        already_scored = {row[0] for row in session.query(RiskScoreAudit.txn_id).all()}

        query = session.query(Transaction)
        if limit:
            query = query.limit(limit)
        txns = query.all()
        print(f"[backfill] Scoring {len(txns)} transactions...")

        scored, skipped = 0, 0
        for i, t in enumerate(txns, start=1):
            if t.txn_id in already_scored:
                skipped += 1
                continue

            payload = {
                "txn_id": t.txn_id,
                "timestamp": t.timestamp,
                "sender_msisdn": t.sender_msisdn,
                "receiver_msisdn": t.receiver_msisdn,
                "amount": t.amount,
                "txn_type": t.txn_type,
                "operator": t.operator,
                "channel": t.channel,
                "province": t.province,
                "device_fingerprint": t.device_fingerprint,
            }
            features = fe.engineer(payload)

            ml_prob = None
            if model is not None:
                feature_array = fe.to_array(features).reshape(1, -1)
                ml_prob = float(model.predict_proba(feature_array)[0, 1])

            result = risk_engine.score(
                features,
                receiver_msisdn=t.receiver_msisdn,
                ml_fraud_probability=ml_prob,
            )

            session.add(RiskScoreAudit(
                score_id=str(uuid.uuid4()),
                txn_id=t.txn_id,
                numeric_score=result.risk_score,
                risk_level=result.risk_level,
                reason_codes_json=json.dumps(result.reason_codes),
                fraud_probability=result.fraud_probability,
                calculated_at=t.timestamp,
            ))
            scored += 1

            if i % 250 == 0:
                session.commit()
                print(f"[backfill]   ...{i}/{len(txns)} processed")

        session.commit()

    print(f"[backfill] Done. Scored {scored} transaction(s); skipped {skipped} already-audited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill risk_scores_audit from existing transactions.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N transactions.")
    parser.add_argument("--no-model", action="store_true", help="Score with the rule engine only (skip ML blend).")
    parser.add_argument("--keep-existing", action="store_true", help="Append instead of wiping risk_scores_audit first.")
    args = parser.parse_args()

    backfill(limit=args.limit, use_model=not args.no_model, wipe=not args.keep_existing)
