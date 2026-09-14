"""
MODULE 9: Fraud Decision Overrides & Manual Flags
Mobile Money Fraud Detection System - Zambia
overrides.py

Implements two use cases from the actor diagram:
    - "Override Fraud Decision" (Fraud Analyst)
        Analyst can confirm a flagged transaction as genuine fraud or
        mark it as legitimate, with a mandatory justification.
    - "Manually Flag Transaction" (Fraud Analyst)
        Analyst can manually flag a transaction that was not auto-flagged,
        adding it to the investigation queue.

Both actions <<include>> "Record Audit Entry" — an audit trail is
maintained for all overrides and flags.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Session

from db_config import Base, get_engine, RiskScoreAudit, RetrainingCorpus


# ─────────────────────────────────────────────
# ORM MODELS
# ─────────────────────────────────────────────

class FraudOverride(Base):
    """Records an analyst's decision to override the system's fraud classification."""
    __tablename__ = "fraud_overrides"

    override_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    txn_id = Column(String(36), nullable=False, index=True)
    original_risk_level = Column(String(20), nullable=False)
    original_risk_score = Column(Float, nullable=False)
    override_decision = Column(String(30), nullable=False)   # CONFIRM_FRAUD or MARK_LEGITIMATE
    justification = Column(Text, nullable=False)
    overridden_by = Column(String(50), nullable=False)       # username
    created_at = Column(DateTime, default=datetime.utcnow)


class ManualFlag(Base):
    """Records a manual flag placed on a transaction by an analyst."""
    __tablename__ = "manual_flags"

    flag_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    txn_id = Column(String(36), nullable=False, index=True)
    flagged_by = Column(String(50), nullable=False)          # username
    reason = Column(Text, nullable=False)
    status = Column(String(20), default="OPEN")              # OPEN, REVIEWED, DISMISSED
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(50), nullable=True)


# ─────────────────────────────────────────────
# OVERRIDE FUNCTIONS
# ─────────────────────────────────────────────

def submit_override(
    txn_id: str,
    original_risk_level: str,
    original_risk_score: float,
    decision: str,
    justification: str,
    overridden_by: str,
) -> FraudOverride:
    """
    Submit a fraud decision override.

    Parameters
    ----------
    decision : 'CONFIRM_FRAUD' or 'MARK_LEGITIMATE'
    justification : mandatory reason text (must not be empty)

    Also writes a new RiskScoreAudit entry to maintain the audit trail
    (the <<include>> "Record Audit Entry" relationship from the use case diagram).
    """
    if decision not in ("CONFIRM_FRAUD", "MARK_LEGITIMATE"):
        raise ValueError(f"Invalid override decision: {decision}")
    if not justification.strip():
        raise ValueError("Justification is required for all overrides.")

    engine = get_engine()
    with Session(engine) as session:
        override = FraudOverride(
            override_id=str(uuid.uuid4()),
            txn_id=txn_id,
            original_risk_level=original_risk_level,
            original_risk_score=original_risk_score,
            override_decision=decision,
            justification=justification.strip(),
            overridden_by=overridden_by,
            created_at=datetime.utcnow(),
        )
        session.add(override)

        # <<include>> Record Audit Entry — write a new audit record reflecting
        # the analyst's override decision
        new_score = 100.0 if decision == "CONFIRM_FRAUD" else 0.0
        new_level = "High" if decision == "CONFIRM_FRAUD" else "Low"
        reason_codes = [
            f"OVERRIDE: {decision.replace('_', ' ').title()} by {overridden_by}",
            f"Justification: {justification.strip()[:200]}",
        ]

        audit_entry = RiskScoreAudit(
            score_id=str(uuid.uuid4()),
            txn_id=txn_id,
            numeric_score=new_score,
            risk_level=new_level,
            reason_codes_json=json.dumps(reason_codes),
            fraud_probability=1.0 if decision == "CONFIRM_FRAUD" else 0.0,
            calculated_at=datetime.utcnow(),
        )
        session.add(audit_entry)

        # <<include>> Add to Retraining Corpus (FR-13)
        # Confirmed labels feed the next monthly retraining cycle.
        confirmed_label = 1 if decision == "CONFIRM_FRAUD" else 0
        corpus_entry = RetrainingCorpus(
            corpus_id=str(uuid.uuid4()),
            txn_id=txn_id,
            confirmed_label=confirmed_label,
            source="analyst_override",
            added_at=datetime.utcnow(),
            used_in_retrain=False,
        )
        session.add(corpus_entry)

        session.commit()

        return override


def get_overrides_for_txn(txn_id: str) -> list[dict]:
    """Return all overrides for a specific transaction."""
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(FraudOverride)
            .filter_by(txn_id=txn_id)
            .order_by(FraudOverride.created_at.desc())
            .all()
        )
        return [
            {
                "override_id": r.override_id,
                "txn_id": r.txn_id,
                "original_risk_level": r.original_risk_level,
                "original_risk_score": r.original_risk_score,
                "override_decision": r.override_decision,
                "justification": r.justification,
                "overridden_by": r.overridden_by,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records
        ]


def get_override_history(limit: int = 100) -> list[dict]:
    """Return the most recent override decisions across all transactions."""
    engine = get_engine()
    with Session(engine) as session:
        records = (
            session.query(FraudOverride)
            .order_by(FraudOverride.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "override_id": r.override_id,
                "txn_id": r.txn_id,
                "original_risk_level": r.original_risk_level,
                "original_risk_score": r.original_risk_score,
                "override_decision": r.override_decision,
                "justification": r.justification,
                "overridden_by": r.overridden_by,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records
        ]


# ─────────────────────────────────────────────
# MANUAL FLAG FUNCTIONS
# ─────────────────────────────────────────────

def flag_transaction(
    txn_id: str,
    flagged_by: str,
    reason: str,
) -> ManualFlag:
    """
    Manually flag a transaction for investigation.
    Raises ValueError if the transaction is already flagged and OPEN.
    """
    if not reason.strip():
        raise ValueError("A reason is required when flagging a transaction.")

    engine = get_engine()
    with Session(engine) as session:
        existing = (
            session.query(ManualFlag)
            .filter_by(txn_id=txn_id, status="OPEN")
            .first()
        )
        if existing:
            raise ValueError(
                f"Transaction {txn_id} already has an open flag "
                f"(flag_id: {existing.flag_id})."
            )

        flag = ManualFlag(
            flag_id=str(uuid.uuid4()),
            txn_id=txn_id,
            flagged_by=flagged_by,
            reason=reason.strip(),
            status="OPEN",
            created_at=datetime.utcnow(),
        )
        session.add(flag)

        # <<include>> Add to Retraining Corpus (label=None = pending, FR-13)
        # Label will be resolved when the flag is reviewed or dismissed.
        corpus_entry = RetrainingCorpus(
            corpus_id=str(uuid.uuid4()),
            txn_id=txn_id,
            confirmed_label=None,  # resolved in update_flag_status()
            source="manual_flag",
            added_at=datetime.utcnow(),
            used_in_retrain=False,
        )
        session.add(corpus_entry)

        session.commit()
        return flag


def get_flagged_transactions(status: str | None = None, limit: int = 100) -> list[dict]:
    """Return manually flagged transactions, optionally filtered by status."""
    engine = get_engine()
    with Session(engine) as session:
        query = session.query(ManualFlag).order_by(ManualFlag.created_at.desc())
        if status:
            query = query.filter_by(status=status)
        records = query.limit(limit).all()
        return [
            {
                "flag_id": r.flag_id,
                "txn_id": r.txn_id,
                "flagged_by": r.flagged_by,
                "reason": r.reason,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "reviewed_by": r.reviewed_by,
            }
            for r in records
        ]


def update_flag_status(
    flag_id: str,
    new_status: str,
    reviewed_by: str,
) -> bool:
    """
    Update the status of a manual flag.
    new_status: 'REVIEWED' or 'DISMISSED'
    """
    if new_status not in ("REVIEWED", "DISMISSED"):
        raise ValueError(f"Invalid flag status: {new_status}")

    engine = get_engine()
    with Session(engine) as session:
        flag = session.query(ManualFlag).filter_by(flag_id=flag_id).first()
        if not flag:
            return False
        flag.status = new_status
        flag.reviewed_at = datetime.utcnow()
        flag.reviewed_by = reviewed_by

        # Resolve the pending RetrainingCorpus label for this transaction
        # REVIEWED → confirmed fraud (label=1), DISMISSED → not fraud (label=0)
        resolved_label = 1 if new_status == "REVIEWED" else 0
        corpus_row = (
            session.query(RetrainingCorpus)
            .filter_by(txn_id=flag.txn_id, source="manual_flag", confirmed_label=None)
            .first()
        )
        if corpus_row:
            corpus_row.confirmed_label = resolved_label

        session.commit()
        return True
