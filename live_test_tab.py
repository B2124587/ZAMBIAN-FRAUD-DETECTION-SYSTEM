"""
MODULE: Live Test Tab
Mobile Money Fraud Detection System - Zambia
live_test_tab.py

Provides render_live_test_tab() for the Streamlit dashboard.
Lets analysts submit real transactions through the scoring pipeline
and classify SMS text through the smishing NLP model.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import streamlit as st
from sqlalchemy.orm import Session

from db_config import (
    Transaction, RiskScoreAudit, get_engine, sha256_hash,
)
from data_pipeline import FeatureEngineer
from scoring_engine import RiskScoringEngine
from models import SmishingNLPClassifier


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TXN_TYPES = ["P2P", "CASH_OUT", "CASH_IN", "BILL_PAY", "MERCHANT"]
OPERATORS = ["MTN", "AIRTEL", "ZAMTEL"]
CHANNELS = ["USSD", "APP", "AGENT", "WEB"]
PROVINCES = [
    "Lusaka", "Copperbelt", "Central", "Southern", "Eastern",
    "Western", "Northern", "Muchinga", "North-Western", "Luapula",
]


# ─────────────────────────────────────────────
# CACHED SMISHING MODEL
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading smishing classifier…")
def _smishing_model() -> SmishingNLPClassifier:
    """Train once and cache for the lifetime of the Streamlit process."""
    clf = SmishingNLPClassifier()
    clf.fit()
    return clf


# ─────────────────────────────────────────────
# MAIN TAB RENDERER
# ─────────────────────────────────────────────

def render_live_test_tab():
    """Render the Live Test tab: transaction scoring + SMS classification."""

    # ── SECTION 1: TRANSACTION SCORING ──────────────────────────────
    st.subheader("🧪 Submit a Live Transaction for Scoring")
    st.caption(
        "Fill in the form below to score a real transaction against the "
        "rule-based engine.  The result is persisted to `transactions` and "
        "`risk_scores_audit` so it will appear in the Investigation Terminal "
        "and Overview KPIs on next refresh."
    )

    with st.form("live_txn_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            sender_msisdn = st.text_input(
                "Sender MSISDN",
                placeholder="e.g. +260961234567",
            )
            receiver_msisdn = st.text_input(
                "Receiver MSISDN",
                placeholder="e.g. +260977654321",
            )
            amount = st.number_input(
                "Amount (ZMW)", min_value=0.0, value=0.0, step=10.0, format="%.2f",
            )
            txn_type = st.selectbox("Transaction Type", TXN_TYPES)

        with col2:
            operator = st.selectbox("Operator", OPERATORS)
            channel = st.selectbox("Channel", CHANNELS)
            province = st.selectbox("Province", PROVINCES)
            device_fingerprint = st.text_input(
                "Device Fingerprint (optional)",
                placeholder="e.g. DEV_abc123…",
            )

        submitted = st.form_submit_button("⚡ Score Transaction", type="primary")

    # ── Validation & scoring (outside the form context) ──
    if submitted:
        # Validate required fields
        if not sender_msisdn.strip():
            st.error("**Sender MSISDN** is required.")
            return
        if not receiver_msisdn.strip():
            st.error("**Receiver MSISDN** is required.")
            return
        if amount <= 0:
            st.error("**Amount** must be greater than zero.")
            return

        now = datetime.now(timezone.utc)
        txn_id = str(uuid.uuid4())

        # Build payload for FeatureEngineer (raw MSISDNs — it hashes internally)
        payload = {
            "txn_id": txn_id,
            "timestamp": now,
            "sender_msisdn": sender_msisdn.strip(),
            "receiver_msisdn": receiver_msisdn.strip(),
            "amount": amount,
            "txn_type": txn_type,
            "operator": operator,
            "channel": channel,
            "province": province,
            "device_fingerprint": device_fingerprint.strip() or None,
        }

        with st.spinner("Engineering features and scoring…"):
            # 1. Feature engineering (real 11-feature vector)
            fe = FeatureEngineer()
            features = fe.engineer(payload)

            # 2. Risk scoring — rule-only for now.
            # TODO: To plug in a trained model later, load it here and pass
            #       its predict_proba output as ml_fraud_probability:
            #
            #   model = load_production_model()
            #   feature_array = fe.to_array(features).reshape(1, -1)
            #   ml_prob = float(model.predict_proba(feature_array)[0, 1])
            #   result = engine.score(features, receiver_msisdn=receiver_h,
            #                         ml_fraud_probability=ml_prob)
            engine = RiskScoringEngine(ml_blend_weight=0.0)

            # Hash the receiver MSISDN for the blocklist lookup
            receiver_h = (
                receiver_msisdn.strip()
                if len(receiver_msisdn.strip()) == 64
                else sha256_hash(receiver_msisdn.strip())
            )
            result = engine.score(
                features,
                receiver_msisdn=receiver_h,
            )

            # 3. Persist Transaction row (hashed PII)
            sender_h = (
                sender_msisdn.strip()
                if len(sender_msisdn.strip()) == 64
                else sha256_hash(sender_msisdn.strip())
            )
            device_h = None
            if device_fingerprint.strip():
                raw_dev = device_fingerprint.strip()
                device_h = raw_dev if len(raw_dev) == 64 else sha256_hash(raw_dev)

            txn_row = Transaction(
                txn_id=txn_id,
                timestamp=now,
                sender_msisdn=sender_h,
                receiver_msisdn=receiver_h,
                amount=amount,
                txn_type=txn_type,
                operator=operator,
                channel=channel,
                province=province,
                device_fingerprint=device_h,
                is_fraud=0,  # ground-truth unknown at submission time
            )

            # 4. Persist RiskScoreAudit row (mirrors backfill_audit_log.py)
            score_id = str(uuid.uuid4())
            audit_row = RiskScoreAudit(
                score_id=score_id,
                txn_id=txn_id,
                numeric_score=result.risk_score,
                risk_level=result.risk_level,
                reason_codes_json=json.dumps(result.reason_codes),
                fraud_probability=result.fraud_probability,
                calculated_at=now,
            )

            try:
                with Session(get_engine()) as session:
                    session.add(txn_row)
                    session.add(audit_row)
                    session.commit()
            except Exception as exc:
                st.error(f"Database write failed: {exc}")
                return

        # ── Display results ──
        st.success("✅ Transaction scored and persisted successfully.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{result.risk_score:.1f} / 100")
        m2.metric("Risk Tier", result.risk_level)
        m3.metric("Recommended Action", result.recommended_action)

        if result.reason_codes:
            st.write("**Reason Codes:**")
            for rc in result.reason_codes:
                st.write(f"- {rc}")
        else:
            st.info("No risk rules triggered — clean transaction.")

        st.caption(
            f"Transaction ID: `{txn_id}` · Score ID: `{score_id}`  \n"
            "This record is now in the audit trail and will appear in the "
            "Investigation Terminal and Overview tab on next data refresh."
        )

    # ── SECTION 2: SMS SMISHING CLASSIFIER ──────────────────────────
    st.divider()
    st.subheader("📱 SMS Smishing Classifier")
    st.caption(
        "Paste a real SMS message below and classify it as smishing or legitimate "
        "using the trained NLP model (TF-IDF + Multinomial Naive Bayes)."
    )

    sms_text = st.text_area(
        "SMS Text",
        placeholder="e.g. Congratulations! You have won K5000. Click http://bit.ly/claim to receive your prize.",
        height=120,
    )

    if st.button("🔍 Classify SMS", type="secondary"):
        if not sms_text.strip():
            st.error("Please enter an SMS message to classify.")
        else:
            clf = _smishing_model()
            proba = clf.predict_proba([sms_text.strip()])[0, 1]

            if proba >= 0.5:
                st.error(
                    f"🚨 **SMISHING DETECTED** — probability {proba:.1%}\n\n"
                    "This message shows characteristics of a phishing/smishing attempt."
                )
            else:
                st.success(
                    f"✅ **LEGITIMATE** — smishing probability {proba:.1%}\n\n"
                    "This message appears to be a normal notification."
                )

            st.metric("Smishing Probability", f"{proba:.2%}")
