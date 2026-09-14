"""
MODULE 1: Dashboard Database Connector & Query Layer
Mobile Money Fraud Detection System - Zambia
dashboard_db.py

DashboardDBConnector handles connection lifecycle and all parameterized
SQL reads required by the Sprint 3 Streamlit dashboard.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

load_dotenv()

# ─────────────────────────────────────────────
# ENGINE FACTORY
# ─────────────────────────────────────────────

from db_config import get_engine, DB_TYPE

def _build_engine(url: str | None = None):
    if url:
        return create_engine(url, pool_pre_ping=True, pool_recycle=3600), "Custom URL"
    
    return get_engine(), DB_TYPE


# ─────────────────────────────────────────────
# CONNECTOR CLASS
# ─────────────────────────────────────────────

class DashboardDBConnector:
    """
    Thin wrapper around SQLAlchemy that exposes read-only query methods
    returning pandas DataFrames for the Streamlit dashboard.
    """

    def __init__(self, database_url: str | None = None):
        self._engine, self.db_type = _build_engine(database_url)

    # ------------------------------------------------------------------
    # Connection health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except OperationalError:
            return False

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _read(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    # ------------------------------------------------------------------
    # SUMMARY CARD METRICS
    # ------------------------------------------------------------------

    def get_summary_metrics(
        self,
        operators: list[str] | None = None,
        provinces: list[str] | None = None,
        risk_levels: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict:
        """
        Aggregate KPI values for the top-level summary cards.
        Returns a dict: total_txns, fraud_rate, blocked_funds_zmw, avg_risk_score.
        """
        where, params = _build_txn_filter(
            operators, provinces, risk_levels, date_from, date_to
        )

        sql = f"""
            SELECT
                COUNT(t.txn_id)                                         AS total_txns,
                COALESCE(SUM(t.is_fraud), 0)                            AS total_fraud,
                COALESCE(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 0)
                                                                        AS blocked_funds,
                COALESCE(AVG(r.numeric_score), 0)                       AS avg_risk_score
            FROM transactions t
            LEFT JOIN risk_scores_audit r ON t.txn_id = r.txn_id
            {where}
        """
        row = self._read(sql, params).iloc[0]
        total = int(row["total_txns"]) or 1
        return {
            "total_txns": int(row["total_txns"]),
            "fraud_rate": round(float(row["total_fraud"]) / total * 100, 2),
            "blocked_funds_zmw": round(float(row["blocked_funds"]), 2),
            "avg_risk_score": round(float(row["avg_risk_score"]), 1),
        }

    # ------------------------------------------------------------------
    # HIGH-RISK ALERT FEED
    # ------------------------------------------------------------------

    def get_high_risk_alerts(
        self,
        threshold: float = 60.0,
        limit: int = 50,
        operators: list[str] | None = None,
        provinces: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> pd.DataFrame:
        """Transactions with risk_score >= threshold, most recent first."""
        where, params = _build_txn_filter(
            operators, provinces, None, date_from, date_to
        )
        params["threshold"] = threshold
        params["lim"] = limit

        masked_sender_sql = (
            "SUBSTR(t.sender_msisdn, 1, 8) || '****'" if self.db_type == "SQLite fallback"
            else "CONCAT(SUBSTRING(t.sender_msisdn, 1, 8), '****')"
        )

        sql = f"""
            SELECT
                t.txn_id,
                t.timestamp,
                {masked_sender_sql} AS masked_sender,
                t.amount,
                t.operator,
                t.province,
                t.txn_type,
                r.numeric_score,
                r.risk_level,
                r.reason_codes_json,
                r.fraud_probability
            FROM transactions t
            INNER JOIN risk_scores_audit r ON t.txn_id = r.txn_id
            {where}
              {"AND" if where else "WHERE"} r.numeric_score >= :threshold
            ORDER BY r.numeric_score DESC, r.calculated_at DESC
            LIMIT :lim
        """
        return self._read(sql, params)

    # ------------------------------------------------------------------
    # PROVINCIAL FRAUD DISTRIBUTION
    # ------------------------------------------------------------------

    def get_province_fraud_counts(
        self,
        operators: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> pd.DataFrame:
        """Fraud counts per Zambian province for the bar chart."""
        where, params = _build_txn_filter(operators, None, None, date_from, date_to)
        sql = f"""
            SELECT
                t.province,
                COUNT(*) AS total_txns,
                SUM(t.is_fraud) AS fraud_count,
                ROUND(SUM(t.is_fraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
            FROM transactions t
            {where}
            GROUP BY t.province
            ORDER BY fraud_count DESC
        """
        return self._read(sql, params)

    # ------------------------------------------------------------------
    # OPERATOR FRAUD DISTRIBUTION
    # ------------------------------------------------------------------

    def get_operator_fraud_counts(
        self,
        provinces: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> pd.DataFrame:
        """Fraud counts per operator for the pie/donut chart."""
        where, params = _build_txn_filter(None, provinces, None, date_from, date_to)
        sql = f"""
            SELECT
                t.operator,
                COUNT(*) AS total_txns,
                SUM(t.is_fraud) AS fraud_count
            FROM transactions t
            {where}
            GROUP BY t.operator
            ORDER BY fraud_count DESC
        """
        return self._read(sql, params)

    # ------------------------------------------------------------------
    # DAILY TREND (time-series)
    # ------------------------------------------------------------------

    def get_daily_trend(
        self,
        days: int = 30,
        operators: list[str] | None = None,
        provinces: list[str] | None = None,
    ) -> pd.DataFrame:
        date_from = datetime.utcnow() - timedelta(days=days)
        where, params = _build_txn_filter(operators, provinces, None, date_from, None)
        sql = f"""
            SELECT
                DATE(t.timestamp) AS txn_date,
                COUNT(*) AS total_txns,
                SUM(t.is_fraud) AS fraud_count
            FROM transactions t
            {where}
            GROUP BY DATE(t.timestamp)
            ORDER BY txn_date ASC
        """
        return self._read(sql, params)

    # ------------------------------------------------------------------
    # AUDIT QUEUE
    # ------------------------------------------------------------------

    def get_audit_queue(
        self,
        operators: list[str] | None = None,
        provinces: list[str] | None = None,
        risk_levels: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Full audit log for the Investigation Terminal tab."""
        where, params = _build_txn_filter(
            operators, provinces, risk_levels, date_from, date_to
        )
        params["lim"] = limit
        masked_sender_sql = (
            "SUBSTR(t.sender_msisdn, 1, 8) || '****'" if self.db_type == "SQLite fallback"
            else "CONCAT(SUBSTRING(t.sender_msisdn, 1, 8), '****')"
        )
        sql = f"""
            SELECT
                r.score_id,
                t.txn_id,
                t.timestamp,
                {masked_sender_sql} AS masked_sender,
                t.amount,
                t.operator,
                t.province,
                t.txn_type,
                t.channel,
                r.numeric_score,
                r.risk_level,
                r.reason_codes_json,
                r.fraud_probability,
                r.calculated_at,
                t.is_fraud
            FROM risk_scores_audit r
            INNER JOIN transactions t ON r.txn_id = t.txn_id
            {where}
            ORDER BY r.calculated_at DESC
            LIMIT :lim
        """
        return self._read(sql, params)

    # ------------------------------------------------------------------
    # TRANSACTION DRILL-DOWN
    # ------------------------------------------------------------------

    def get_transaction_detail(self, txn_id: str) -> Optional[pd.DataFrame]:
        """Full detail view for a single transaction (join with audit)."""
        sql = """
            SELECT
                t.txn_id,
                t.timestamp,
                t.sender_msisdn,
                t.receiver_msisdn,
                t.amount,
                t.txn_type,
                t.operator,
                t.channel,
                t.province,
                t.device_fingerprint,
                t.is_fraud,
                r.score_id,
                r.numeric_score,
                r.risk_level,
                r.reason_codes_json,
                r.fraud_probability,
                r.calculated_at
            FROM transactions t
            LEFT JOIN risk_scores_audit r ON t.txn_id = r.txn_id
            WHERE t.txn_id = :txn_id
            ORDER BY r.calculated_at DESC
            LIMIT 1
        """
        df = self._read(sql, {"txn_id": txn_id})
        return df if not df.empty else None

    def search_by_msisdn(self, msisdn_fragment: str, limit: int = 50) -> pd.DataFrame:
        """Search audit queue by partial sender MSISDN (hashed prefix)."""
        masked_sender_sql = (
            "SUBSTR(t.sender_msisdn, 1, 8) || '****'" if self.db_type == "SQLite fallback"
            else "CONCAT(SUBSTRING(t.sender_msisdn, 1, 8), '****')"
        )
        sql = f"""
            SELECT
                r.score_id,
                t.txn_id,
                t.timestamp,
                {masked_sender_sql} AS masked_sender,
                t.amount,
                t.operator,
                t.province,
                r.numeric_score,
                r.risk_level,
                r.reason_codes_json,
                r.fraud_probability
            FROM risk_scores_audit r
            INNER JOIN transactions t ON r.txn_id = t.txn_id
            WHERE t.sender_msisdn LIKE :fragment
               OR t.txn_id LIKE :fragment
            ORDER BY r.calculated_at DESC
            LIMIT :lim
        """
        return self._read(
            sql,
            {"fragment": f"%{msisdn_fragment}%", "lim": limit},
        )

    # ------------------------------------------------------------------
    # FEATURE VECTOR FOR DRILL-DOWN (last audit record for a txn)
    # ------------------------------------------------------------------

    def get_feature_vector(self, txn_id: str) -> dict:
        """
        Reconstruct the feature vector by querying behavioral data.
        Returns a flat dict of feature_name -> value.
        Falls back to zeros for missing records.
        """
        sql = """
            SELECT
                t.amount,
                t.province,
                t.channel,
                t.timestamp,
                bp.avg_30day_amount,
                bp.typical_active_hours,
                bp.usual_province_centroid,
                bp.sim_swap_flag,
                r.reason_codes_json
            FROM transactions t
            LEFT JOIN users_profiles up ON t.sender_msisdn = up.hashed_msisdn
            LEFT JOIN behavioral_profiles bp ON up.hashed_user_id = bp.hashed_user_id
            LEFT JOIN risk_scores_audit r ON t.txn_id = r.txn_id
            WHERE t.txn_id = :txn_id
            LIMIT 1
        """
        df = self._read(sql, {"txn_id": txn_id})
        if df.empty:
            return {}
        row = df.iloc[0]

        avg_amt = float(row.get("avg_30day_amount") or 500.0)
        amount = float(row.get("amount") or 0.0)
        reason_codes = json.loads(row.get("reason_codes_json") or "[]")

        features = {
            "amount_to_average_ratio": round(amount / avg_amt, 4) if avg_amt else 0,
            "velocity_1h": _extract_feature_from_reasons(reason_codes, "velocity_1h"),
            "velocity_24h": _extract_feature_from_reasons(reason_codes, "velocity_24h"),
            "new_beneficiary_flag": 1 if any("new beneficiary" in r.lower() for r in reason_codes) else 0,
            "off_hours_flag": 1 if any("off-hours" in r.lower() or "active hours" in r.lower() for r in reason_codes) else 0,
            "kyc_limit_exceeded": 1 if any("kyc" in r.lower() for r in reason_codes) else 0,
            "location_deviation_km": _extract_km_from_reasons(reason_codes),
            "sim_swap_72h": 1 if row.get("sim_swap_flag") else 0,
            "new_device_flag": 1 if any("device" in r.lower() for r in reason_codes) else 0,
            "smishing_correlation_30m": 1 if any("smishing" in r.lower() for r in reason_codes) else 0,
            "agent_complaint_rate": _extract_agent_rate(reason_codes),
        }
        return features

    # ------------------------------------------------------------------
    # SYSTEM METRICS (simulated from audit data)
    # ------------------------------------------------------------------

    def get_model_performance_metrics(self) -> dict:
        """
        Derive approximate model performance from audit vs actual fraud labels.
        Returns precision, recall, f1, mcc estimates.
        """
        sql = """
            SELECT
                t.is_fraud,
                r.numeric_score,
                r.risk_level
            FROM risk_scores_audit r
            INNER JOIN transactions t ON r.txn_id = t.txn_id
        """
        df = self._read(sql)
        if df.empty:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "sample_size": 0}

        # Binary predictions: High risk = predicted fraud
        df["predicted_fraud"] = (df["numeric_score"] >= 60).astype(int)
        tp = ((df["predicted_fraud"] == 1) & (df["is_fraud"] == 1)).sum()
        fp = ((df["predicted_fraud"] == 1) & (df["is_fraud"] == 0)).sum()
        fn = ((df["predicted_fraud"] == 0) & (df["is_fraud"] == 1)).sum()
        tn = ((df["predicted_fraud"] == 0) & (df["is_fraud"] == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "sample_size": len(df),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        }

    def get_risk_score_distribution(self) -> pd.DataFrame:
        """Score distribution bucketed by 10-point intervals."""
        score_bucket_sql = (
            "CAST(numeric_score / 10 AS INTEGER) * 10" if self.db_type == "SQLite fallback"
            else "FLOOR(numeric_score / 10) * 10"
        )
        sql = f"""
            SELECT
                {score_bucket_sql} AS score_bucket,
                COUNT(*) AS count
            FROM risk_scores_audit
            GROUP BY score_bucket
            ORDER BY score_bucket
        """
        return self._read(sql)

    # ------------------------------------------------------------------
    # PROFILE MANAGEMENT (Admin)
    # ------------------------------------------------------------------

    def search_user_profiles(self, fragment: str, limit: int = 20) -> pd.DataFrame:
        """Search for user behavioral profiles by hashed MSISDN or UID."""
        sql = """
            SELECT
                up.hashed_user_id,
                up.hashed_msisdn,
                up.kyc_tier,
                bp.avg_30day_amount,
                bp.typical_active_hours,
                bp.usual_province_centroid,
                bp.sim_swap_flag,
                bp.last_updated
            FROM users_profiles up
            INNER JOIN behavioral_profiles bp ON up.hashed_user_id = bp.hashed_user_id
            WHERE up.hashed_msisdn LIKE :frag
               OR up.hashed_user_id LIKE :frag
            ORDER BY bp.last_updated DESC
            LIMIT :lim
        """
        return self._read(sql, {"frag": f"%{fragment}%", "lim": limit})

    def update_user_profile(
        self,
        hashed_msisdn: str,
        kyc_tier: int,
        avg_30day_amount: float,
        typical_active_hours: str,
        usual_province_centroid: str,
    ) -> bool:
        """Update a user's KYC tier and behavioral profile parameters."""
        try:
            from sqlalchemy.orm import Session
            with Session(self._engine) as session:
                from db_config import UserProfile, BehavioralProfile
                up = session.query(UserProfile).filter_by(hashed_msisdn=hashed_msisdn).first()
                if not up:
                    return False
                
                up.kyc_tier = kyc_tier
                if up.behavioral_profile:
                    up.behavioral_profile.avg_30day_amount = avg_30day_amount
                    up.behavioral_profile.typical_active_hours = typical_active_hours
                    up.behavioral_profile.usual_province_centroid = usual_province_centroid
                    up.behavioral_profile.last_updated = datetime.utcnow()
                session.commit()
                return True
        except Exception:
            return False


# ─────────────────────────────────────────────
# FILTER BUILDER HELPERS
# ─────────────────────────────────────────────

def _build_txn_filter(
    operators: list[str] | None,
    provinces: list[str] | None,
    risk_levels: list[str] | None,
    date_from: datetime | None,
    date_to: datetime | None,
) -> tuple[str, dict]:
    clauses = []
    params: dict = {}

    if operators:
        placeholders = ", ".join(f":op{i}" for i in range(len(operators)))
        clauses.append(f"t.operator IN ({placeholders})")
        for i, op in enumerate(operators):
            params[f"op{i}"] = op

    if provinces:
        placeholders = ", ".join(f":prov{i}" for i in range(len(provinces)))
        clauses.append(f"t.province IN ({placeholders})")
        for i, p in enumerate(provinces):
            params[f"prov{i}"] = p

    if risk_levels:
        placeholders = ", ".join(f":rl{i}" for i in range(len(risk_levels)))
        clauses.append(f"r.risk_level IN ({placeholders})")
        for i, rl in enumerate(risk_levels):
            params[f"rl{i}"] = rl

    if date_from:
        clauses.append("t.timestamp >= :date_from")
        params["date_from"] = date_from

    if date_to:
        clauses.append("t.timestamp <= :date_to")
        params["date_to"] = date_to

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ─────────────────────────────────────────────
# REASON CODE PARSING HELPERS
# ─────────────────────────────────────────────

def _extract_feature_from_reasons(reasons: list[str], feature: str) -> int:
    """Rough extraction of velocity counts from reason text."""
    for r in reasons:
        if "velocity" in r.lower() or "txns/24h" in r.lower():
            import re
            match = re.search(r"(\d+)", r)
            if match:
                return int(match.group(1))
    return 0


def _extract_km_from_reasons(reasons: list[str]) -> float:
    import re
    for r in reasons:
        if "km" in r.lower():
            match = re.search(r"(\d+(?:\.\d+)?)\s*km", r)
            if match:
                return float(match.group(1))
    return 0.0


def _extract_agent_rate(reasons: list[str]) -> float:
    import re
    for r in reasons:
        if "agent complaint" in r.lower():
            match = re.search(r"(\d+\.\d+)%", r)
            if match:
                return float(match.group(1)) / 100
    return 0.0