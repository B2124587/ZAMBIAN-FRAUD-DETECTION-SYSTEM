"""
MODULE 2: SQL-Driven Feature Engineering Pipeline
Mobile Money Fraud Detection System - Zambia
data_pipeline.py

Queries MySQL/SQLite for behavioral features, engineers the 11-feature vector,
and prepares stratified train/val/test splits with SMOTE oversampling.
"""

import json
from datetime import datetime, timedelta
from math import radians, cos, sin, sqrt, atan2

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sqlalchemy import text
from sqlalchemy.orm import Session

from db_config import (
    BehavioralProfile, Transaction, UserProfile, PROVINCE_CENTROIDS,
    RetrainingCorpus, get_engine, sha256_hash, get_kyc_limit
)


FEATURE_NAMES = [
    "amount_to_average_ratio",
    "velocity_1h",
    "velocity_24h",
    "new_beneficiary_flag",
    "off_hours_flag",
    "kyc_limit_exceeded",
    "location_deviation_km",
    "sim_swap_72h",
    "new_device_flag",
    "smishing_correlation_30m",
    "agent_complaint_rate",
]


# ─────────────────────────────────────────────
# HAVERSINE HELPER
# ─────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    Accepts a transaction payload and queries the database to produce
    the 11-element engineered feature vector.
    """

    def __init__(self):
        self.engine = get_engine()

    # ------------------------------------------------------------------
    # 1. amount_to_average_ratio
    # ------------------------------------------------------------------
    def _amount_to_avg_ratio(self, amount: float, bp: BehavioralProfile) -> float:
        avg = bp.avg_30day_amount if bp and bp.avg_30day_amount > 0 else 500.0
        return round(amount / avg, 4)

    # ------------------------------------------------------------------
    # 2 & 3. velocity_1h / velocity_24h
    # ------------------------------------------------------------------
    def _velocity(self, sender_msisdn: str, timestamp: datetime, session: Session):
        ts_1h = timestamp - timedelta(hours=1)
        ts_24h = timestamp - timedelta(hours=24)

        v1h = session.execute(
            text(
                "SELECT COUNT(*) FROM transactions "
                "WHERE sender_msisdn = :msisdn AND timestamp >= :ts1h AND timestamp < :ts"
            ),
            {"msisdn": sender_msisdn, "ts1h": ts_1h, "ts": timestamp},
        ).scalar() or 0

        v24h = session.execute(
            text(
                "SELECT COUNT(*) FROM transactions "
                "WHERE sender_msisdn = :msisdn AND timestamp >= :ts24h AND timestamp < :ts"
            ),
            {"msisdn": sender_msisdn, "ts24h": ts_24h, "ts": timestamp},
        ).scalar() or 0

        return int(v1h), int(v24h)

    # ------------------------------------------------------------------
    # 4. new_beneficiary_flag
    # ------------------------------------------------------------------
    def _new_beneficiary_flag(self, receiver_msisdn: str, bp: BehavioralProfile) -> int:
        if bp is None:
            return 1
        known = json.loads(bp.known_beneficiaries_json or "[]")
        return 0 if receiver_msisdn in known else 1

    # ------------------------------------------------------------------
    # 5. off_hours_flag
    # ------------------------------------------------------------------
    def _off_hours_flag(self, timestamp: datetime, bp: BehavioralProfile) -> int:
        if bp is None:
            return 0
        try:
            parts = bp.typical_active_hours.split("-")
            h_start, h_end = int(parts[0]), int(parts[1])
        except Exception:
            return 0
        hour = timestamp.hour
        return 0 if h_start <= hour <= h_end else 1

    # ------------------------------------------------------------------
    # 6. kyc_limit_exceeded
    # ------------------------------------------------------------------
    def _kyc_limit_exceeded(self, amount: float, kyc_tier: int) -> int:
        limit = get_kyc_limit(kyc_tier)
        return 1 if amount > limit else 0

    # ------------------------------------------------------------------
    # 7. location_deviation_km
    # ------------------------------------------------------------------
    def _location_deviation_km(self, txn_province: str, bp: BehavioralProfile) -> float:
        if bp is None:
            return 0.0
        home = PROVINCE_CENTROIDS.get(bp.usual_province_centroid, (-15.4167, 28.2833))
        current = PROVINCE_CENTROIDS.get(txn_province, home)
        return round(haversine_km(home[0], home[1], current[0], current[1]), 2)

    # ------------------------------------------------------------------
    # 8. sim_swap_72h
    # ------------------------------------------------------------------
    def _sim_swap_72h(self, bp: BehavioralProfile) -> int:
        if bp is None:
            return 0
        # Treat sim_swap_flag as the 72h indicator (set during recent swap events)
        return 1 if bp.sim_swap_flag else 0

    # ------------------------------------------------------------------
    # 9. new_device_flag
    # ------------------------------------------------------------------
    def _new_device_flag(self, device_fp: str | None, bp: BehavioralProfile) -> int:
        if device_fp is None or bp is None:
            return 0
        registered = json.loads(bp.registered_devices_json or "[]")
        return 0 if device_fp in registered else 1

    # ------------------------------------------------------------------
    # 10. smishing_correlation_30m
    # ------------------------------------------------------------------
    def _smishing_correlation_30m(
        self, sender_msisdn: str, timestamp: datetime, session: Session
    ) -> int:
        ts_30m = timestamp - timedelta(minutes=30)
        count = session.execute(
            text(
                "SELECT COUNT(*) FROM smishing_signals "
                "WHERE hashed_msisdn = :msisdn AND detected_at >= :ts30m AND detected_at <= :ts"
            ),
            {"msisdn": sender_msisdn, "ts30m": ts_30m, "ts": timestamp},
        ).scalar() or 0
        return 1 if count > 0 else 0

    # ------------------------------------------------------------------
    # 11. agent_complaint_rate
    # ------------------------------------------------------------------
    def _agent_complaint_rate(self, receiver_msisdn: str, channel: str, session: Session) -> float:
        if channel != "AGENT":
            return 0.0
        row = session.execute(
            text(
                "SELECT complaint_rate FROM agent_metadata WHERE hashed_msisdn = :msisdn"
            ),
            {"msisdn": receiver_msisdn},
        ).fetchone()
        return float(row[0]) if row else 0.0

    # ------------------------------------------------------------------
    # MAIN: engineer features for one transaction payload
    # ------------------------------------------------------------------
    def engineer(self, payload: dict) -> dict:
        """
        payload keys:
            txn_id, timestamp, sender_msisdn (raw or hashed), receiver_msisdn,
            amount, txn_type, operator, channel, province, device_fingerprint
        Returns dict of feature_name -> value.
        """
        # Normalise MSISDNs to hashed form
        sender_h = (
            payload["sender_msisdn"]
            if len(payload["sender_msisdn"]) == 64
            else sha256_hash(payload["sender_msisdn"])
        )
        receiver_h = (
            payload["receiver_msisdn"]
            if len(payload["receiver_msisdn"]) == 64
            else sha256_hash(payload["receiver_msisdn"])
        )
        device_h = None
        if payload.get("device_fingerprint"):
            raw_dev = payload["device_fingerprint"]
            device_h = raw_dev if len(raw_dev) == 64 else sha256_hash(raw_dev)

        amount = float(payload["amount"])
        timestamp = payload["timestamp"] if isinstance(payload["timestamp"], datetime) else datetime.fromisoformat(payload["timestamp"])
        txn_province = payload.get("province", "Lusaka")
        channel = payload.get("channel", "USSD")

        # Lookup sender profile & query features
        bp = None
        kyc_tier = 1
        v1h, v24h = 0, 0
        smishing = 0
        agent_rate = 0.0

        try:
            with Session(self.engine) as session:
                user = session.query(UserProfile).filter_by(hashed_msisdn=sender_h).first()
                if user:
                    bp = session.query(BehavioralProfile).filter_by(
                        hashed_user_id=user.hashed_user_id
                    ).first()
                    kyc_tier = user.kyc_tier

                v1h, v24h = self._velocity(sender_h, timestamp, session)
                smishing = self._smishing_correlation_30m(sender_h, timestamp, session)
                agent_rate = self._agent_complaint_rate(receiver_h, channel, session)
        except Exception as e:
            # DB unavailable - use default feature values to allow degraded rule-based scoring
            print(f"[data_pipeline] Feature engineer DB fail: {e}")

        features = {
            "amount_to_average_ratio": self._amount_to_avg_ratio(amount, bp),
            "velocity_1h": v1h,
            "velocity_24h": v24h,
            "new_beneficiary_flag": self._new_beneficiary_flag(receiver_h, bp),
            "off_hours_flag": self._off_hours_flag(timestamp, bp),
            "kyc_limit_exceeded": self._kyc_limit_exceeded(amount, kyc_tier),
            "location_deviation_km": self._location_deviation_km(txn_province, bp),
            "sim_swap_72h": self._sim_swap_72h(bp),
            "new_device_flag": self._new_device_flag(device_h, bp),
            "smishing_correlation_30m": smishing,
            "agent_complaint_rate": agent_rate,
        }
        return features

    def to_array(self, features: dict) -> np.ndarray:
        return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)


# ─────────────────────────────────────────────
# DATASET BUILDER (for model training)
# ─────────────────────────────────────────────

class DatasetBuilder:
    """Loads all transactions from DB, engineers features, and splits into train/val/test."""

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.engine = get_engine()

    def build(self, include_corpus: bool = True) -> tuple:
        """
        Returns X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.
        SMOTE applied to training partition only.

        include_corpus: if True, analyst-confirmed labels from retraining_corpus
                        are appended to the training partition before SMOTE (FR-13).
        """
        print("[data_pipeline] Loading transactions from database...")

        with Session(self.engine) as session:
            txns = session.query(Transaction).all()
            rows = []
            for t in txns:
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
                rows.append((payload, t.is_fraud))

        print(f"[data_pipeline] Engineering features for {len(rows)} transactions...")

        feature_rows = []
        labels = []
        for payload, label in rows:
            feats = self.engineer.engineer(payload)
            feature_rows.append([feats[k] for k in FEATURE_NAMES])
            labels.append(label)

        X = np.array(feature_rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Stratified 70 / 15 / 15 split
        X_train_full, X_temp, y_train_full, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        # Optionally augment training set with analyst-confirmed corpus rows (FR-13)
        if include_corpus:
            corpus_builder = RetrainingCorpusBuilder(engineer=self.engineer)
            X_corpus, y_corpus = corpus_builder.build(unused_only=False)
            if X_corpus.shape[0] > 0:
                X_train_full = np.vstack([X_train_full, X_corpus])
                y_train_full = np.concatenate([y_train_full, y_corpus])
                print(f"[data_pipeline] Appended {X_corpus.shape[0]} corpus rows to training set.")

        # SMOTE on training set only
        print(f"[data_pipeline] Applying SMOTE to training set (fraud: {y_train_full.sum()}/{len(y_train_full)})...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, max(1, y_train_full.sum() - 1)))
        X_train, y_train = smote.fit_resample(X_train_full, y_train_full)
        print(f"[data_pipeline] Post-SMOTE training set: {X_train.shape}, fraud: {y_train.sum()}/{len(y_train)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# RETRAINING CORPUS BUILDER (FR-13)
# ─────────────────────────────────────────────

class RetrainingCorpusBuilder:
    """
    Joins retraining_corpus → transactions, engineers features for each
    confirmed-label row, and returns (X, y) numpy arrays.

    Used by DatasetBuilder (include_corpus=True) and by RetrainingPipeline
    to prepare the augmented dataset before the monthly retraining run.
    """

    def __init__(self, engineer: FeatureEngineer | None = None):
        self.engineer = engineer or FeatureEngineer()
        self.engine = get_engine()

    def build(self, unused_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        unused_only: if True, only return rows where used_in_retrain=False
                     (prevents double-counting across retraining cycles).

        Returns
        -------
        X : float32 array of shape (n, len(FEATURE_NAMES))
        y : int32 array of shape (n,)
        """
        with Session(self.engine) as session:
            query = (
                session.query(RetrainingCorpus, Transaction)
                .join(Transaction, RetrainingCorpus.txn_id == Transaction.txn_id)
                .filter(RetrainingCorpus.confirmed_label.isnot(None))
            )
            if unused_only:
                query = query.filter(RetrainingCorpus.used_in_retrain == False)  # noqa: E712

            rows = query.all()

        if not rows:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int32)

        print(f"[data_pipeline] RetrainingCorpusBuilder: engineering {len(rows)} confirmed rows...")
        feature_rows, labels = [], []
        for corpus_row, txn in rows:
            payload = {
                "txn_id": txn.txn_id,
                "timestamp": txn.timestamp,
                "sender_msisdn": txn.sender_msisdn,
                "receiver_msisdn": txn.receiver_msisdn,
                "amount": txn.amount,
                "txn_type": txn.txn_type,
                "operator": txn.operator,
                "channel": txn.channel,
                "province": txn.province,
                "device_fingerprint": txn.device_fingerprint,
            }
            try:
                feats = self.engineer.engineer(payload)
                feature_rows.append([feats[k] for k in FEATURE_NAMES])
                labels.append(int(corpus_row.confirmed_label))
            except Exception as e:
                print(f"[data_pipeline] Skipping corpus row {corpus_row.corpus_id}: {e}")

        return (
            np.array(feature_rows, dtype=np.float32),
            np.array(labels, dtype=np.int32),
        )

    def mark_used(self, session=None) -> int:
        """
        Flip used_in_retrain=True for all currently unused confirmed rows.
        Called by RetrainingPipeline after a successful promotion cycle.
        Returns the count of rows marked.
        """
        close_session = session is None
        if session is None:
            session = Session(self.engine)
        try:
            rows = (
                session.query(RetrainingCorpus)
                .filter(
                    RetrainingCorpus.confirmed_label.isnot(None),
                    RetrainingCorpus.used_in_retrain == False,  # noqa: E712
                )
                .all()
            )
            for r in rows:
                r.used_in_retrain = True
            session.commit()
            return len(rows)
        finally:
            if close_session:
                session.close()


if __name__ == "__main__":
    from db_config import create_schema, populate_database
    create_schema()
    populate_database()

    builder = DatasetBuilder()
    X_train, X_val, X_test, y_train, y_val, y_test = builder.build()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
