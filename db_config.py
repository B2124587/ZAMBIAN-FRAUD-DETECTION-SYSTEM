"""
MODULE 1: Database Setup & Population
Mobile Money Fraud Detection System - Zambia
db_config.py

Handles MySQL/SQLite schema creation, synthetic data generation, and PII hashing.
Uses SQLite for local development (swap engine URL for MySQL in production).
"""

import hashlib
import hmac
import json
import os
import random
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from math import radians, cos, sin, sqrt, atan2

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

load_dotenv()

# ─────────────────────────────────────────────
# DATABASE ENGINE
# Credentials must come from environment variables — never hardcode them.
# Copy .env.example to .env and fill in real values before running anything.
# ─────────────────────────────────────────────

DB_TYPE = "Unknown"

def _build_engine_and_type():
    global DB_TYPE
    
    # 1. DATABASE_URL environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        DB_TYPE = "Configured URL"
        return create_engine(db_url, echo=False, pool_pre_ping=True)

    # 2. Try configured MySQL
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if not missing:
        db_user = os.getenv("DB_USER")
        db_pass = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME")
        mysql_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        eng = create_engine(mysql_url, echo=False, pool_pre_ping=True, pool_recycle=3600)
        try:
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            DB_TYPE = "MySQL"
            return eng
        except Exception:
            pass # fallback to sqlite

    # 3. SQLite fallback
    DB_TYPE = "SQLite fallback"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sqlite_path = os.path.join(base_dir, "fraud_system.db")
    return create_engine(f"sqlite:///{sqlite_path}", echo=False, pool_pre_ping=True)

engine = _build_engine_and_type()
DATABASE_URL = str(engine.url)

# ─────────────────────────────────────────────
# MSISDN HASH SECRET
# Plain SHA-256 of a phone number is brute-forceable (the input space is only
# a few million numbers). We HMAC with a server-side secret instead so the
# hash can't be reversed without that secret.
# ─────────────────────────────────────────────

_HASH_SECRET = os.getenv("MSISDN_HASH_SECRET")
if not _HASH_SECRET:
    raise RuntimeError(
        "MSISDN_HASH_SECRET is not set. Generate one with "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"` and add it "
        "to your .env file. This changes the hashing scheme, so it must be set "
        "before you (re)seed the database — existing hashed values won't match "
        "new ones if this changes later."
    )


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# ORM MODELS
# ─────────────────────────────────────────────

class UserProfile(Base):
    __tablename__ = "users_profiles"

    hashed_user_id = Column(String(64), primary_key=True)
    hashed_msisdn = Column(String(64), nullable=False, unique=True)
    kyc_tier = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    behavioral_profile = relationship("BehavioralProfile", back_populates="user", uselist=False)


class BehavioralProfile(Base):
    __tablename__ = "behavioral_profiles"

    profile_id = Column(String(36), primary_key=True)
    hashed_user_id = Column(String(64), ForeignKey("users_profiles.hashed_user_id"), nullable=False)
    avg_30day_amount = Column(Float, default=500.0)
    typical_active_hours = Column(String(50), default="8-20")   # "HH_start-HH_end"
    usual_province_centroid = Column(String(100), default="Lusaka")
    known_beneficiaries_json = Column(Text, default="[]")
    registered_devices_json = Column(Text, default="[]")
    sim_swap_flag = Column(Boolean, default=False)
    last_updated = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserProfile", back_populates="behavioral_profile")


class Transaction(Base):
    __tablename__ = "transactions"

    txn_id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    sender_msisdn = Column(String(64), nullable=False)   # hashed
    receiver_msisdn = Column(String(64), nullable=False)  # hashed
    amount = Column(Float, nullable=False)
    txn_type = Column(String(20), nullable=False)
    operator = Column(String(20), nullable=False)
    channel = Column(String(30), nullable=False)
    province = Column(String(50), nullable=False)
    device_fingerprint = Column(String(64), nullable=True)  # hashed
    is_fraud = Column(Integer, default=0)

    risk_scores = relationship("RiskScoreAudit", back_populates="transaction")

    __table_args__ = (
        # FeatureEngineer._velocity() filters on sender_msisdn + a timestamp
        # range on every single /score call — without this index it's a full
        # table scan per request.
        Index("ix_transactions_sender_ts", "sender_msisdn", "timestamp"),
        Index("ix_transactions_receiver", "receiver_msisdn"),
    )


class RiskScoreAudit(Base):
    __tablename__ = "risk_scores_audit"

    score_id = Column(String(36), primary_key=True)
    txn_id = Column(String(36), ForeignKey("transactions.txn_id"), nullable=False)
    numeric_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    reason_codes_json = Column(Text, default="[]")
    fraud_probability = Column(Float, default=0.0)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="risk_scores")


class SmishingSignal(Base):
    __tablename__ = "smishing_signals"

    signal_id = Column(String(36), primary_key=True)
    hashed_msisdn = Column(String(64), nullable=False)
    detected_at = Column(DateTime, nullable=False)
    signal_type = Column(String(30), default="smishing")


class AgentMetadata(Base):
    __tablename__ = "agent_metadata"

    agent_id = Column(String(36), primary_key=True)
    hashed_msisdn = Column(String(64), nullable=False, unique=True)
    complaint_count = Column(Integer, default=0)
    total_transactions = Column(Integer, default=1)
    complaint_rate = Column(Float, default=0.0)
    province = Column(String(50), nullable=False)


class ReferenceData(Base):
    """
    Admin-managed reference table (blocklist, KYC limits, operator prefixes, etc.).
    Every row is logged; use INSERT-only updates (new row per change) for audit.
    Replaces the in-memory _FRAUD_BLOCKLIST set and hardcoded KYC_LIMITS dict.
    """
    __tablename__ = "reference_data"

    ref_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ref_type = Column(String(30), nullable=False, index=True)  # 'blocklist'|'kyc_limit'|'prefix'|...
    ref_key = Column(String(100), nullable=False)              # e.g. hashed MSISDN or tier number
    ref_value = Column(String(255), nullable=False)            # e.g. '1' (active) or limit amount
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    updated_by = Column(String(50), nullable=False, default="system")
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_reference_data_type_key", "ref_type", "ref_key"),
    )


class ModelVersion(Base):
    """
    Tracks every training run (candidate and production) for the model monitor page.
    Implements the version/deployment-date/rolling-performance view from the report.
    """
    __tablename__ = "model_versions"

    version_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(50), nullable=False)          # e.g. 'RandomForest'
    version_tag = Column(String(30), nullable=False)         # e.g. '2026-08-01-v1'
    trained_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False)               # True = currently in production
    is_candidate = Column(Boolean, default=True)             # True until evaluated
    metrics_json = Column(Text, default="{}")                # precision, recall, f1, auc_roc, mcc
    promoted_by = Column(String(50), nullable=True)
    promotion_reason = Column(Text, nullable=True)           # e.g. 'AUC 0.97 > incumbent 0.95'
    rejection_reason = Column(Text, nullable=True)           # if NFR-03 not met


class RetrainingCorpus(Base):
    """
    Join table collecting analyst-confirmed labels for the next retraining cycle.
    Rows are inserted by overrides.submit_override() and overrides.flag_transaction().
    DatasetBuilder pulls these rows to augment the training set (FR-13).
    """
    __tablename__ = "retraining_corpus"

    corpus_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    txn_id = Column(String(36), nullable=False, index=True)
    confirmed_label = Column(Integer, nullable=True)    # 1=fraud, 0=legitimate, None=pending
    source = Column(String(30), nullable=False)         # 'analyst_override' | 'manual_flag'
    added_at = Column(DateTime, default=datetime.utcnow)
    used_in_retrain = Column(Boolean, default=False)    # flipped to True after consumption


class CustomerNotification(Base):
    """
    Stub for outbound customer alerts (FR gap #8).
    In production, the send_customer_notification() function in alerts.py
    would dispatch via Africa's Talking SMS API / push notification gateway.
    This table provides an audit trail of every attempted notification.
    """
    __tablename__ = "customer_notifications"

    notification_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    txn_id = Column(String(36), nullable=False, index=True)
    hashed_msisdn = Column(String(64), nullable=False)   # recipient (hashed)
    message_type = Column(String(30), nullable=False)    # 'BLOCK_ALERT' | 'STEP_UP_AUTH'
    channel = Column(String(20), default="SMS")          # SMS | PUSH
    status = Column(String(20), default="PENDING")       # PENDING | SENT | FAILED
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# PII UTILITIES
# ─────────────────────────────────────────────

def sha256_hash(value: str) -> str:
    """
    Keyed (HMAC-SHA256) hash for PII anonymisation.
    Kept the name `sha256_hash` for backwards compatibility with callers
    across the codebase — internally this is now HMAC, not plain SHA-256,
    because plain SHA-256 of a phone number is brute-forceable.
    """
    return hmac.new(_HASH_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


# ─────────────────────────────────────────────
# ZAMBIAN DOMAIN CONSTANTS
# ─────────────────────────────────────────────

OPERATOR_PREFIXES = {
    "MTN": ["096", "076"],
    "Airtel": ["097", "077"],
    "Zamtel": ["095"],
}

PROVINCES = [
    "Lusaka", "Copperbelt", "Central", "Southern", "Eastern",
    "Western", "Northern", "Muchinga", "North-Western", "Luapula",
]

# Province approximate centroids (lat, lon)
PROVINCE_CENTROIDS = {
    "Lusaka": (-15.4167, 28.2833),
    "Copperbelt": (-12.8, 28.2),
    "Central": (-14.4, 28.5),
    "Southern": (-16.5, 27.0),
    "Eastern": (-13.5, 32.0),
    "Western": (-15.0, 23.0),
    "Northern": (-10.0, 30.0),
    "Muchinga": (-11.0, 31.5),
    "North-Western": (-12.5, 24.0),
    "Luapula": (-11.5, 29.0),
}

TXN_TYPES = ["P2P", "P2B", "CASHOUT", "CASHIN", "BILLPAY", "INTL_TRANSFER"]
CHANNELS = ["USSD", "APP", "AGENT", "WEB"]

# KYC tier transaction limits (ZMW)
KYC_LIMITS = {1: 1000.0, 2: 5000.0, 3: 50000.0}

# KYC limit cache (tier -> limit)
_kyc_limit_cache: dict[int, float] = {}
_kyc_limit_loaded_at: float = 0.0
_KYC_LIMIT_TTL_SECONDS: float = 300.0  # 5 minutes

def _refresh_kyc_limits_if_stale() -> None:
    """Refresh KYC tier limits from the reference_data table if TTL expired."""
    global _kyc_limit_cache, _kyc_limit_loaded_at
    now = time.monotonic()
    if now - _kyc_limit_loaded_at < _KYC_LIMIT_TTL_SECONDS:
        return
    try:
        with Session(get_engine()) as s:
            rows = (
                s.query(ReferenceData)
                .filter_by(ref_type="kyc_limit", is_active=True)
                .all()
            )
            _kyc_limit_cache = {int(r.ref_key): float(r.ref_value) for r in rows}
            _kyc_limit_loaded_at = now
    except Exception:
        # DB unavailable — keep existing cache
        pass

def get_kyc_limit(kyc_tier: int) -> float:
    """Return the KYC limit for the given tier, using cached DB values."""
    _refresh_kyc_limits_if_stale()
    return _kyc_limit_cache.get(kyc_tier, KYC_LIMITS.get(kyc_tier, 1000.0))


def random_msisdn(operator: str | None = None) -> str:
    """Generate a random Zambian MSISDN."""
    if operator is None:
        operator = random.choice(list(OPERATOR_PREFIXES.keys()))
    prefix = random.choice(OPERATOR_PREFIXES[operator])
    suffix = "".join([str(random.randint(0, 9)) for _ in range(7)])
    return f"+260{prefix[1:]}{suffix}"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in kilometres between two coordinates."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────
# SCHEMA CREATION
# ─────────────────────────────────────────────

def create_schema():
    """Create all tables in the database."""
    # Import modules to register their ORM models before create_all
    import auth
    import alerts
    import overrides
    try:
        import retraining  # noqa: F401  registers no ORM but imports this module to init scheduler
    except ImportError:
        pass

    Base.metadata.create_all(engine)
    print("[db_config] Schema created successfully.")


def seed_reference_data():
    """
    Populate the reference_data table with system defaults.
    Safe to call repeatedly — skips rows that already exist.
    Replaces the hardcoded KYC_LIMITS dict and the in-memory blocklist.
    """
    with Session(engine) as session:
        def _upsert(ref_type, ref_key, ref_value, description=""):
            existing = (
                session.query(ReferenceData)
                .filter_by(ref_type=ref_type, ref_key=ref_key, is_active=True)
                .first()
            )
            if existing:
                return
            session.add(ReferenceData(
                ref_id=str(uuid.uuid4()),
                ref_type=ref_type,
                ref_key=ref_key,
                ref_value=ref_value,
                description=description,
                is_active=True,
                updated_by="system",
                updated_at=datetime.utcnow(),
            ))

        # KYC tier limits (ZMW)
        _upsert("kyc_limit", "1", "1000.0",  "KYC Tier 1 max transaction (ZMW)")
        _upsert("kyc_limit", "2", "5000.0",  "KYC Tier 2 max transaction (ZMW)")
        _upsert("kyc_limit", "3", "50000.0", "KYC Tier 3 max transaction (ZMW)")

        # Operator prefixes
        _upsert("operator_prefix", "MTN",    "096,076", "MTN Zambia MSISDN prefixes")
        _upsert("operator_prefix", "Airtel", "097,077", "Airtel Zambia MSISDN prefixes")
        _upsert("operator_prefix", "Zamtel", "095",     "Zamtel MSISDN prefixes")

        # Provinces
        for prov in PROVINCES:
            _upsert("province", prov, prov, f"Zambia province: {prov}")

        session.commit()
    print("[db_config] Reference data seeded.")


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def _generate_users(n: int = 500) -> list[dict]:
    users = []
    for _ in range(n):
        raw_msisdn = random_msisdn()
        raw_uid = f"USER_{uuid.uuid4().hex[:8]}"
        device = f"DEV_{uuid.uuid4().hex[:12]}"
        kyc = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        province = random.choice(PROVINCES)
        hours_start = random.randint(6, 10)
        hours_end = random.randint(17, 22)

        # Generate 1-5 known beneficiaries
        beneficiaries = [sha256_hash(random_msisdn()) for _ in range(random.randint(1, 5))]

        # 5% sim-swap flag
        sim_swap = random.random() < 0.05

        users.append({
            "raw_uid": raw_uid,
            "hashed_user_id": sha256_hash(raw_uid),
            "raw_msisdn": raw_msisdn,
            "hashed_msisdn": sha256_hash(raw_msisdn),
            "kyc_tier": kyc,
            "avg_30day_amount": round(random.lognormvariate(6.5, 0.8), 2),
            "typical_active_hours": f"{hours_start}-{hours_end}",
            "usual_province_centroid": province,
            "known_beneficiaries_json": json.dumps(beneficiaries),
            "registered_devices_json": json.dumps([sha256_hash(device)]),
            "sim_swap_flag": sim_swap,
            "province": province,
        })
    return users


def _generate_transactions(users: list[dict], n: int = 5000) -> list[dict]:
    """
    Generate synthetic transactions with Zambian fraud typology injection:
    - 60% SIM-swap/account-takeover
    - 20% social engineering (smishing/vishing)
    - 12% agent-based fraud
    - 8% other
    Overall fraud prevalence ≈ 2.5%
    """
    transactions = []
    fraud_count = int(n * 0.025)
    fraud_indices = set(random.sample(range(n), fraud_count))

    now = datetime.utcnow()

    for i in range(n):
        sender = random.choice(users)
        receiver = random.choice(users)
        while receiver["hashed_msisdn"] == sender["hashed_msisdn"]:
            receiver = random.choice(users)

        is_fraud = 1 if i in fraud_indices else 0
        txn_type = random.choice(TXN_TYPES)
        operator = random.choice(list(OPERATOR_PREFIXES.keys()))
        channel = random.choice(CHANNELS)
        province = sender["province"]

        # Time: spread across last 30 days
        ts = now - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        if is_fraud:
            fraud_roll = random.random()
            if fraud_roll < 0.60:           # SIM-swap
                amount = random.uniform(2000, 15000)
                device_fp = sha256_hash(f"UNKNOWN_DEV_{uuid.uuid4().hex}")
            elif fraud_roll < 0.80:         # Social engineering
                amount = random.uniform(500, 8000)
                device_fp = sha256_hash(f"DEV_{uuid.uuid4().hex[:12]}")
            elif fraud_roll < 0.92:         # Agent fraud
                amount = random.uniform(100, 3000)
                channel = "AGENT"
                device_fp = sha256_hash(f"AGENT_DEV_{uuid.uuid4().hex}")
            else:                           # Other
                amount = random.uniform(50, 5000)
                device_fp = sha256_hash(f"DEV_{uuid.uuid4().hex[:12]}")
        else:
            amount = round(abs(random.lognormvariate(6.0, 0.9)), 2)
            device_fp = random.choice(
                json.loads(sender["registered_devices_json"])
                + [sha256_hash(f"DEV_{uuid.uuid4().hex[:12]}")]
            )

        transactions.append({
            "txn_id": str(uuid.uuid4()),
            "timestamp": ts,
            "sender_msisdn": sender["hashed_msisdn"],
            "receiver_msisdn": receiver["hashed_msisdn"],
            "amount": round(amount, 2),
            "txn_type": txn_type,
            "operator": operator,
            "channel": channel,
            "province": province,
            "device_fingerprint": device_fp,
            "is_fraud": is_fraud,
        })

    return transactions


def populate_database(n_users: int = 500, n_transactions: int = 5200):
    """Seed the database with synthetic users, profiles, transactions, and agent data."""
    print(f"[db_config] Generating {n_users} users and {n_transactions} transactions...")

    users = _generate_users(n_users)
    transactions = _generate_transactions(users, n_transactions)

    with Session(engine) as session:
        # Wipe existing data for clean seeding
        session.execute(text("DELETE FROM fraud_overrides"))
        session.execute(text("DELETE FROM manual_flags"))
        session.execute(text("DELETE FROM alert_records"))
        session.execute(text("DELETE FROM dashboard_users"))
        session.execute(text("DELETE FROM risk_scores_audit"))
        session.execute(text("DELETE FROM smishing_signals"))
        session.execute(text("DELETE FROM agent_metadata"))
        session.execute(text("DELETE FROM transactions"))
        session.execute(text("DELETE FROM behavioral_profiles"))
        session.execute(text("DELETE FROM users_profiles"))
        session.commit()
        
        # Seed default users
        import auth
        auth.seed_default_users()

        # Insert users and behavioral profiles
        for u in users:
            up = UserProfile(
                hashed_user_id=u["hashed_user_id"],
                hashed_msisdn=u["hashed_msisdn"],
                kyc_tier=u["kyc_tier"],
                created_at=datetime.utcnow() - timedelta(days=random.randint(30, 365)),
            )
            session.add(up)

            bp = BehavioralProfile(
                profile_id=str(uuid.uuid4()),
                hashed_user_id=u["hashed_user_id"],
                avg_30day_amount=u["avg_30day_amount"],
                typical_active_hours=u["typical_active_hours"],
                usual_province_centroid=u["usual_province_centroid"],
                known_beneficiaries_json=u["known_beneficiaries_json"],
                registered_devices_json=u["registered_devices_json"],
                sim_swap_flag=u["sim_swap_flag"],
                last_updated=datetime.utcnow(),
            )
            session.add(bp)

        session.commit()

        # Insert transactions in batches
        batch_size = 500
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i: i + batch_size]
            for t in batch:
                session.add(Transaction(**t))
            session.commit()

        # Generate smishing signals (linked to ~15% of fraud txns)
        fraud_txns = [t for t in transactions if t["is_fraud"] == 1]
        for ft in random.sample(fraud_txns, max(1, int(len(fraud_txns) * 0.15))):
            session.add(SmishingSignal(
                signal_id=str(uuid.uuid4()),
                hashed_msisdn=ft["sender_msisdn"],
                detected_at=ft["timestamp"] - timedelta(minutes=random.randint(5, 25)),
                signal_type="smishing",
            ))
        session.commit()

        # Generate agent metadata
        agent_msisdns = list({t["sender_msisdn"] for t in transactions if t.get("channel") == "AGENT"})
        for msisdn in agent_msisdns[:100]:
            total = random.randint(10, 500)
            complaints = random.randint(0, max(1, int(total * random.uniform(0, 0.15))))
            session.add(AgentMetadata(
                agent_id=str(uuid.uuid4()),
                hashed_msisdn=msisdn,
                complaint_count=complaints,
                total_transactions=total,
                complaint_rate=round(complaints / total, 4),
                province=random.choice(PROVINCES),
            ))
        session.commit()

    print(f"[db_config] Database populated: {len(users)} users, {len(transactions)} transactions.")
    return users, transactions


_sqlite_initialized = False

def get_engine():
    global _sqlite_initialized
    if DB_TYPE == "SQLite fallback" and not _sqlite_initialized:
        _sqlite_initialized = True
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sqlite_path = os.path.join(base_dir, "fraud_system.db")
        if not os.path.exists(sqlite_path) or os.path.getsize(sqlite_path) == 0:
            print("[db_config] Initializing SQLite fallback database...")
            create_schema()
            populate_database()
    return engine


if __name__ == "__main__":
    create_schema()
    populate_database()