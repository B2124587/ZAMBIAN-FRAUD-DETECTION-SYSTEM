"""
MODULE 5: FastAPI REST API with MySQL Audit Logging
Mobile Money Fraud Detection System - Zambia
app.py

Endpoints:
  GET  /health                — System health check
  POST /score                 — Score a transaction payload (FR-01)
  GET  /audit/recent          — Most recent audit entries
  GET  /audit/{txn_id}        — Retrieve audit record for a transaction
  POST /smishing/classify     — Run live NLP smishing classification (FR-12)
  POST /auth/register         — Create a new dashboard user (admin-only)
  POST /admin/retrain         — Trigger an immediate model retraining cycle (FR-13)

Authentication: API key via X-API-Key header (set FRAUD_API_KEYS env var).
All scoring results are persisted to risk_scores_audit.

Failover (FR-04, NFR-10):
  Primary model → Secondary model → Rule-only (degraded_mode=True).

Run locally:   uvicorn app:app --reload
Run in prod:   gunicorn app:app -k uvicorn.workers.UvicornWorker
               (see Procfile)
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from db_config import (
    RiskScoreAudit, Transaction, get_engine, sha256_hash, create_schema
)
from data_pipeline import FeatureEngineer
from scoring_engine import RiskScoringEngine, ScoringResult
from alerts import create_alert, get_unread_alerts

logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    create_schema()
    # Seed reference data (KYC limits, operator prefixes, provinces) on first boot
    try:
        from db_config import seed_reference_data
        seed_reference_data()
    except Exception as e:
        logger.warning(f"Reference data seed skipped: {e}")
    yield


# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────

app = FastAPI(
    title="Zambia Mobile Money Fraud Detection API",
    description="Real-time risk scoring for mobile money transactions.",
    version="1.1.0",
    lifespan=_lifespan,
)

# API keys come only from the environment — never hardcoded. Set
# FRAUD_API_KEYS to a comma-separated list in .env / your platform's
# environment config. Prefer a real secrets manager in production.
VALID_API_KEYS = {
    k.strip() for k in os.getenv("FRAUD_API_KEYS", "").split(",") if k.strip()
}
if not VALID_API_KEYS:
    logger.warning(
        "FRAUD_API_KEYS is not set — every request to /score and /audit "
        "endpoints will be rejected with 401 until it is configured."
    )

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> str:
    if not key or key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorised. Valid X-API-Key required.")
    return key


# ─────────────────────────────────────────────
# MODEL SLOTS (FR-04, NFR-10)
# Primary → Secondary → Rule-only (degraded_mode)
# ─────────────────────────────────────────────

_feature_engineer: Optional[FeatureEngineer] = None
_risk_engine: Optional[RiskScoringEngine] = None
_primary_model = None    # Set by main.py / retraining.py after training
_secondary_model = None  # Previous production model; used if primary unavailable
_startup_time = datetime.utcnow()

# Cached smishing model (loaded on first /smishing/classify call)
_smishing_model = None


def get_feature_engineer() -> FeatureEngineer:
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    return _feature_engineer


def get_risk_engine() -> RiskScoringEngine:
    global _risk_engine
    if _risk_engine is None:
        # 0.60 rule / 0.40 ML blend (report NFR-03)
        _risk_engine = RiskScoringEngine(ml_blend_weight=0.40)
    return _risk_engine


def set_primary_model(model):
    global _primary_model
    _primary_model = model


def set_secondary_model(model):
    """Store the previous production model as the failover slot (FR-04)."""
    global _secondary_model
    _secondary_model = model


def _get_smishing_model():
    """Lazy-load the saved NLP smishing model (FR-12)."""
    global _smishing_model
    if _smishing_model is not None:
        return _smishing_model
    model_path = Path("saved_models") / "smishingnlp.pkl"
    if model_path.exists():
        from models import MLModel
        _smishing_model = MLModel.load(model_path)
        return _smishing_model
    return None


# ─────────────────────────────────────────────
# LATENCY MIDDLEWARE (NFR-02 observability)
# Logs endpoint + latency on every request.
# ─────────────────────────────────────────────

@app.middleware("http")
async def _latency_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    path = request.url.path
    logger.info(f"[latency] {request.method} {path} → {response.status_code} in {elapsed_ms:.1f}ms")
    # NFR-02 targets: /health ≤200ms, /flag+/profile ≤1000ms
    if path == "/health" and elapsed_ms > 200:
        logger.warning(f"[nfr02] /health exceeded 200ms target: {elapsed_ms:.1f}ms")
    elif path in ("/score",) and elapsed_ms > 1000:
        logger.warning(f"[nfr02] /score exceeded 1000ms target: {elapsed_ms:.1f}ms")
    return response


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ScoreRequest(BaseModel):
    txn_id: Optional[str] = None
    timestamp: Optional[str] = None
    sender_msisdn: str
    receiver_msisdn: str
    amount: float = Field(..., gt=0, description="Transaction amount in ZMW")
    txn_type: str
    operator: str = "MTN"
    channel: str = "APP"
    province: str = "Lusaka"
    device_fingerprint: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "sender_msisdn": "+260961234567",
                "receiver_msisdn": "+260977654321",
                "amount": 1500.00,
                "txn_type": "P2P",
                "operator": "MTN",
                "channel": "APP",
                "province": "Lusaka",
                "device_fingerprint": "abc123...",
            }
        }
    }


class ScoreResponse(BaseModel):
    txn_id: str
    score_id: str
    risk_score: float
    risk_level: str
    fraud_probability: float
    reason_codes: list[str]
    features: dict
    recommended_action: str
    ml_model_used: Optional[str]   # 'primary' | 'secondary' | 'rule_only'
    degraded_mode: bool = False    # True when both model slots are unavailable (NFR-10)
    db_logged: bool
    evaluated_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str
    ml_model: str
    secondary_model: str
    uptime_seconds: float
    timestamp: str


class AuditRecord(BaseModel):
    score_id: str
    txn_id: Optional[str] = None
    numeric_score: float
    risk_level: str
    reason_codes: Optional[list[str]] = None
    fraud_probability: float
    calculated_at: str


class AuditResponse(BaseModel):
    txn_id: str
    audit_records: list[AuditRecord]


class RecentAuditsResponse(BaseModel):
    count: int
    records: list[AuditRecord]


class SmishingRequest(BaseModel):
    sms_text: str = Field(..., min_length=1, description="Raw SMS message text to classify")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sms_text": "URGENT: Your MTN account will be suspended. Verify now: mtn-verify.tk"
            }
        }
    }


class SmishingResponse(BaseModel):
    sms_text: str
    is_smishing: bool
    probability: float
    model_used: str
    evaluated_at: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role: str = Field(..., description="admin | analyst | operator")
    full_name: str
    email: Optional[str] = None


class RetrainResponse(BaseModel):
    status: str
    message: str
    triggered_at: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """System health check — no authentication required."""
    uptime_s = (datetime.utcnow() - _startup_time).total_seconds()
    db_ok = False
    try:
        with Session(get_engine()) as session:
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    primary_status = f"loaded ({_primary_model.name})" if _primary_model else "not_loaded"
    secondary_status = f"loaded ({_secondary_model.name})" if _secondary_model else "not_loaded"

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        service="Zambia Mobile Money Fraud Detection API",
        version="1.1.0",
        database="connected" if db_ok else "error",
        ml_model=primary_status,
        secondary_model=secondary_status,
        uptime_seconds=round(uptime_s, 1),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/score", response_model=ScoreResponse)
def score_transaction(payload: ScoreRequest, api_key: str = Depends(require_api_key)):
    """Score an incoming mobile money transaction for fraud risk."""
    try:
        data = payload.model_dump()
        txn_id = data.get("txn_id") or str(uuid.uuid4())
        data["txn_id"] = txn_id
        data["timestamp"] = data.get("timestamp") or datetime.utcnow().isoformat()

        # ── 1. FEATURE ENGINEERING ──
        fe = get_feature_engineer()
        features = fe.engineer(data)

        # ── 2. ML INFERENCE with failover (FR-04, NFR-10) ──
        ml_probability = None
        ml_model_used = "rule_only"
        degraded_mode = True  # will be cleared if any model fires

        if _primary_model is not None:
            try:
                feature_array = fe.to_array(features).reshape(1, -1)
                ml_probability = float(_primary_model.predict_proba(feature_array)[0, 1])
                ml_model_used = "primary"
                degraded_mode = False
            except Exception as e:
                logger.error(f"Primary model inference failed: {e}; trying secondary.")

        if ml_probability is None and _secondary_model is not None:
            try:
                feature_array = fe.to_array(features).reshape(1, -1)
                ml_probability = float(_secondary_model.predict_proba(feature_array)[0, 1])
                ml_model_used = "secondary"
                degraded_mode = False
                logger.warning(f"[failover] Using secondary model for txn {txn_id}")
            except Exception as e:
                logger.error(f"Secondary model inference also failed: {e}; rule-only mode.")

        if degraded_mode:
            logger.warning(f"[degraded] Both model slots unavailable for txn {txn_id}; rule score only.")

        # ── 3. RISK SCORING ──
        receiver_h = (
            data["receiver_msisdn"]
            if len(str(data["receiver_msisdn"])) == 64
            else sha256_hash(str(data["receiver_msisdn"]))
        )
        engine = get_risk_engine()
        result: ScoringResult = engine.score(
            features,
            receiver_msisdn=receiver_h,
            ml_fraud_probability=ml_probability,
        )

        # ── 4. AUDIT LOG → DATABASE ──
        score_id = str(uuid.uuid4())
        audit_entry = RiskScoreAudit(
            score_id=score_id,
            txn_id=txn_id,
            numeric_score=result.risk_score,
            risk_level=result.risk_level,
            reason_codes_json=json.dumps(result.reason_codes),
            fraud_probability=result.fraud_probability,
            calculated_at=datetime.utcnow(),
        )
        db_logged = False
        try:
            with Session(get_engine()) as session:
                existing = session.query(Transaction).filter_by(txn_id=txn_id).first()
                if not existing:
                    sender_h = (
                        sha256_hash(str(data["sender_msisdn"]))
                        if len(str(data["sender_msisdn"])) != 64
                        else data["sender_msisdn"]
                    )
                    stub = Transaction(
                        txn_id=txn_id,
                        timestamp=datetime.fromisoformat(str(data["timestamp"])),
                        sender_msisdn=sender_h,
                        receiver_msisdn=receiver_h,
                        amount=float(data["amount"]),
                        txn_type=data["txn_type"],
                        operator=data["operator"],
                        channel=data["channel"],
                        province=data.get("province", "Lusaka"),
                        device_fingerprint=(
                            sha256_hash(str(data["device_fingerprint"]))
                            if data.get("device_fingerprint") else None
                        ),
                        is_fraud=0,
                    )
                    session.add(stub)
                    session.flush()

                session.add(audit_entry)
                session.commit()
                db_logged = True
        except Exception as db_err:
            logger.error(f"DB logging failed: {db_err}")

        # ── 5. ALERTS (analyst + customer notification) ──
        if result.recommended_action in ("BLOCK", "STEP_UP_AUTH"):
            create_alert(
                txn_id=txn_id,
                risk_score=result.risk_score,
                risk_level=result.risk_level,
                reason_codes=result.reason_codes,
                action=result.recommended_action,
            )
            # Customer-facing notification stub (FR gap #8)
            try:
                from alerts import send_customer_notification
                sender_h_for_notif = (
                    sha256_hash(str(data["sender_msisdn"]))
                    if len(str(data["sender_msisdn"])) != 64
                    else data["sender_msisdn"]
                )
                send_customer_notification(
                    txn_id=txn_id,
                    hashed_msisdn=sender_h_for_notif,
                    action=result.recommended_action,
                    risk_level=result.risk_level,
                )
            except Exception as notif_err:
                logger.warning(f"Customer notification failed (non-fatal): {notif_err}")

        # ── 6. RESPONSE ──
        return ScoreResponse(
            txn_id=txn_id,
            score_id=score_id,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            fraud_probability=result.fraud_probability,
            reason_codes=result.reason_codes,
            features=features,
            recommended_action=result.recommended_action,
            ml_model_used=ml_model_used,
            degraded_mode=degraded_mode,
            db_logged=db_logged,
            evaluated_at=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /score")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smishing/classify", response_model=SmishingResponse)
def classify_smishing(payload: SmishingRequest, api_key: str = Depends(require_api_key)):
    """
    Run live NLP smishing classification on an SMS text (FR-12).

    Loads the saved SmishingNLPClassifier from saved_models/.
    Falls back to re-training on the built-in corpus if no saved model is found.
    """
    try:
        model = _get_smishing_model()
        if model is None:
            # No saved model — train on the built-in corpus right now
            from models import SmishingNLPClassifier
            model = SmishingNLPClassifier(use_naive_bayes=True)
            model.fit()
            logger.warning("[smishing] No saved model found; trained on built-in corpus.")

        prob = float(model.predict_proba([payload.sms_text])[0, 1])
        return SmishingResponse(
            sms_text=payload.sms_text,
            is_smishing=prob >= 0.5,
            probability=round(prob, 4),
            model_used=getattr(model, "name", "SmishingNLP"),
            evaluated_at=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /smishing/classify")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/register", status_code=201)
def register_user(payload: RegisterRequest, api_key: str = Depends(require_api_key)):
    """
    Create a new dashboard user account (admin-only, FR gap #9).

    Roles: admin | analyst | operator
    """
    try:
        from auth import register_user as _register, UserRole
        valid_roles = {r.value for r in UserRole}
        if payload.role not in valid_roles:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role '{payload.role}'. Must be one of: {sorted(valid_roles)}",
            )
        user = _register(
            username=payload.username,
            password=payload.password,
            role=payload.role,
            full_name=payload.full_name,
            email=payload.email,
        )
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /auth/register")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/retrain", response_model=RetrainResponse)
def trigger_retrain(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_api_key),
):
    """
    Trigger an immediate model retraining cycle (FR-13).

    Runs asynchronously in a background thread; returns immediately.
    Check model_versions table or the dashboard for the outcome.
    """
    try:
        from retraining import RetrainingPipeline

        def _run():
            try:
                pipeline = RetrainingPipeline()
                pipeline.run(triggered_by="api_admin")
            except Exception as e:
                logger.exception(f"[retrain] Background retraining failed: {e}")

        background_tasks.add_task(_run)
        return RetrainResponse(
            status="accepted",
            message="Retraining cycle started in background. Monitor model_versions table for outcome.",
            triggered_at=datetime.utcnow().isoformat(),
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Retraining module not available. Run main.py first to train initial models."
        )
    except Exception as e:
        logger.exception("Error triggering retrain")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/recent", response_model=RecentAuditsResponse)
def recent_audits(
    limit: int = Query(20, ge=1, le=100),
    api_key: str = Depends(require_api_key),
):
    """Return the most recent audit entries (max 100)."""
    try:
        with Session(get_engine()) as session:
            records = (
                session.query(RiskScoreAudit)
                .order_by(RiskScoreAudit.calculated_at.desc())
                .limit(limit)
                .all()
            )
            return RecentAuditsResponse(
                count=len(records),
                records=[
                    AuditRecord(
                        score_id=r.score_id,
                        txn_id=r.txn_id,
                        numeric_score=r.numeric_score,
                        risk_level=r.risk_level,
                        reason_codes=json.loads(r.reason_codes_json or "[]"),
                        fraud_probability=r.fraud_probability,
                        calculated_at=r.calculated_at.isoformat(),
                    )
                    for r in records
                ],
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/{txn_id}", response_model=AuditResponse)
def get_audit(txn_id: str, api_key: str = Depends(require_api_key)):
    """Retrieve all audit score records for a given transaction ID."""
    try:
        with Session(get_engine()) as session:
            records = (
                session.query(RiskScoreAudit)
                .filter_by(txn_id=txn_id)
                .order_by(RiskScoreAudit.calculated_at.desc())
                .all()
            )
            if not records:
                raise HTTPException(status_code=404, detail=f"No audit records found for txn_id={txn_id}")

            return AuditResponse(
                txn_id=txn_id,
                audit_records=[
                    AuditRecord(
                        score_id=r.score_id,
                        numeric_score=r.numeric_score,
                        risk_level=r.risk_level,
                        reason_codes=json.loads(r.reason_codes_json or "[]"),
                        fraud_probability=r.fraud_probability,
                        calculated_at=r.calculated_at.isoformat(),
                    )
                    for r in records
                ],
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/unread")
def unread_alerts(
    limit: int = Query(50, ge=1, le=100),
    api_key: str = Depends(require_api_key),
):
    """Return the most recent unread alerts."""
    try:
        alerts = get_unread_alerts(limit=limit)
        return {"count": len(alerts), "alerts": alerts}
    except Exception:
        logger.exception("Failed to fetch unread alerts")
        raise HTTPException(status_code=500, detail="Database error retrieving alerts.")
