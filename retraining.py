"""
MODULE 11: Model Retraining Pipeline (FR-13)
Mobile Money Fraud Detection System - Zambia
retraining.py

Implements the "Retrain Machine Learning Model" use case (Table F.4):

  Trigger:
    - Monthly schedule: 1st of every month at 02:00 (APScheduler BackgroundScheduler)
    - On-demand:        POST /admin/retrain  or  RetrainingPipeline().run()

  Cycle:
    1. Build augmented dataset (historical transactions + analyst-confirmed corpus rows)
    2. Train a fresh candidate model (same architecture as current primary)
    3. Evaluate candidate on held-out test partition
    4. Compare against NFR-03 thresholds:
           precision ≥ 0.85,  recall ≥ 0.80,  AUC-ROC ≥ 0.90
    5a. If met  → promote: current primary becomes secondary, candidate becomes primary.
                  Mark corpus rows as used.  Write ModelVersion row (is_active=True).
    5b. If not  → reject: keep current production model.
                  Write ModelVersion row (is_candidate=True, rejection_reason=...).
    6.  Record outcome in risk_scores_audit as a system-generated entry.

  Model slot rotation:
    _secondary_model ← current _primary_model
    _primary_model   ← new candidate

  NFR-03 targets (promotion gate):
    precision ≥ 0.85  ·  recall ≥ 0.80  ·  AUC-ROC ≥ 0.90
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fraud_retrain")


# ─────────────────────────────────────────────
# NFR-03 THRESHOLDS
# ─────────────────────────────────────────────

NFR03_PRECISION_MIN = 0.85
NFR03_RECALL_MIN    = 0.80
NFR03_AUC_MIN       = 0.90

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# RETRAINING PIPELINE
# ─────────────────────────────────────────────

class RetrainingPipeline:
    """
    Orchestrates a full model retraining cycle.

    Designed to be called:
      - By the APScheduler monthly cron (start_scheduler())
      - By the FastAPI /admin/retrain endpoint
      - Directly from tests / admin scripts
    """

    def __init__(self, model_class_name: str = "RandomForest"):
        """
        model_class_name: which model architecture to use for the candidate.
        Defaults to 'RandomForest' (primary model in the report).
        """
        self.model_class_name = model_class_name

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def run(self, triggered_by: str = "scheduler") -> dict:
        """
        Execute a full retraining cycle.

        Returns a summary dict with keys:
            promoted, version_id, metrics, reason
        """
        logger.info(f"[retrain] Cycle started (triggered_by={triggered_by})")
        version_tag = datetime.utcnow().strftime("%Y-%m-%d-v1")
        version_id = str(uuid.uuid4())

        # ── 1. Build augmented dataset ──
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = self._build_dataset()
        except Exception as e:
            msg = f"Dataset build failed: {e}"
            logger.error(f"[retrain] {msg}")
            self._write_model_version(
                version_id=version_id,
                model_name=self.model_class_name,
                version_tag=version_tag,
                metrics={},
                promoted=False,
                promoted_by=triggered_by,
                rejection_reason=msg,
            )
            self._record_audit_entry(
                outcome=f"RETRAIN_FAILED: {msg}",
                triggered_by=triggered_by,
            )
            return {"promoted": False, "version_id": version_id, "metrics": {}, "reason": msg}

        # ── 2. Train candidate model ──
        try:
            candidate = self._train_candidate(X_train, y_train)
        except Exception as e:
            msg = f"Training failed: {e}"
            logger.error(f"[retrain] {msg}")
            self._write_model_version(
                version_id=version_id,
                model_name=self.model_class_name,
                version_tag=version_tag,
                metrics={},
                promoted=False,
                promoted_by=triggered_by,
                rejection_reason=msg,
            )
            self._record_audit_entry(outcome=f"RETRAIN_FAILED: {msg}", triggered_by=triggered_by)
            return {"promoted": False, "version_id": version_id, "metrics": {}, "reason": msg}

        # ── 3. Evaluate on test partition ──
        metrics = self._evaluate_candidate(candidate, X_test, y_test)
        logger.info(f"[retrain] Candidate metrics: {metrics}")

        # ── 4. Compare against NFR-03 ──
        promoted, reason = self._meets_nfr03(metrics)

        # ── 5a / 5b. Promote or reject ──
        if promoted:
            self._promote(
                candidate=candidate,
                version_id=version_id,
                version_tag=version_tag,
                metrics=metrics,
                promoted_by=triggered_by,
                reason=reason,
            )
            outcome_msg = f"RETRAIN_PROMOTED: {reason}"
        else:
            logger.warning(f"[retrain] Candidate rejected: {reason}")
            outcome_msg = f"RETRAIN_REJECTED: {reason}"

        # Always write a ModelVersion row regardless of outcome
        self._write_model_version(
            version_id=version_id,
            model_name=self.model_class_name,
            version_tag=version_tag,
            metrics=metrics,
            promoted=promoted,
            promoted_by=triggered_by,
            promotion_reason=reason if promoted else None,
            rejection_reason=reason if not promoted else None,
        )

        # ── 6. Record in audit log ──
        self._record_audit_entry(outcome=outcome_msg, triggered_by=triggered_by)

        logger.info(f"[retrain] Cycle complete. promoted={promoted}, reason={reason}")
        return {
            "promoted": promoted,
            "version_id": version_id,
            "metrics": metrics,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _build_dataset(self):
        from data_pipeline import DatasetBuilder
        builder = DatasetBuilder()
        # include_corpus=True appends analyst-confirmed labels (FR-13)
        return builder.build(include_corpus=True)

    def _train_candidate(self, X_train, y_train):
        from models import RandomForestFraudModel, XGBoostFraudModel, LogisticRegressionFraudModel
        registry = {
            "RandomForest": RandomForestFraudModel,
            "XGBoost": XGBoostFraudModel,
            "LogisticRegression": LogisticRegressionFraudModel,
        }
        ModelClass = registry.get(self.model_class_name, RandomForestFraudModel)
        candidate = ModelClass()
        logger.info(f"[retrain] Training candidate {candidate.name}...")
        candidate.fit(X_train, y_train)
        return candidate

    def _evaluate_candidate(self, candidate, X_test, y_test) -> dict:
        """Evaluate model on the held-out test partition and return metrics dict."""
        # Use the model-specific threshold if defined in main.py, else 0.40
        THRESHOLDS_LOCAL = {"RandomForest": 0.35, "XGBoost": 0.35, "LogisticRegression": 0.40}
        thr = THRESHOLDS_LOCAL.get(self.model_class_name, 0.40)
        return candidate.evaluate(X_test, y_test, threshold=thr)

    def _meets_nfr03(self, metrics: dict) -> tuple[bool, str]:
        """
        Check NFR-03 promotion gate.

        Returns (True, reason_string) if all thresholds are met,
        (False, reason_string) if any threshold fails.
        """
        failures = []
        if metrics.get("precision", 0) < NFR03_PRECISION_MIN:
            failures.append(
                f"precision {metrics.get('precision', 0):.3f} < {NFR03_PRECISION_MIN}"
            )
        if metrics.get("recall", 0) < NFR03_RECALL_MIN:
            failures.append(
                f"recall {metrics.get('recall', 0):.3f} < {NFR03_RECALL_MIN}"
            )
        if metrics.get("auc_roc", 0) < NFR03_AUC_MIN:
            failures.append(
                f"AUC-ROC {metrics.get('auc_roc', 0):.3f} < {NFR03_AUC_MIN}"
            )

        if failures:
            return False, "NFR-03 thresholds not met: " + "; ".join(failures)

        return True, (
            f"NFR-03 cleared — precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, AUC={metrics['auc_roc']:.3f}"
        )

    def _promote(
        self,
        candidate,
        version_id: str,
        version_tag: str,
        metrics: dict,
        promoted_by: str,
        reason: str,
    ) -> None:
        """
        Rotate model slots and persist the new candidate.

        Current primary → secondary slot
        Candidate       → primary slot
        """
        # Save candidate to disk
        candidate_path = MODEL_DIR / f"retrained_{self.model_class_name.lower()}_{version_tag}.pkl"
        candidate.save(path=candidate_path)

        # Rotate live model slots in app module
        try:
            import app as _app
            old_primary = _app._primary_model
            _app.set_secondary_model(old_primary)   # Failover slot
            _app.set_primary_model(candidate)        # New production model
            logger.info(
                f"[retrain] Model slots rotated: "
                f"primary={candidate.name}, secondary={getattr(old_primary, 'name', 'None')}"
            )
        except Exception as e:
            logger.warning(f"[retrain] Could not update live model slots (API may not be running): {e}")

        # Mark corpus rows as consumed so they're not re-used next cycle
        try:
            from data_pipeline import RetrainingCorpusBuilder
            n_marked = RetrainingCorpusBuilder().mark_used()
            logger.info(f"[retrain] Marked {n_marked} corpus rows as used.")
        except Exception as e:
            logger.warning(f"[retrain] Could not mark corpus rows as used: {e}")

    def _write_model_version(
        self,
        version_id: str,
        model_name: str,
        version_tag: str,
        metrics: dict,
        promoted: bool,
        promoted_by: str,
        promotion_reason: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Persist a ModelVersion row for auditability."""
        try:
            from db_config import ModelVersion, get_engine
            from sqlalchemy.orm import Session
            with Session(get_engine()) as s:
                # Deactivate previous active version if promoting
                if promoted:
                    prev = s.query(ModelVersion).filter_by(is_active=True).all()
                    for p in prev:
                        p.is_active = False

                mv = ModelVersion(
                    version_id=version_id,
                    model_name=model_name,
                    version_tag=version_tag,
                    trained_at=datetime.utcnow(),
                    deployed_at=datetime.utcnow() if promoted else None,
                    is_active=promoted,
                    is_candidate=not promoted,
                    metrics_json=json.dumps(metrics),
                    promoted_by=promoted_by if promoted else None,
                    promotion_reason=promotion_reason,
                    rejection_reason=rejection_reason,
                )
                s.add(mv)
                s.commit()
        except Exception as e:
            logger.error(f"[retrain] Failed to write ModelVersion row: {e}")

    def _record_audit_entry(self, outcome: str, triggered_by: str) -> None:
        """Append a system-generated entry to risk_scores_audit recording the retrain outcome."""
        try:
            from db_config import RiskScoreAudit, Transaction, get_engine
            from sqlalchemy.orm import Session
            # We need a valid txn_id — use a synthetic sentinel transaction
            sentinel_txn_id = f"RETRAIN-{uuid.uuid4().hex[:28]}"
            with Session(get_engine()) as s:
                # Insert a sentinel Transaction row so the FK constraint is satisfied
                sentinel = Transaction(
                    txn_id=sentinel_txn_id,
                    timestamp=datetime.utcnow(),
                    sender_msisdn="system",
                    receiver_msisdn="system",
                    amount=0.0,
                    txn_type="RETRAIN",
                    operator="SYSTEM",
                    channel="SYSTEM",
                    province="Lusaka",
                    is_fraud=0,
                )
                s.add(sentinel)
                s.flush()

                audit = RiskScoreAudit(
                    score_id=str(uuid.uuid4()),
                    txn_id=sentinel_txn_id,
                    numeric_score=0.0,
                    risk_level="System",
                    reason_codes_json=json.dumps([
                        f"SYSTEM: {outcome}",
                        f"Triggered by: {triggered_by}",
                        f"At: {datetime.utcnow().isoformat()}",
                    ]),
                    fraud_probability=0.0,
                    calculated_at=datetime.utcnow(),
                )
                s.add(audit)
                s.commit()
        except Exception as e:
            logger.error(f"[retrain] Failed to record audit entry: {e}")


# ─────────────────────────────────────────────
# MONTHLY SCHEDULER (APScheduler)
# Trigger: 1st of every month at 02:00 local time
# ─────────────────────────────────────────────

_scheduler = None


def start_scheduler() -> None:
    """
    Start the APScheduler BackgroundScheduler with a monthly retraining cron.

    Safe to call multiple times — no-ops if the scheduler is already running.
    Called from main.py after initial training.
    """
    global _scheduler

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "[retrain] APScheduler not installed — monthly scheduler disabled. "
            "Install with: pip install apscheduler>=3.10"
        )
        return

    if _scheduler is not None and _scheduler.running:
        logger.info("[retrain] Scheduler already running — skipping re-init.")
        return

    _scheduler = BackgroundScheduler(daemon=True)

    def _monthly_retrain():
        logger.info("[retrain] Monthly scheduler fired — starting retraining cycle.")
        try:
            pipeline = RetrainingPipeline()
            result = pipeline.run(triggered_by="monthly_scheduler")
            logger.info(f"[retrain] Monthly cycle result: {result}")
        except Exception as exc:
            logger.exception(f"[retrain] Monthly retraining cycle raised an exception: {exc}")

    # Cron: 1st of every month at 02:00
    _scheduler.add_job(
        _monthly_retrain,
        CronTrigger(day=1, hour=2, minute=0),
        id="monthly_retrain",
        name="Monthly Model Retraining",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,  # Allow up to 1h late if server was down
    )

    _scheduler.start()
    logger.info("[retrain] APScheduler started — monthly retraining cron active (day=1, hour=2).")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler (called on application teardown)."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[retrain] Scheduler stopped.")


# ─────────────────────────────────────────────
# CLI QUICK-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("Running a manual retraining cycle...")
    pipeline = RetrainingPipeline()
    result = pipeline.run(triggered_by="manual_cli")
    print(f"\nResult: promoted={result['promoted']}")
    print(f"Reason: {result['reason']}")
    print(f"Metrics: {result['metrics']}")
    sys.exit(0 if result["promoted"] else 1)
