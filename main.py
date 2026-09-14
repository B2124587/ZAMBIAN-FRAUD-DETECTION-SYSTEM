"""
MODULE 6: Evaluation & Execution Runner
Mobile Money Fraud Detection System - Zambia
main.py

End-to-end orchestration:
  1. Database setup & synthetic data generation
  2. Feature engineering & dataset preparation
  3. Model training (RF, XGBoost, LR, NLP)
  4. Comprehensive evaluation matrix
  5. Top model selection
  6. Live API test via FastAPI test client
  7. Display saved MySQL/SQLite entries
"""

import sys
import traceback
from datetime import datetime

from sqlalchemy.orm import Session

# ─────────────────────────────────────────────
# STEP 0: ENVIRONMENT BOOTSTRAP
# ─────────────────────────────────────────────

def _banner(title: str, char: str = "═"):
    w = 62
    print(f"\n{'':>1}{char * w}")
    print(f"  {title.upper()}")
    print(f"{'':>1}{char * w}")


# ─────────────────────────────────────────────
# STEP 1: DB SETUP
# ─────────────────────────────────────────────

def setup_database():
    _banner("Step 1 · Database Setup & Data Generation")
    from db_config import create_schema, populate_database
    create_schema()
    users, transactions = populate_database(n_users=500, n_transactions=5200)
    fraud_count = sum(1 for t in transactions if t["is_fraud"] == 1)
    print(f"  ► Users:        {len(users)}")
    print(f"  ► Transactions: {len(transactions)}")
    print(f"  ► Fraud cases:  {fraud_count} ({fraud_count/len(transactions):.2%})")
    return users, transactions


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING & DATASET SPLIT
# ─────────────────────────────────────────────

def build_dataset():
    _banner("Step 2 · Feature Engineering & Dataset Split (70/15/15)")
    from data_pipeline import DatasetBuilder, FEATURE_NAMES
    builder = DatasetBuilder()
    X_train, X_val, X_test, y_train, y_val, y_test = builder.build()
    print(f"  ► Features:          {len(FEATURE_NAMES)}")
    print(f"  ► Train set:         {X_train.shape}  fraud={y_train.sum()}")
    print(f"  ► Validation set:    {X_val.shape}  fraud={y_val.sum()}")
    print(f"  ► Test set:          {X_test.shape}  fraud={y_test.sum()}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# STEP 3: MODEL TRAINING
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    _banner("Step 3 · Model Training")
    from models import get_all_models, SmishingNLPClassifier

    trained = []
    for model in get_all_models():
        model.fit(X_train, y_train)
        model.save()
        trained.append(model)

    # Train NLP classifier (self-contained corpus)
    smishing_clf = SmishingNLPClassifier(use_naive_bayes=True)
    smishing_clf.fit()
    smishing_clf.save()
    print(f"  ► Trained {len(trained)} tabular models + 1 NLP smishing classifier")
    return trained, smishing_clf


# ─────────────────────────────────────────────
# STEP 4: EVALUATION MATRIX
# ─────────────────────────────────────────────

METRIC_COLS = ["model", "accuracy", "precision", "recall", "f1", "auc_roc", "mcc"]
THRESHOLDS = {"RandomForest": 0.35, "XGBoost": 0.35, "LogisticRegression": 0.40}


def _print_evaluation_table(results: list[dict]):
    col_w = [20, 10, 10, 10, 10, 10, 10]
    header = "".join(h.ljust(w) for h, w in zip(METRIC_COLS, col_w))
    sep = "─" * sum(col_w)
    print(f"\n  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for r in results:
        row = "".join(str(r.get(c, "—")).ljust(w) for c, w in zip(METRIC_COLS, col_w))
        print(f"  {row}")
    print(f"  {sep}")


def evaluate_models(trained_models, X_val, y_val, X_test, y_test):
    _banner("Step 4 · Evaluation Matrix (Test Set)")
    val_results, test_results = [], []

    for model in trained_models:
        thr = THRESHOLDS.get(model.name, 0.40)
        val_m = model.evaluate(X_val, y_val, threshold=thr)
        test_m = model.evaluate(X_test, y_test, threshold=thr)
        val_results.append(val_m)
        test_results.append(test_m)

    print("\n  ── VALIDATION SET ──")
    _print_evaluation_table(val_results)
    print("\n  ── TEST SET ──")
    _print_evaluation_table(test_results)

    # Performance requirements check
    print("\n  ── PERFORMANCE REQUIREMENTS CHECK ──")
    for m in test_results:
        ok_precision = "✓" if m["precision"] >= 0.85 else "✗ BELOW THRESHOLD"
        ok_recall    = "✓" if m["recall"]    >= 0.80 else "✗ BELOW THRESHOLD"
        ok_auc       = "✓" if m["auc_roc"]   >= 0.90 else "✗ BELOW THRESHOLD"
        print(f"  {m['model']:<22} Precision≥85%:{ok_precision}  Recall≥80%:{ok_recall}  AUC>0.90:{ok_auc}")

    return test_results


def select_best_model(trained_models, test_results) -> tuple[object, object]:
    """Select top model by AUC-ROC × F1 composite. Also returns second-best for secondary slot."""
    _banner("Step 5 · Model Selection")
    scores = [
        test_results[i]["auc_roc"] * 0.6 + test_results[i]["f1"] * 0.4
        for i in range(len(test_results))
    ]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best_idx   = ranked[0]
    second_idx = ranked[1] if len(ranked) > 1 else ranked[0]

    best    = trained_models[best_idx]
    second  = trained_models[second_idx]
    metrics = test_results[best_idx]
    print(f"  ► Best model:    {best.name}")
    print(f"  ► AUC-ROC:       {metrics['auc_roc']}")
    print(f"  ► F1 Score:      {metrics['f1']}")
    print(f"  ► Precision:     {metrics['precision']}")
    print(f"  ► Recall:        {metrics['recall']}")
    print(f"  ► MCC:           {metrics['mcc']}")
    print(f"  ► Secondary (failover): {second.name}")
    return best, second


# ─────────────────────────────────────────────
# STEP 5: SMISHING NLP EVALUATION
# ─────────────────────────────────────────────

def evaluate_smishing(smishing_clf):
    _banner("Step 5b · NLP Smishing Classifier Evaluation")
    from models import SMISHING_CORPUS
    texts, labels = zip(*SMISHING_CORPUS)
    result = smishing_clf.evaluate_text(list(texts), list(labels))
    print(f"  Model       : {result['model']}")
    print(f"  Accuracy    : {result['accuracy']}")
    print(f"  Precision   : {result['precision']}")
    print(f"  Recall      : {result['recall']}")
    print(f"  F1          : {result['f1']}")
    print(f"  AUC-ROC     : {result['auc_roc']}")
    print(f"  MCC         : {result['mcc']}")

    # Demo predictions
    samples = [
        "URGENT: Your MTN account will be suspended. Verify now: mtn-verify.tk",
        "Your Airtel Money balance is K320. Dial *115# to check.",
        "You have won K10,000! Claim your prize at zamtel-winner.cf now!",
    ]
    print("\n  ── SMISHING DETECTION DEMOS ──")
    for sms in samples:
        prob = smishing_clf.predict_proba([sms])[0, 1]
        verdict = "SMISHING ⚠" if prob >= 0.5 else "Legitimate ✓"
        print(f"  [{verdict}] ({prob:.2f}) {sms[:60]}...")


# ─────────────────────────────────────────────
# STEP 6: LIVE API TEST
# ─────────────────────────────────────────────

def run_api_test(best_model, secondary_model=None):
    _banner("Step 6 · Live API Pipeline Test (FastAPI Test Client)")
    from fastapi.testclient import TestClient
    import app as api_app

    api_app.set_primary_model(best_model)
    if secondary_model is not None:
        api_app.set_secondary_model(secondary_model)
        print(f"  ► Secondary (failover) model set: {secondary_model.name}")

    client = TestClient(api_app.app)

    # Test API key comes from FRAUD_API_KEYS — no more hardcoded dev key.
    test_key = next(iter(api_app.VALID_API_KEYS), None)
    if not test_key:
        print("\n  [SKIPPED] FRAUD_API_KEYS is not set — cannot run authenticated API test.")
        print("  Set FRAUD_API_KEYS in your .env and re-run.")
        return {}

    # Health check
    resp = client.get("/health")
    health = resp.json()
    print(f"\n  GET /health → {health['status'].upper()}")
    print(f"    DB: {health['database']}  |  Model: {health['ml_model']}")

    # Construct a suspicious transaction payload
    payload = {
        "txn_id": "TEST-TXN-20240601-0001",
        "timestamp": "2024-06-01T02:15:00",   # Off-hours (2 AM)
        "sender_msisdn": "+260961234567",
        "receiver_msisdn": "+260977999001",    # Unknown beneficiary
        "amount": 9500.00,                     # High amount
        "txn_type": "P2P",
        "operator": "MTN",
        "channel": "APP",
        "province": "Copperbelt",              # Different from Lusaka home
        "device_fingerprint": "UNKNOWN_DEVICE_XYZ_999",
    }

    resp = client.post(
        "/score",
        json=payload,
        headers={"X-API-Key": test_key},
    )
    result = resp.json()

    print(f"\n  POST /score → HTTP {resp.status_code}")
    print(f"  ┌─ Transaction ID : {result.get('txn_id')}")
    print(f"  ├─ Risk Score     : {result.get('risk_score')}/100")
    print(f"  ├─ Risk Level     : {result.get('risk_level')}")
    print(f"  ├─ Fraud Prob     : {result.get('fraud_probability'):.4f}")
    print(f"  ├─ ML Model Used  : {result.get('ml_model_used')}")
    print(f"  ├─ DB Logged      : {result.get('db_logged')}")
    print("  └─ Reason Codes   :")
    for rc in result.get("reason_codes", []):
        print(f"       • {rc}")

    # Audit retrieval
    txn_id = result.get("txn_id")
    resp2 = client.get(
        f"/audit/{txn_id}",
        headers={"X-API-Key": test_key},
    )
    audit = resp2.json()
    print(f"\n  GET /audit/{txn_id[:25]}... → HTTP {resp2.status_code}")
    if audit.get("audit_records"):
        ar = audit["audit_records"][0]
        print(f"  ┌─ Score ID  : {ar['score_id']}")
        print(f"  ├─ Score     : {ar['numeric_score']}")
        print(f"  ├─ Level     : {ar['risk_level']}")
        print(f"  └─ Logged at : {ar['calculated_at']}")

    # Auth test
    resp3 = client.post("/score", json=payload)
    print(f"\n  POST /score (no key) → HTTP {resp3.status_code} (Unauthorised expected)")

    return result


# ─────────────────────────────────────────────
# STEP 7: DATABASE INSPECTION
# ─────────────────────────────────────────────

def inspect_database():
    _banner("Step 7 · Database Records Inspection")
    from db_config import (
        get_engine, Transaction, RiskScoreAudit,
        UserProfile, BehavioralProfile, SmishingSignal, AgentMetadata,
    )

    db = get_engine()
    with Session(db) as session:
        n_users     = session.query(UserProfile).count()
        n_profiles  = session.query(BehavioralProfile).count()
        n_txns      = session.query(Transaction).count()
        n_fraud     = session.query(Transaction).filter_by(is_fraud=1).count()
        n_audits    = session.query(RiskScoreAudit).count()
        n_smishing  = session.query(SmishingSignal).count()
        n_agents    = session.query(AgentMetadata).count()

        print(f"  ┌─ users_profiles      : {n_users:,}")
        print(f"  ├─ behavioral_profiles : {n_profiles:,}")
        print(f"  ├─ transactions        : {n_txns:,}  (fraud: {n_fraud:,} = {n_fraud/max(n_txns,1):.2%})")
        print(f"  ├─ risk_scores_audit   : {n_audits:,}")
        print(f"  ├─ smishing_signals    : {n_smishing:,}")
        print(f"  └─ agent_metadata      : {n_agents:,}")

        # Sample recent audit records
        recent = (
            session.query(RiskScoreAudit)
            .order_by(RiskScoreAudit.calculated_at.desc())
            .limit(5)
            .all()
        )
        if recent:
            print("\n  ── RECENT AUDIT RECORDS ──")
            print(f"  {'score_id':<36} {'txn_id':<36} {'score':>6}  {'level':<10}  {'prob':>6}")
            print(f"  {'─'*36} {'─'*36} {'─'*6}  {'─'*10}  {'─'*6}")
            for r in recent:
                print(
                    f"  {r.score_id[:36]}  {r.txn_id[:36]}  "
                    f"{r.numeric_score:>6.1f}  {r.risk_level:<10}  {r.fraud_probability:>6.4f}"
                )

        # Risk level distribution
        from sqlalchemy import func
        dist = (
            session.query(RiskScoreAudit.risk_level, func.count(RiskScoreAudit.score_id))
            .group_by(RiskScoreAudit.risk_level)
            .all()
        )
        if dist:
            print("\n  ── AUDIT RISK LEVEL DISTRIBUTION ──")
            for level, cnt in dist:
                bar = "█" * min(cnt, 40)
                print(f"  {level:<10} {cnt:>4}  {bar}")


# ─────────────────────────────────────────────
# MAIN ORCHESTRATION
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 64)
    print("  ZAMBIA MOBILE MONEY FRAUD DETECTION SYSTEM")
    print("  Sprint 1 & 2 — End-to-End Execution")
    print(f"  Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("═" * 64)

    try:
        # 1. Database setup
        users, transactions = setup_database()

        # 2. Feature engineering & splits
        X_train, X_val, X_test, y_train, y_val, y_test = build_dataset()

        # 3. Train models
        trained_models, smishing_clf = train_models(X_train, y_train)

        # 4. Evaluate
        test_results = evaluate_models(trained_models, X_val, y_val, X_test, y_test)

        # 5. Select best
        best_model, second_model = select_best_model(trained_models, test_results)

        # 5b. NLP evaluation
        evaluate_smishing(smishing_clf)

        # 5c. Seed reference data
        _banner("Step 5c · Seed Reference Data")
        from db_config import seed_reference_data
        seed_reference_data()
        print("  ► Reference data (KYC limits, operator prefixes, provinces) seeded.")

        # 5d. Write initial ModelVersion row for the production model
        _banner("Step 5d · Record Initial Model Version")
        try:
            from retraining import RetrainingPipeline
            pipeline_meta = RetrainingPipeline(model_class_name=best_model.name)
            pipeline_meta._write_model_version(
                version_id=__import__('uuid').uuid4().__str__(),
                model_name=best_model.name,
                version_tag=datetime.utcnow().strftime("%Y-%m-%d-v0"),
                metrics=test_results[[r["model"] for r in test_results].index(best_model.name)],
                promoted=True,
                promoted_by="main.py",
                promotion_reason="Initial training run",
            )
            print(f"  ► ModelVersion row written for {best_model.name}.")
        except Exception as mv_err:
            print(f"  ► ModelVersion write skipped: {mv_err}")

        # 6. API pipeline test
        api_result = run_api_test(best_model, secondary_model=second_model)

        # 7. DB inspection
        inspect_database()

        # 8. Start monthly retraining scheduler
        _banner("Step 8 · Starting Monthly Retraining Scheduler")
        try:
            from retraining import start_scheduler
            start_scheduler()
            print("  ► APScheduler started — monthly retraining cron active (day=1, hour=2).")
        except Exception as sched_err:
            print(f"  ► Scheduler start skipped: {sched_err}")

        # Final summary
        _banner("✓ Execution Complete")
        print(f"  Best model     : {best_model.name}")
        print(f"  API risk score : {api_result.get('risk_score')}/100 ({api_result.get('risk_level')})")
        print(f"  Fraud prob     : {api_result.get('fraud_probability'):.4f}")
        print(f"  Completed at   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

    except Exception:
        print("\n[FATAL ERROR]", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
