"""
MODULE 3: Main Dashboard Application Execution
Mobile Money Fraud Detection System - Zambia
app_dashboard.py

Single-entry-point Streamlit application. Assembles the sidebar filters,
builds the 3 operational tabs, and wires dashboard_db (data) to
dashboard_ui (presentation).

Run with:
    streamlit run app_dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

import dashboard_ui as ui
from dashboard_db import DashboardDBConnector

import auth
import alerts
import overrides
import compliance

# ─────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Zambia Mobile Money Fraud Detection — Analyst Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

OPERATORS = ["MTN", "Airtel", "Zamtel"]
PROVINCES = [
    "Lusaka", "Copperbelt", "Central", "Southern", "Eastern",
    "Western", "Northern", "Muchinga", "North-Western", "Luapula",
]
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


# ─────────────────────────────────────────────
# DB CONNECTION (cached for the life of the session)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_connector() -> DashboardDBConnector:
    return DashboardDBConnector()


# ─────────────────────────────────────────────
# CACHED DATA FETCHERS
# Leading underscore on `_db` tells Streamlit's hasher to skip that arg.
# ─────────────────────────────────────────────

@st.cache_data(ttl=20, show_spinner=False)
def fetch_summary(_db, operators, provinces, risk_levels, date_from, date_to):
    return _db.get_summary_metrics(operators, provinces, risk_levels, date_from, date_to)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_alerts(_db, operators, provinces, date_from, date_to, threshold=60.0):
    return _db.get_high_risk_alerts(
        threshold=threshold, limit=50,
        operators=operators, provinces=provinces,
        date_from=date_from, date_to=date_to,
    )


@st.cache_data(ttl=20, show_spinner=False)
def fetch_province_counts(_db, operators, date_from, date_to):
    return _db.get_province_fraud_counts(operators, date_from, date_to)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_operator_counts(_db, provinces, date_from, date_to):
    return _db.get_operator_fraud_counts(provinces, date_from, date_to)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_daily_trend(_db, days, operators, provinces):
    return _db.get_daily_trend(days, operators, provinces)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_audit_queue(_db, operators, provinces, risk_levels, date_from, date_to, limit=300):
    return _db.get_audit_queue(operators, provinces, risk_levels, date_from, date_to, limit)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_search(_db, fragment, limit=100):
    return _db.search_by_msisdn(fragment, limit)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_metrics(_db):
    return _db.get_model_performance_metrics()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_score_distribution(_db):
    return _db.get_risk_score_distribution()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_state():
    st.session_state.setdefault("selected_txn_id", None)
    st.session_state.setdefault("user", None)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(db: DashboardDBConnector) -> dict:
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Ops Console")
        st.caption("Zambia Mobile Money Fraud Detection System")

        if st.session_state.user:
            st.markdown(f"**👤 {st.session_state.user.full_name}**")
            st.caption(f"Role: {st.session_state.user.role.title()}")
            if st.button("🚪 Logout", width='stretch'):
                st.session_state.user = None
                st.rerun()
            st.divider()

        connected = db.ping()
        ui.render_connection_status(connected)
        if not connected:
            st.error(
                "Cannot reach MySQL. Check your `.env` credentials and confirm "
                "the database server is running and reachable."
            )

        ui.section_header("Global Filters")

        sel_operators = st.multiselect("Mobile Operator", OPERATORS, default=OPERATORS)
        sel_provinces = st.multiselect("Province", PROVINCES, default=PROVINCES)
        sel_risk_levels = st.multiselect("Fraud Risk Level", RISK_LEVELS, default=RISK_LEVELS)

        st.markdown("**Date / Time Range**")
        days_back = st.slider("Lookback window (days)", min_value=1, max_value=90, value=30)
        date_from = datetime.now(timezone.utc) - timedelta(days=days_back)
        date_to = datetime.now(timezone.utc)
        st.caption(f"From **{date_from:%Y-%m-%d}** to **{date_to:%Y-%m-%d}**")

        st.divider()
        if st.button("🔄 Refresh data", width='stretch'):
            st.cache_data.clear()
            st.rerun()

        st.caption("Data auto-refreshes every ~20s while this tab is open.")

    return {
        # Empty multiselect = "show everything" (no filter applied), not "show nothing"
        "operators": sel_operators or None,
        "provinces": sel_provinces or None,
        "risk_levels": sel_risk_levels or None,
        "date_from": date_from,
        "date_to": date_to,
    }


# ─────────────────────────────────────────────
# TAB 1 — EXECUTIVE OPERATIONS OVERVIEW
# ─────────────────────────────────────────────

def render_tab_overview(db: DashboardDBConnector, f: dict):
    # Alert Banner
    unread_alerts = alerts.get_unread_alerts(limit=5)
    if unread_alerts:
        st.error(f"🚨 **{len(unread_alerts)} Unread High-Priority Alert(s)**")
        for alt in unread_alerts:
            st.markdown(f"**{alt['alert_type']} ({alt['severity']} Risk)**: {alt['message']}")
            if st.button("Mark Read", key=f"read_{alt['alert_id']}"):
                alerts.mark_alert_read(alt['alert_id'], resolved_by=st.session_state.user.user_id)
                st.rerun()
        st.divider()

    metrics = fetch_summary(db, f["operators"], f["provinces"], f["risk_levels"], f["date_from"], f["date_to"])
    ui.render_kpi_row(metrics)

    st.write("")
    ui.section_header("Real-Time High-Risk Alerts (Score ≥ 60)")
    alerts_df = fetch_alerts(db, f["operators"], f["provinces"], f["date_from"], f["date_to"])
    ui.render_alerts_table(alerts_df)

    st.write("")
    col1, col2 = st.columns([3, 2])
    with col1:
        province_df = fetch_province_counts(db, f["operators"], f["date_from"], f["date_to"])
        st.plotly_chart(ui.province_bar_chart(province_df), width='stretch')
    with col2:
        operator_df = fetch_operator_counts(db, f["provinces"], f["date_from"], f["date_to"])
        st.plotly_chart(ui.operator_donut_chart(operator_df), width='stretch')

    ui.section_header("Daily Transaction & Fraud Trend")
    span_days = max((f["date_to"] - f["date_from"]).days, 1)
    trend_df = fetch_daily_trend(db, span_days, f["operators"], f["provinces"])
    st.plotly_chart(ui.daily_trend_chart(trend_df), width='stretch')


# ─────────────────────────────────────────────
# TAB 2 — FRAUD ANALYST INVESTIGATION TERMINAL
# ─────────────────────────────────────────────

def render_tab_investigation(db: DashboardDBConnector, f: dict):
    ui.section_header("Audit Queue")

    search_col, count_col = st.columns([3, 1])
    with search_col:
        query = st.text_input(
            "🔍 Search by Transaction ID or hashed Sender MSISDN fragment",
            placeholder="e.g. TEST-TXN-2024-0001 or a1b2c3d4...",
        )

    if query.strip():
        queue_df = fetch_search(db, query.strip(), 100)
    else:
        queue_df = fetch_audit_queue(
            db, f["operators"], f["provinces"], f["risk_levels"],
            f["date_from"], f["date_to"], 300,
        )

    with count_col:
        st.metric("Records shown", len(queue_df))

    if queue_df.empty:
        st.info(
            "No audit records match the current filters/search. If the database "
            "was just seeded, run the backfill script so transactions have "
            "risk scores attached (see project README)."
        )
        return

    display_cols = [
        c for c in [
            "txn_id", "timestamp", "masked_sender", "amount",
            "operator", "province", "txn_type", "numeric_score", "risk_level",
        ] if c in queue_df.columns
    ]
    table_df = queue_df[display_cols].copy()
    table_df.insert(0, "Investigate", False)

    st.caption("Tick **Investigate** on a row to open its full drill-down panel below.")
    edited = st.data_editor(
        table_df,
        hide_index=True,
        width='stretch',
        disabled=display_cols,
        column_config={
            "Investigate": st.column_config.CheckboxColumn("🔎", width="small"),
            "txn_id": st.column_config.TextColumn("Transaction ID"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp"),
            "masked_sender": st.column_config.TextColumn("Sender (masked)"),
            "amount": st.column_config.NumberColumn("Amount (ZMW)", format="K %.2f"),
            "operator": st.column_config.TextColumn("Operator"),
            "province": st.column_config.TextColumn("Province"),
            "txn_type": st.column_config.TextColumn("Type"),
            "numeric_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%.1f"
            ),
            "risk_level": st.column_config.TextColumn("Risk Level"),
        },
        key="audit_queue_editor",
    )

    checked = edited[edited["Investigate"]]
    if not checked.empty:
        st.session_state.selected_txn_id = checked.iloc[0]["txn_id"]

    if st.session_state.selected_txn_id:
        st.write("")
        detail_df = db.get_transaction_detail(st.session_state.selected_txn_id)
        if detail_df is not None and not detail_df.empty:
            row = detail_df.iloc[0]
            features = db.get_feature_vector(st.session_state.selected_txn_id)
            ui.render_transaction_detail(row, features)
            
            # OVERRIDES & FLAGS
            st.write("---")
            col_ov, col_fl = st.columns(2)
            
            with col_ov:
                st.subheader("Override Decision")
                with st.form("override_form"):
                    decision = st.selectbox("Decision", ["CONFIRM_FRAUD", "MARK_LEGITIMATE"])
                    justification = st.text_area("Justification (Required)")
                    if st.form_submit_button("Submit Override"):
                        if not justification:
                            st.error("Justification is required.")
                        else:
                            try:
                                overrides.submit_override(
                                    txn_id=row["txn_id"],
                                    original_risk_level=row["risk_level"],
                                    original_risk_score=row["numeric_score"],
                                    decision=decision,
                                    justification=justification,
                                    overridden_by=st.session_state.user.username
                                )
                                st.success("Override submitted successfully.")
                            except Exception as e:
                                st.error(f"Error: {e}")

            with col_fl:
                st.subheader("Manual Flag")
                with st.form("flag_form"):
                    reason = st.text_area("Reason for Flagging (Required)")
                    if st.form_submit_button("Flag Transaction"):
                        if not reason:
                            st.error("Reason is required.")
                        else:
                            try:
                                overrides.flag_transaction(
                                    txn_id=row["txn_id"],
                                    flagged_by=st.session_state.user.username,
                                    reason=reason
                                )
                                st.success("Transaction flagged successfully.")
                            except Exception as e:
                                st.error(f"Error: {e}")
                                
            if st.button("✕ Close detail panel"):
                st.session_state.selected_txn_id = None
                st.rerun()
        else:
            st.warning("No audit detail found for the selected transaction.")


# ─────────────────────────────────────────────
# TAB 3 — SYSTEM PERFORMANCE & EXPLAINABILITY
# ─────────────────────────────────────────────

def render_tab_performance(db: DashboardDBConnector):
    ui.section_header("Model Health Tracking")
    metrics = fetch_model_metrics(db)

    if metrics.get("sample_size", 0) == 0:
        st.info(
            "No audited transactions yet — model health metrics are derived by "
            "comparing `risk_scores_audit` against ground-truth fraud labels in "
            "`transactions`. Score some transactions first."
        )
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            ui.render_model_metrics(metrics)
            st.caption(
                f"Derived from {metrics['sample_size']:,} audited transactions vs. "
                f"ground-truth fraud labels (rule: risk score ≥ 60 ⇒ predicted fraud)."
            )
        with col2:
            ui.render_confusion_matrix(metrics)

    st.write("")
    col3, col4 = st.columns([3, 2])
    with col3:
        st.plotly_chart(ui.feature_importance_chart(), width='stretch')
    with col4:
        dist_df = fetch_score_distribution(db)
        st.plotly_chart(ui.score_distribution_chart(dist_df), width='stretch')

    # ── MODEL VERSION HISTORY (report's "model monitor page") ──
    st.write("")
    ui.section_header("📆 Model Version History")
    try:
        from db_config import ModelVersion, get_engine
        from sqlalchemy.orm import Session
        import json
        with Session(get_engine()) as s:
            versions = (
                s.query(ModelVersion)
                .order_by(ModelVersion.trained_at.desc())
                .limit(20)
                .all()
            )
        if not versions:
            st.info("No model versions recorded yet. Run main.py to register the initial model.")
        else:
            rows = []
            for v in versions:
                m = json.loads(v.metrics_json or "{}")
                rows.append({
                    "Version Tag": v.version_tag,
                    "Model": v.model_name,
                    "Trained At": v.trained_at.strftime("%Y-%m-%d %H:%M") if v.trained_at else "—",
                    "Deployed At": v.deployed_at.strftime("%Y-%m-%d %H:%M") if v.deployed_at else "—",
                    "Active": "✅ Production" if v.is_active else ("🔍 Candidate" if v.is_candidate else "❌ Rejected"),
                    "Precision": f"{m.get('precision', 0):.3f}",
                    "Recall":    f"{m.get('recall', 0):.3f}",
                    "AUC-ROC":   f"{m.get('auc_roc', 0):.3f}",
                    "F1":        f"{m.get('f1', 0):.3f}",
                    "Promoted By": v.promoted_by or "—",
                    "Outcome": v.promotion_reason or v.rejection_reason or "—",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    except Exception as e:
        st.warning(f"Model version history unavailable: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def render_tab_user_management(db: DashboardDBConnector):
    ui.section_header("User Profile Management (Admin)")
    search = st.text_input("Search user profiles by hashed MSISDN or UID")
    if search:
        profiles = db.search_user_profiles(search)
        if not profiles.empty:
            st.dataframe(profiles, width='stretch')
            
            st.subheader("Update User Profile")
            sel_msisdn = st.selectbox("Select MSISDN to update", profiles["hashed_msisdn"].unique())
            sel_prof = profiles[profiles["hashed_msisdn"] == sel_msisdn].iloc[0]
            
            with st.form("update_profile_form"):
                new_kyc = st.number_input("KYC Tier", min_value=1, max_value=3, value=int(sel_prof["kyc_tier"]))
                new_amt = st.number_input("Avg 30-Day Amount", value=float(sel_prof["avg_30day_amount"]))
                new_hrs = st.text_input("Typical Active Hours", value=sel_prof["typical_active_hours"])
                new_prov = st.selectbox("Usual Province", PROVINCES, index=PROVINCES.index(sel_prof["usual_province_centroid"]) if sel_prof["usual_province_centroid"] in PROVINCES else 0)
                
                if st.form_submit_button("Update Profile"):
                    success = db.update_user_profile(sel_msisdn, new_kyc, new_amt, new_hrs, new_prov)
                    if success:
                        st.success("Profile updated.")
                    else:
                        st.error("Failed to update profile.")
        else:
            st.info("No profiles found.")

    st.divider()
    st.subheader("📧 Dashboard User Management")
    existing_users = auth.get_all_users()
    if existing_users:
        import pandas as pd
        st.dataframe(pd.DataFrame(existing_users), width='stretch', hide_index=True)

    # Self-registration form (FR gap #9)
    st.subheader("➕ Register New Dashboard User")
    with st.form("register_user_form"):
        r_username  = st.text_input("Username", placeholder="e.g. jmulenga")
        r_fullname  = st.text_input("Full Name", placeholder="e.g. John Mulenga")
        r_email     = st.text_input("Email (optional)")
        r_role      = st.selectbox("Role", ["operator", "analyst", "admin"])
        r_password  = st.text_input("Temporary Password", type="password")
        r_password2 = st.text_input("Confirm Password", type="password")
        reg_btn = st.form_submit_button("📨 Create Account")

    if reg_btn:
        if not r_username or not r_password:
            st.error("Username and password are required.")
        elif r_password != r_password2:
            st.error("Passwords do not match.")
        elif len(r_password) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            try:
                auth.register_user(
                    username=r_username.strip(),
                    password=r_password,
                    role=r_role,
                    full_name=r_fullname.strip(),
                    email=r_email.strip() or None,
                )
                st.success(f"✅ User '{r_username}' created with role '{r_role}'.")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Registration failed: {e}")

def render_tab_compliance():
    ui.section_header("BoZ Compliance Reports (Admin)")
    with st.form("report_form"):
        st.write("Generate a regulatory compliance report for the Bank of Zambia.")
        d1 = st.date_input("Start Date", value=datetime.now(timezone.utc).date() - timedelta(days=30))
        d2 = st.date_input("End Date", value=datetime.now(timezone.utc).date())
        col_html, col_pdf = st.columns(2)
        with col_html:
            generate_html = st.form_submit_button("📊 Generate HTML Preview")
        with col_pdf:
            generate_pdf = st.form_submit_button("⬇️ Download PDF (FR-15)")

    start_dt = datetime.combine(d1, datetime.min.time())
    end_dt   = datetime.combine(d2, datetime.max.time())
    generated_by = st.session_state.user.full_name

    if generate_html:
        html = compliance.generate_compliance_report(
            date_from=start_dt, date_to=end_dt, generated_by=generated_by
        )
        st.html(html)

    if generate_pdf:
        try:
            with st.spinner("Generating PDF..."):
                pdf_bytes = compliance.generate_compliance_report_pdf(
                    date_from=start_dt, date_to=end_dt, generated_by=generated_by
                )
            fname = f"BOZ_Compliance_Report_{d1}_{d2}.pdf"
            st.download_button(
                label="📅 Click to download PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
            )
            st.success(f"PDF ready: {fname}")
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


def render_tab_reference_data():
    """Admin-only: CRUD UI for the reference_data table (blocklist, KYC limits)."""
    ui.section_header("📚 Reference Data Management (Admin)")
    try:
        from db_config import ReferenceData, get_engine
        from sqlalchemy.orm import Session
        from scoring_engine import add_to_blocklist_db, remove_from_blocklist_db
        import pandas as pd

        with Session(get_engine()) as s:
            all_rows = s.query(ReferenceData).filter_by(is_active=True).all()

        st.subheader("🚫 Fraud Blocklist")
        blocklist_rows = [r for r in all_rows if r.ref_type == "blocklist"]
        if blocklist_rows:
            bl_df = pd.DataFrame([
                {"Hashed MSISDN": r.ref_key, "Added By": r.updated_by,
                 "Added At": r.updated_at.strftime("%Y-%m-%d %H:%M") if r.updated_at else "—"}
                for r in blocklist_rows
            ])
            st.dataframe(bl_df, width='stretch', hide_index=True)
        else:
            st.info("Blocklist is empty.")

        with st.expander("➕ Add MSISDN to Blocklist"):
            with st.form("blocklist_add_form"):
                raw_msisdn = st.text_input("MSISDN (e.g. +260961234567)")
                add_btn = st.form_submit_button("Add to Blocklist")
            if add_btn and raw_msisdn.strip():
                from db_config import sha256_hash
                hashed = sha256_hash(raw_msisdn.strip())
                add_to_blocklist_db(hashed, added_by=st.session_state.user.username)
                st.success(f"Added to blocklist (hashed). Refresh to confirm.")
                st.cache_data.clear()

        with st.expander("❌ Remove MSISDN from Blocklist"):
            with st.form("blocklist_remove_form"):
                rem_msisdn = st.text_input("Raw MSISDN to remove")
                rem_btn = st.form_submit_button("Remove from Blocklist")
            if rem_btn and rem_msisdn.strip():
                from db_config import sha256_hash
                hashed = sha256_hash(rem_msisdn.strip())
                ok = remove_from_blocklist_db(hashed, removed_by=st.session_state.user.username)
                if ok:
                    st.success("Entry deactivated.")
                else:
                    st.warning("MSISDN not found on active blocklist.")
                st.cache_data.clear()

        st.divider()
        st.subheader("💰 KYC Tier Limits")
        kyc_rows = [r for r in all_rows if r.ref_type == "kyc_limit"]
        if kyc_rows:
            kyc_df = pd.DataFrame([
                {"Tier": r.ref_key, "Limit (ZMW)": r.ref_value, "Description": r.description or ""}
                for r in sorted(kyc_rows, key=lambda x: x.ref_key)
            ])
            st.dataframe(kyc_df, width='stretch', hide_index=True)

        with st.expander("✏️ Update KYC Limit"):
            with st.form("kyc_update_form"):
                tier = st.selectbox("Tier", ["1", "2", "3"])
                new_limit = st.number_input("New limit (ZMW)", min_value=1.0, value=1000.0)
                kyc_btn = st.form_submit_button("Update KYC Limit")
            if kyc_btn:
                with Session(get_engine()) as s:
                    # Soft-deactivate old row
                    old = s.query(ReferenceData).filter_by(ref_type="kyc_limit", ref_key=str(tier), is_active=True).first()
                    if old:
                        old.is_active = False
                    import uuid, datetime as _dt
                    s.add(ReferenceData(
                        ref_type="kyc_limit",
                        ref_key=str(tier),
                        ref_value=str(new_limit),
                        description=f"KYC Tier {tier} max transaction (ZMW) — updated by {st.session_state.user.username}",
                        is_active=True,
                        updated_by=st.session_state.user.username,
                        updated_at=_dt.datetime.utcnow(),
                    ))
                    s.commit()
                st.success(f"KYC Tier {tier} limit updated to ZMW {new_limit:,.2f}.")
                st.cache_data.clear()

    except Exception as e:
        st.error(f"Reference data management unavailable: {e}")


def render_tab_retrain():
    """Admin-only: trigger retraining cycle and show last ModelVersion outcome."""
    ui.section_header("🔄 Model Retraining (FR-13, Admin)")
    st.markdown(
        "Trigger a full retraining cycle. The candidate model will be evaluated against "
        "NFR-03 thresholds (Precision ≥ 0.85, Recall ≥ 0.80, AUC-ROC ≥ 0.90). "
        "If all thresholds are met it is promoted to production; otherwise the current model is retained."
    )

    # Show retraining corpus stats
    try:
        from db_config import RetrainingCorpus, get_engine
        from sqlalchemy.orm import Session
        with Session(get_engine()) as s:
            total_corpus = s.query(RetrainingCorpus).filter(
                RetrainingCorpus.confirmed_label.isnot(None)
            ).count()
            unused_corpus = s.query(RetrainingCorpus).filter(
                RetrainingCorpus.confirmed_label.isnot(None),
                RetrainingCorpus.used_in_retrain == False  # noqa: E712
            ).count()
        col1, col2 = st.columns(2)
        col1.metric("Total Confirmed Labels in Corpus", total_corpus)
        col2.metric("Unused Labels (next cycle)", unused_corpus)
    except Exception:
        pass

    st.divider()
    if st.button("▶️ Trigger Retraining Now", type="primary", key="retrain_trigger_btn"):
        try:
            from retraining import RetrainingPipeline
            import threading
            result_store = {}
            current_user = st.session_state.user.username

            def _run():
                p = RetrainingPipeline()
                result_store["result"] = p.run(triggered_by=f"dashboard:{current_user}")

            with st.spinner("Retraining in progress (this may take several minutes)..."):
                t = threading.Thread(target=_run, daemon=True)
                t.start()
                t.join(timeout=300)  # Wait up to 5 min

            if "result" in result_store:
                r = result_store["result"]
                if r["promoted"]:
                    st.success(f"✅ Model promoted! {r['reason']}")
                else:
                    st.warning(f"⚠️ Candidate rejected: {r['reason']}")
                st.json(r["metrics"])
            else:
                st.warning("Retraining is still running. Check model_versions table for the outcome.")
        except ImportError:
            st.error("Retraining module not available. Run main.py first.")
        except Exception as e:
            st.error(f"Retraining error: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def render_login():
    st.title("🛡️ Fraud Ops Console - Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = auth.authenticate(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

def main():
    init_state()
    db = get_connector()
    
    if not st.session_state.user:
        render_login()
        return

    st.title("🛡️ Zambia Mobile Money Fraud Detection")
    st.caption("Real-Time Fraud Analyst Dashboard & Auditing Interface · Sprint 3")

    filters = render_sidebar(db)

    tabs = ["📊 Executive Overview"]
    if auth.has_role(st.session_state.user, "analyst"):
        tabs.append("🕵️ Investigation Terminal")
    tabs.append("🧠 System Performance & XAI")
    
    if auth.has_role(st.session_state.user, "admin"):
        tabs.append("👤 User Management")
        tabs.append("📚 Reference Data")
        tabs.append("🔄 Retrain Model")
        tabs.append("📋 Compliance Reports")

    st_tabs = st.tabs(tabs)

    with st_tabs[0]:
        render_tab_overview(db, filters)
        
    idx = 1
    if "🕵️ Investigation Terminal" in tabs:
        with st_tabs[idx]:
            render_tab_investigation(db, filters)
        idx += 1
        
    with st_tabs[idx]:
        render_tab_performance(db)
    idx += 1
    
    if "👤 User Management" in tabs:
        with st_tabs[idx]:
            render_tab_user_management(db)
        idx += 1

    if "📚 Reference Data" in tabs:
        with st_tabs[idx]:
            render_tab_reference_data()
        idx += 1

    if "🔄 Retrain Model" in tabs:
        with st_tabs[idx]:
            render_tab_retrain()
        idx += 1
        
    if "📋 Compliance Reports" in tabs:
        with st_tabs[idx]:
            render_tab_compliance()


if __name__ == "__main__":
    main()
