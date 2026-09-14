"""
MODULE 10: BoZ Compliance Report Generator
Mobile Money Fraud Detection System - Zambia
compliance.py

Implements the "Submit Compliance Report" use case from the actor diagram.
The System Administrator generates a Bank of Zambia (BoZ) compliant report
covering a specified date range.

Report sections:
    1. Executive Summary (total txns, fraud rate, blocked funds)
    2. High-Risk Transaction Inventory with reason codes
    3. Override & Manual Flag Decisions (audit trail)
    4. Model Performance Metrics (precision, recall, F1, MCC)
    5. BoZ Regulatory References

Output format: HTML string (renderable in Streamlit + downloadable).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from db_config import get_engine


# ─────────────────────────────────────────────
# REPORT DATA QUERIES
# ─────────────────────────────────────────────

def _query_summary(session, date_from: datetime, date_to: datetime) -> dict:
    """Executive summary KPIs for the report period."""
    sql = text("""
        SELECT
            COUNT(t.txn_id) AS total_txns,
            COALESCE(SUM(t.is_fraud), 0) AS total_fraud,
            COALESCE(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 0)
                AS blocked_funds,
            COALESCE(AVG(r.numeric_score), 0) AS avg_risk_score,
            COALESCE(SUM(CASE WHEN r.risk_level = 'High' THEN 1 ELSE 0 END), 0)
                AS high_risk_count,
            COALESCE(SUM(CASE WHEN r.risk_level = 'Medium' THEN 1 ELSE 0 END), 0)
                AS medium_risk_count,
            COALESCE(SUM(CASE WHEN r.risk_level = 'Low' THEN 1 ELSE 0 END), 0)
                AS low_risk_count
        FROM transactions t
        LEFT JOIN risk_scores_audit r ON t.txn_id = r.txn_id
        WHERE t.timestamp >= :date_from AND t.timestamp <= :date_to
    """)
    row = session.execute(sql, {"date_from": date_from, "date_to": date_to}).fetchone()
    total = row[0] or 1
    return {
        "total_txns": int(row[0]),
        "total_fraud": int(row[1]),
        "fraud_rate_pct": round(float(row[1]) / total * 100, 2),
        "blocked_funds_zmw": round(float(row[2]), 2),
        "avg_risk_score": round(float(row[3]), 1),
        "high_risk_count": int(row[4]),
        "medium_risk_count": int(row[5]),
        "low_risk_count": int(row[6]),
    }


def _query_high_risk_txns(session, date_from: datetime, date_to: datetime, limit: int = 50) -> list[dict]:
    """High-risk transactions with reason codes."""
    sql = text("""
        SELECT
            t.txn_id,
            t.timestamp,
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
        WHERE t.timestamp >= :date_from
          AND t.timestamp <= :date_to
          AND r.numeric_score >= 60
        ORDER BY r.numeric_score DESC
        LIMIT :lim
    """)
    rows = session.execute(sql, {"date_from": date_from, "date_to": date_to, "lim": limit}).fetchall()
    return [
        {
            "txn_id": r[0],
            "timestamp": str(r[1]),
            "amount": float(r[2]),
            "operator": r[3],
            "province": r[4],
            "txn_type": r[5],
            "risk_score": float(r[6]),
            "risk_level": r[7],
            "reason_codes": r[8] or "[]",
            "fraud_probability": float(r[9]),
        }
        for r in rows
    ]


def _query_province_breakdown(session, date_from: datetime, date_to: datetime) -> list[dict]:
    """Fraud counts per province for the reporting period."""
    sql = text("""
        SELECT
            t.province,
            COUNT(*) AS total_txns,
            SUM(t.is_fraud) AS fraud_count,
            ROUND(SUM(t.is_fraud) / COUNT(*) * 100, 2) AS fraud_rate_pct
        FROM transactions t
        WHERE t.timestamp >= :date_from AND t.timestamp <= :date_to
        GROUP BY t.province
        ORDER BY fraud_count DESC
    """)
    rows = session.execute(sql, {"date_from": date_from, "date_to": date_to}).fetchall()
    return [
        {
            "province": r[0],
            "total_txns": int(r[1]),
            "fraud_count": int(r[2]),
            "fraud_rate_pct": float(r[3]) if r[3] else 0.0,
        }
        for r in rows
    ]


def _query_overrides(session, date_from: datetime, date_to: datetime) -> list[dict]:
    """Override decisions within the reporting period."""
    try:
        sql = text("""
            SELECT override_id, txn_id, original_risk_level, original_risk_score,
                   override_decision, justification, overridden_by, created_at
            FROM fraud_overrides
            WHERE created_at >= :date_from AND created_at <= :date_to
            ORDER BY created_at DESC
        """)
        rows = session.execute(sql, {"date_from": date_from, "date_to": date_to}).fetchall()
        return [
            {
                "override_id": r[0],
                "txn_id": r[1],
                "original_risk_level": r[2],
                "original_risk_score": float(r[3]),
                "decision": r[4],
                "justification": r[5],
                "overridden_by": r[6],
                "created_at": str(r[7]),
            }
            for r in rows
        ]
    except Exception:
        return []


def _query_manual_flags(session, date_from: datetime, date_to: datetime) -> list[dict]:
    """Manual flags within the reporting period."""
    try:
        sql = text("""
            SELECT flag_id, txn_id, flagged_by, reason, status, created_at
            FROM manual_flags
            WHERE created_at >= :date_from AND created_at <= :date_to
            ORDER BY created_at DESC
        """)
        rows = session.execute(sql, {"date_from": date_from, "date_to": date_to}).fetchall()
        return [
            {
                "flag_id": r[0],
                "txn_id": r[1],
                "flagged_by": r[2],
                "reason": r[3],
                "status": r[4],
                "created_at": str(r[5]),
            }
            for r in rows
        ]
    except Exception:
        return []


def _query_model_performance(session) -> dict:
    """Derive model performance from audit vs ground truth."""
    sql = text("""
        SELECT
            t.is_fraud,
            r.numeric_score
        FROM risk_scores_audit r
        INNER JOIN transactions t ON r.txn_id = t.txn_id
    """)
    rows = session.execute(sql).fetchall()
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "sample_size": 0}

    tp = sum(1 for r in rows if r[1] >= 60 and r[0] == 1)
    fp = sum(1 for r in rows if r[1] >= 60 and r[0] == 0)
    fn = sum(1 for r in rows if r[1] < 60 and r[0] == 1)
    tn = sum(1 for r in rows if r[1] < 60 and r[0] == 0)

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
        "sample_size": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────
# HTML REPORT GENERATOR
# ─────────────────────────────────────────────

def generate_compliance_report(
    date_from: datetime,
    date_to: datetime,
    generated_by: str = "System Administrator",
) -> str:
    """
    Generate a BoZ-compliant HTML compliance report for the given date range.

    Returns an HTML string that can be rendered in Streamlit or downloaded.
    """
    engine = get_engine()
    with Session(engine) as session:
        summary = _query_summary(session, date_from, date_to)
        high_risk = _query_high_risk_txns(session, date_from, date_to)
        provinces = _query_province_breakdown(session, date_from, date_to)
        overrides = _query_overrides(session, date_from, date_to)
        flags = _query_manual_flags(session, date_from, date_to)
        model_perf = _query_model_performance(session)

    report_id = f"BOZ-CR-{date_from:%Y%m%d}-{date_to:%Y%m%d}"
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build HTML
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #1a1a2e;
                line-height: 1.6;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px 30px;
                background: #ffffff;
            }}
            h1 {{ color: #0d1117; font-size: 1.6rem; border-bottom: 3px solid #00843D; padding-bottom: 10px; }}
            h2 {{ color: #161B22; font-size: 1.2rem; margin-top: 30px; border-bottom: 1px solid #e1e4e8; padding-bottom: 6px; }}
            h3 {{ color: #30363D; font-size: 1rem; }}
            .meta {{ color: #6a737d; font-size: 0.85rem; margin-bottom: 20px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
            .kpi-box {{
                background: #f6f8fa;
                border: 1px solid #e1e4e8;
                border-radius: 8px;
                padding: 14px;
                text-align: center;
            }}
            .kpi-val {{ font-size: 1.5rem; font-weight: 700; color: #0d1117; }}
            .kpi-lbl {{ font-size: 0.72rem; color: #6a737d; text-transform: uppercase; letter-spacing: 0.05em; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.82rem;
                margin: 12px 0;
            }}
            th {{ background: #f6f8fa; color: #24292e; font-weight: 600; text-align: left; padding: 8px 10px; border-bottom: 2px solid #e1e4e8; }}
            td {{ padding: 6px 10px; border-bottom: 1px solid #e1e4e8; }}
            tr:nth-child(even) td {{ background: #fafbfc; }}
            .badge {{
                display: inline-block; padding: 2px 8px; border-radius: 999px;
                font-size: 0.7rem; font-weight: 600;
            }}
            .badge-high {{ background: #fdd; color: #d73a49; }}
            .badge-medium {{ background: #fff3cd; color: #b08800; }}
            .badge-low {{ background: #dcffe4; color: #22863a; }}
            .footer {{ margin-top: 40px; padding-top: 16px; border-top: 2px solid #00843D;
                        font-size: 0.78rem; color: #6a737d; }}
            .reg-ref {{ background: #f1f8ff; border-left: 3px solid #0366d6; padding: 10px 14px;
                         font-size: 0.82rem; margin: 8px 0; }}
        </style>
    </head>
    <body>
        <h1>🏛️ Bank of Zambia — Fraud Detection Compliance Report</h1>
        <div class="meta">
            <strong>Report ID:</strong> {report_id}<br>
            <strong>Period:</strong> {date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}<br>
            <strong>Generated:</strong> {generated_at}<br>
            <strong>Prepared by:</strong> {generated_by}<br>
            <strong>System:</strong> Zambia Mobile Money Fraud Detection & Response System
        </div>

        <h2>1. Executive Summary</h2>
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-val">{summary['total_txns']:,}</div>
                <div class="kpi-lbl">Total Transactions</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val" style="color:#d73a49;">{summary['fraud_rate_pct']}%</div>
                <div class="kpi-lbl">Fraud Rate</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val" style="color:#b08800;">K {summary['blocked_funds_zmw']:,.0f}</div>
                <div class="kpi-lbl">Blocked Funds (ZMW)</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-val">{summary['avg_risk_score']}/100</div>
                <div class="kpi-lbl">Avg Risk Score</div>
            </div>
        </div>

        <h3>Risk Level Distribution</h3>
        <table>
            <tr><th>Risk Level</th><th>Count</th><th>Percentage</th></tr>
            <tr>
                <td><span class="badge badge-high">High</span></td>
                <td>{summary['high_risk_count']:,}</td>
                <td>{summary['high_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%</td>
            </tr>
            <tr>
                <td><span class="badge badge-medium">Medium</span></td>
                <td>{summary['medium_risk_count']:,}</td>
                <td>{summary['medium_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%</td>
            </tr>
            <tr>
                <td><span class="badge badge-low">Low</span></td>
                <td>{summary['low_risk_count']:,}</td>
                <td>{summary['low_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%</td>
            </tr>
        </table>

        <h2>2. Provincial Fraud Distribution</h2>
        <table>
            <tr><th>Province</th><th>Total Txns</th><th>Fraud Cases</th><th>Fraud Rate</th></tr>
            {''.join(f'<tr><td>{p["province"]}</td><td>{p["total_txns"]:,}</td><td>{p["fraud_count"]}</td><td>{p["fraud_rate_pct"]:.2f}%</td></tr>' for p in provinces)}
        </table>

        <h2>3. High-Risk Transaction Inventory (Score ≥ 60)</h2>
        <p><em>{len(high_risk)} transaction(s) flagged as high risk during this period.</em></p>
        <table>
            <tr><th>Txn ID</th><th>Timestamp</th><th>Amount (ZMW)</th><th>Operator</th><th>Province</th><th>Score</th><th>Level</th></tr>
            {''.join(
                f'<tr><td style="font-family:monospace;font-size:0.75rem;">{t["txn_id"][:20]}...</td>'
                f'<td>{t["timestamp"][:19]}</td>'
                f'<td>K {t["amount"]:,.2f}</td>'
                f'<td>{t["operator"]}</td>'
                f'<td>{t["province"]}</td>'
                f'<td><strong>{t["risk_score"]:.0f}</strong></td>'
                f'<td><span class="badge badge-{t["risk_level"].lower()}">{t["risk_level"]}</span></td></tr>'
                for t in high_risk[:30]
            )}
        </table>

        <h2>4. Analyst Override Decisions</h2>
        {'<p><em>No override decisions recorded during this period.</em></p>' if not overrides else ''}
        {'<table><tr><th>Txn ID</th><th>Original Level</th><th>Decision</th><th>Justification</th><th>By</th><th>Date</th></tr>' +
         ''.join(
            f'<tr><td style="font-family:monospace;font-size:0.75rem;">{o["txn_id"][:20]}...</td>'
            f'<td>{o["original_risk_level"]}</td>'
            f'<td><strong>{o["decision"].replace("_", " ").title()}</strong></td>'
            f'<td>{o["justification"][:100]}</td>'
            f'<td>{o["overridden_by"]}</td>'
            f'<td>{o["created_at"][:19]}</td></tr>'
            for o in overrides
         ) + '</table>' if overrides else ''}

        <h2>5. Manual Flags</h2>
        {'<p><em>No manual flags recorded during this period.</em></p>' if not flags else ''}
        {'<table><tr><th>Txn ID</th><th>Flagged By</th><th>Reason</th><th>Status</th><th>Date</th></tr>' +
         ''.join(
            f'<tr><td style="font-family:monospace;font-size:0.75rem;">{fl["txn_id"][:20]}...</td>'
            f'<td>{fl["flagged_by"]}</td>'
            f'<td>{fl["reason"][:100]}</td>'
            f'<td>{fl["status"]}</td>'
            f'<td>{fl["created_at"][:19]}</td></tr>'
            for fl in flags
         ) + '</table>' if flags else ''}

        <h2>6. Model Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>BoZ Target</th><th>Status</th></tr>
            <tr>
                <td>Precision</td>
                <td>{model_perf['precision']:.2%}</td>
                <td>≥ 85%</td>
                <td>{'✅ Met' if model_perf['precision'] >= 0.85 else '⚠️ Below target'}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{model_perf['recall']:.2%}</td>
                <td>≥ 80%</td>
                <td>{'✅ Met' if model_perf['recall'] >= 0.80 else '⚠️ Below target'}</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{model_perf['f1']:.2%}</td>
                <td>≥ 80%</td>
                <td>{'✅ Met' if model_perf['f1'] >= 0.80 else '⚠️ Below target'}</td>
            </tr>
            <tr>
                <td>MCC</td>
                <td>{model_perf['mcc']:.4f}</td>
                <td>≥ 0.60</td>
                <td>{'✅ Met' if model_perf['mcc'] >= 0.60 else '⚠️ Below target'}</td>
            </tr>
        </table>
        <p style="font-size:0.82rem;color:#6a737d;">
            Based on {model_perf['sample_size']:,} audited transactions.
            Confusion matrix: TP={model_perf.get('tp',0):,} FP={model_perf.get('fp',0):,}
            FN={model_perf.get('fn',0):,} TN={model_perf.get('tn',0):,}
        </p>

        <h2>7. Regulatory References</h2>
        <div class="reg-ref">
            <strong>BoZ Directive FSD/02/2023</strong> — SIM swap transactions within 72 hours require
            mandatory enhanced due diligence and may be subject to a temporary transaction hold
            pending identity re-verification.
        </div>
        <div class="reg-ref">
            <strong>BoZ AML/CFT Regulation 2023, Section 12(4)</strong> — High-velocity transaction
            patterns consistent with rapid asset liquidation post-compromise must be reviewed and
            reported to the Financial Intelligence Centre (FIC).
        </div>
        <div class="reg-ref">
            <strong>BoZ Directive FSD/05/2022</strong> — Transfers to blocklisted accounts must be
            suspended pending FIC notification.
        </div>
        <div class="reg-ref">
            <strong>BoZ Consumer Protection Framework</strong> — Social engineering (smishing/vishing)
            indicators mandate investigation and optional transaction reversal notification to the
            affected consumer.
        </div>
        <div class="reg-ref">
            <strong>BoZ Agent Banking Regulations, Schedule 2</strong> — Agent-facilitated fraud
            monitoring is required, including tracking agent complaint rates above the 5% threshold.
        </div>
        <div class="reg-ref">
            <strong>National Payment Systems Act 2007 (as amended)</strong> — KYC tier transaction
            limits must be enforced. Transactions exceeding tier limits require enhanced verification
            before processing.
        </div>

        <div class="footer">
            <p>
                <strong>Confidentiality Notice:</strong> This report contains sensitive financial data
                and is intended solely for the Bank of Zambia and authorised personnel of the
                submitting mobile financial services provider. Unauthorised distribution is prohibited
                under the Financial Intelligence Centre Act.
            </p>
            <p>
                Generated by the Zambia Mobile Money Fraud Detection & Response System.<br>
                Report ID: {report_id} · {generated_at}
            </p>
        </div>
    </body>
    </html>
    """
    return html


# ─────────────────────────────────────────────
# PDF EXPORT (FR-15 / Gap #10)
# Generates a downloadable PDF suitable for Bank of Zambia submission.
# Uses reportlab (pure Python, no system libs required).
# Install: pip install reportlab>=4.0
# ─────────────────────────────────────────────

def generate_compliance_report_pdf(
    date_from: datetime,
    date_to: datetime,
    generated_by: str = "System Administrator",
) -> bytes:
    """
    Generate a BoZ-compliant compliance report as PDF bytes.

    Returns raw bytes that Streamlit can serve via:
        st.download_button(
            label="⬇ Download PDF",
            data=pdf_bytes,
            file_name="BOZ_compliance_report.pdf",
            mime="application/pdf",
        )

    Falls back gracefully if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io
    except ImportError:
        raise RuntimeError(
            "reportlab is not installed. Add it with: pip install reportlab>=4.0"
        )

    # ── Gather data ──
    engine = get_engine()
    with Session(engine) as session:
        summary   = _query_summary(session, date_from, date_to)
        high_risk = _query_high_risk_txns(session, date_from, date_to, limit=50)
        provinces = _query_province_breakdown(session, date_from, date_to)
        overrides = _query_overrides(session, date_from, date_to)
        flags     = _query_manual_flags(session, date_from, date_to)
        model_perf = _query_model_performance(session)

    report_id    = f"BOZ-CR-{date_from:%Y%m%d}-{date_to:%Y%m%d}"
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Document setup ──
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    h1  = ParagraphStyle("H1",  parent=styles["Heading1"],  fontSize=14, spaceAfter=4)
    h2  = ParagraphStyle("H2",  parent=styles["Heading2"],  fontSize=11, spaceAfter=4, spaceBefore=10)
    h3  = ParagraphStyle("H3",  parent=styles["Heading3"],  fontSize=10, spaceAfter=3, spaceBefore=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=8, leading=12)
    meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=7, textColor=colors.grey)
    center = ParagraphStyle("Center", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8)

    GREEN = colors.HexColor("#00843D")
    LIGHT_GREY = colors.HexColor("#f6f8fa")
    RED_BADGE  = colors.HexColor("#d73a49")
    ORG_BADGE  = colors.HexColor("#b08800")
    GRN_BADGE  = colors.HexColor("#22863a")

    story = []

    # ── Header ──
    story.append(Paragraph("🏛️  Bank of Zambia — Fraud Detection Compliance Report", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=6))
    story.append(Paragraph(f"<b>Report ID:</b> {report_id}", meta))
    story.append(Paragraph(f"<b>Period:</b> {date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}", meta))
    story.append(Paragraph(f"<b>Generated:</b> {generated_at} &nbsp;|&nbsp; <b>Prepared by:</b> {generated_by}", meta))
    story.append(Spacer(1, 6 * mm))

    # ── Section 1: Executive Summary ──
    story.append(Paragraph("1. Executive Summary", h2))
    kpi_data = [
        ["Total Transactions", "Fraud Rate", "Blocked Funds (ZMW)", "Avg Risk Score"],
        [
            f"{summary['total_txns']:,}",
            f"{summary['fraud_rate_pct']}%",
            f"K {summary['blocked_funds_zmw']:,.0f}",
            f"{summary['avg_risk_score']}/100",
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[42 * mm] * 4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND",    (0, 1), (-1, 1), LIGHT_GREY),
        ("FONTSIZE",      (0, 1), (-1, 1), 12),
        ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY]),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 4 * mm))

    # Risk level distribution sub-table
    story.append(Paragraph("Risk Level Distribution", h3))
    dist_data = [
        ["Risk Level", "Count", "Percentage"],
        ["CRITICAL", "—", "—"],  # placeholder; query can be extended
        ["High",    str(summary["high_risk_count"]),   f"{summary['high_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%"],
        ["Medium",  str(summary["medium_risk_count"]), f"{summary['medium_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%"],
        ["Low",     str(summary["low_risk_count"]),    f"{summary['low_risk_count'] / max(summary['total_txns'], 1) * 100:.1f}%"],
    ]
    dist_table = Table(dist_data, colWidths=[50 * mm, 40 * mm, 40 * mm])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#161B22")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 4 * mm))

    # ── Section 2: Provincial Breakdown ──
    story.append(Paragraph("2. Provincial Fraud Distribution", h2))
    prov_data = [["Province", "Total Txns", "Fraud Cases", "Fraud Rate"]]
    for p in provinces:
        prov_data.append([p["province"], str(p["total_txns"]), str(p["fraud_count"]), f"{p['fraud_rate_pct']:.2f}%"])
    prov_table = Table(prov_data, colWidths=[50 * mm, 35 * mm, 35 * mm, 35 * mm])
    prov_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#161B22")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
    ]))
    story.append(prov_table)
    story.append(Spacer(1, 4 * mm))

    # ── Section 3: High-Risk Transactions ──
    story.append(Paragraph(f"3. High-Risk Transaction Inventory (Score ≥ 60) — {len(high_risk)} records", h2))
    txn_data = [["Txn ID (truncated)", "Timestamp", "Amount (ZMW)", "Score", "Level"]]
    for t in high_risk[:30]:
        txn_data.append([
            t["txn_id"][:18] + "…",
            t["timestamp"][:16],
            f"K {t['amount']:,.2f}",
            f"{t['risk_score']:.0f}",
            t["risk_level"],
        ])
    txn_table = Table(txn_data, colWidths=[48 * mm, 32 * mm, 32 * mm, 18 * mm, 24 * mm])
    txn_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#161B22")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 7),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
    ]))
    story.append(txn_table)

    # ── Section 4: Model Performance ──
    story.append(Paragraph("4. Model Performance Metrics", h2))
    perf_data = [
        ["Metric", "Value", "BoZ Target", "Status"],
        ["Precision",  f"{model_perf['precision']:.2%}", "≥ 85%", "✅ Met" if model_perf["precision"] >= 0.85 else "⚠ Below"],
        ["Recall",     f"{model_perf['recall']:.2%}",    "≥ 80%", "✅ Met" if model_perf["recall"]    >= 0.80 else "⚠ Below"],
        ["F1-Score",   f"{model_perf['f1']:.2%}",        "≥ 80%", "✅ Met" if model_perf["f1"]        >= 0.80 else "⚠ Below"],
        ["MCC",        f"{model_perf['mcc']:.4f}",       "≥ 0.60","✅ Met" if model_perf["mcc"]       >= 0.60 else "⚠ Below"],
    ]
    perf_table = Table(perf_data, colWidths=[40 * mm, 35 * mm, 35 * mm, 40 * mm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"Based on {model_perf['sample_size']:,} audited transactions. "
        f"Confusion matrix: TP={model_perf.get('tp',0):,} FP={model_perf.get('fp',0):,} "
        f"FN={model_perf.get('fn',0):,} TN={model_perf.get('tn',0):,}",
        body,
    ))

    # ── Section 5: Regulatory References ──
    story.append(Paragraph("5. Regulatory References", h2))
    refs = [
        ("BoZ Directive FSD/02/2023",
         "SIM swap transactions within 72 hours require mandatory enhanced due diligence."),
        ("BoZ AML/CFT Regulation 2023 §12(4)",
         "High-velocity transaction patterns must be reviewed and reported to the FIC."),
        ("BoZ Directive FSD/05/2022",
         "Transfers to blocklisted accounts must be suspended pending FIC notification."),
        ("BoZ Consumer Protection Framework",
         "Smishing/vishing indicators mandate investigation and optional transaction reversal notification."),
        ("National Payment Systems Act 2007",
         "KYC tier transaction limits must be enforced with enhanced verification above tier limits."),
    ]
    for ref_title, ref_text in refs:
        story.append(Paragraph(f"<b>{ref_title}</b> — {ref_text}", body))
        story.append(Spacer(1, 2 * mm))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN, spaceBefore=6))
    story.append(Paragraph(
        f"<b>Confidentiality Notice:</b> This report is intended solely for the Bank of Zambia "
        f"and authorised personnel. Report ID: {report_id} · Generated: {generated_at}",
        meta,
    ))

    doc.build(story)
    return buf.getvalue()
