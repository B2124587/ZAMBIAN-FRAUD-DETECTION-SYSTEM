"""
MODULE 2: UI Components & Visualisations
Mobile Money Fraud Detection System - Zambia
dashboard_ui.py

Helper functions that transform DataFrames into styled Plotly figures
and Streamlit UI components for the Sprint 3 dashboard.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────

PALETTE = {
    "bg_primary":    "#0D1117",
    "bg_card":       "#161B22",
    "bg_elevated":   "#1C2333",
    "accent_teal":   "#00E5C3",
    "accent_amber":  "#FFB627",
    "accent_red":    "#FF4D6D",
    "accent_blue":   "#4D9FEC",
    "text_primary":  "#E6EDF3",
    "text_muted":    "#7D8590",
    "border":        "#30363D",
    "mtn_yellow":    "#FFCB05",
    "airtel_red":    "#E40613",
    "zamtel_green":  "#00843D",
}

RISK_COLORS = {
    "Low":      "#00E5C3",
    "Medium":   "#FFB627",
    "High":     "#FF4D6D",
    "Critical": "#900C3F",
}

OPERATOR_COLORS = {
    "MTN":    PALETTE["mtn_yellow"],
    "Airtel": PALETTE["airtel_red"],
    "Zamtel": PALETTE["zamtel_green"],
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=PALETTE["text_primary"]),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text_muted"], size=12),
    ),
)
# NOTE: `showlegend` is intentionally NOT included here. Each chart function
# below sets `showlegend` explicitly to whatever it needs (True/False) in its
# own update_layout(**CHART_LAYOUT, ...) call. If it were baked into
# CHART_LAYOUT, any chart that also passed showlegend=... explicitly would
# hit: "TypeError: update_layout() got multiple values for keyword argument
# 'showlegend'" — which is exactly what was happening before this fix.

# ─────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────

def inject_css():
    st.markdown(
        """
        <style>
        /* ── Global ── */
        html, body, [data-testid="stApp"] {
            background-color: #0D1117;
            color: #E6EDF3;
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: #161B22 !important;
            border-right: 1px solid #30363D;
        }
        [data-testid="stSidebar"] * { color: #E6EDF3 !important; }

        /* ── KPI Cards ── */
        .kpi-card {
            background: #161B22;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            transition: border-color .2s;
        }
        .kpi-card:hover { border-color: #00E5C3; }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.1;
            color: #E6EDF3;
            letter-spacing: -0.02em;
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: #7D8590;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 6px;
        }
        .kpi-accent { color: #00E5C3; }
        .kpi-warn   { color: #FFB627; }
        .kpi-danger { color: #FF4D6D; }

        /* ── Section headers ── */
        .section-header {
            font-size: 0.7rem;
            font-weight: 600;
            color: #7D8590;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-bottom: 1px solid #30363D;
            padding-bottom: 8px;
            margin: 24px 0 16px 0;
        }

        /* ── Alert badge ── */
        .risk-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .risk-High   { background: rgba(255,77,109,.15); color: #FF4D6D; }
        .risk-Medium { background: rgba(255,182,39,.15); color: #FFB627; }
        .risk-Low    { background: rgba(0,229,195,.12); color: #00E5C3; }

        /* ── Detail panel ── */
        .detail-panel {
            background: #1C2333;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 12px;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #30363D;
        }
        .detail-key   { color: #7D8590; font-size: 0.82rem; }
        .detail-value { color: #E6EDF3; font-size: 0.88rem; font-weight: 500; }

        /* ── Reason code chip ── */
        .reason-chip {
            display: inline-block;
            background: rgba(77,159,236,.1);
            border: 1px solid rgba(77,159,236,.3);
            color: #4D9FEC;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.75rem;
            margin: 3px 3px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* ── Metric gauge ── */
        .gauge-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 8px 0;
        }
        .gauge-label { color: #7D8590; font-size: 0.8rem; width: 110px; }
        .gauge-bar-bg {
            flex: 1; background: #30363D; border-radius: 4px; height: 8px;
        }
        .gauge-bar { height: 8px; border-radius: 4px; }
        .gauge-value { color: #E6EDF3; font-size: 0.85rem; font-weight: 600; width: 52px; text-align: right; }

        /* ── Connection dot ── */
        .conn-dot { display: inline-block; width: 10px; height: 10px;
                    border-radius: 50%; margin-right: 6px; }
        .conn-ok   { background: #00E5C3; box-shadow: 0 0 8px #00E5C3; }
        .conn-fail { background: #FF4D6D; }

        /* ── Tabs ── */
        [data-testid="stTabs"] button {
            color: #7D8590 !important;
            font-size: 0.88rem;
            font-weight: 500;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #00E5C3 !important;
            border-bottom-color: #00E5C3 !important;
        }

        /* ── DataFrames ── */
        .stDataFrame { border-radius: 8px; overflow: hidden; }

        /* ── Hide Streamlit branding ── */
        #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# KPI CARD
# ─────────────────────────────────────────────

def kpi_card(label: str, value: str, accent_class: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-value {accent_class}">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_kpi_row(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                "Total Transactions",
                f"{metrics['total_txns']:,}",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card(
                "Fraud Prevalence Rate",
                f"{metrics['fraud_rate']}%",
                "kpi-danger",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card(
                "Fraud Funds Blocked (ZMW)",
                f"K {metrics['blocked_funds_zmw']:,.0f}",
                "kpi-warn",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                "Avg Risk Score",
                f"{metrics['avg_risk_score']}/100",
                "kpi-accent",
            ),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# CONNECTION STATUS
# ─────────────────────────────────────────────

def render_connection_status(connected: bool):
    dot = "conn-ok" if connected else "conn-fail"
    status = "Connected" if connected else "Disconnected"
    st.sidebar.markdown(
        f"""
        <div style="margin-bottom:20px; padding:10px 14px; background:#1C2333;
                    border-radius:8px; border:1px solid #30363D;">
            <span class="conn-dot {dot}"></span>
            <span style="font-size:0.82rem; font-weight:600;">{status}</span>
            <div style="font-size:0.7rem; color:#7D8590; margin-top:3px;">
                mobile_money_fraud · MySQL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PROVINCE BAR CHART
# ─────────────────────────────────────────────

def province_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No province data available")

    df = df.sort_values("fraud_count", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["fraud_count"],
            y=df["province"],
            orientation="h",
            marker=dict(
                color=df["fraud_count"],
                colorscale=[[0, "#1C2333"], [0.5, "#FFB627"], [1.0, "#FF4D6D"]],
                showscale=False,
                line=dict(width=0),
            ),
            text=df["fraud_count"],
            textposition="outside",
            textfont=dict(color=PALETTE["text_primary"], size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Fraud cases: %{x}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Fraud Incidents by Province",
            font=dict(size=13, color=PALETTE["text_muted"]),
            x=0,
        ),
        xaxis=dict(
            gridcolor="#30363D",
            zerolinecolor="#30363D",
            title=dict(text="Fraud Cases", font=dict(size=11, color=PALETTE["text_muted"])),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            title=None,
        ),
        height=340,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# OPERATOR DONUT CHART
# ─────────────────────────────────────────────

def operator_donut_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No operator data available")

    colors = [OPERATOR_COLORS.get(op, PALETTE["accent_blue"]) for op in df["operator"]]
    fig = go.Figure(
        go.Pie(
            labels=df["operator"],
            values=df["fraud_count"],
            hole=0.60,
            marker=dict(colors=colors, line=dict(color="#0D1117", width=3)),
            textinfo="label+percent",
            textfont=dict(size=12, color=PALETTE["text_primary"]),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Fraud cases: %{value}<br>"
                "Share: %{percent}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Fraud by Operator",
            font=dict(size=13, color=PALETTE["text_muted"]),
            x=0,
        ),
        height=340,
        annotations=[
            dict(
                text="Operator<br>Split",
                x=0.5, y=0.5,
                font=dict(size=11, color=PALETTE["text_muted"]),
                showarrow=False,
            )
        ],
    )
    return fig


# ─────────────────────────────────────────────
# DAILY TREND LINE
# ─────────────────────────────────────────────

def daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No trend data available")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["txn_date"],
            y=df["total_txns"],
            name="Total Txns",
            line=dict(color=PALETTE["accent_blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(77,159,236,0.08)",
            hovertemplate="Date: %{x}<br>Total: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["txn_date"],
            y=df["fraud_count"],
            name="Fraud",
            line=dict(color=PALETTE["accent_red"], width=2, dash="dot"),
            hovertemplate="Date: %{x}<br>Fraud: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Daily Transaction & Fraud Volume",
            font=dict(size=13, color=PALETTE["text_muted"]),
            x=0,
        ),
        xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
        height=260,
    )
    return fig


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────

# Simulated SHAP global importance weights (representative for this feature set)
FEATURE_IMPORTANCE = {
    "sim_swap_72h":              0.248,
    "amount_to_average_ratio":   0.187,
    "new_beneficiary_flag":      0.142,
    "velocity_24h":              0.118,
    "smishing_correlation_30m":  0.099,
    "new_device_flag":           0.081,
    "location_deviation_km":     0.063,
    "off_hours_flag":            0.028,
    "kyc_limit_exceeded":        0.022,
    "agent_complaint_rate":      0.009,
    "velocity_1h":               0.003,
}


def feature_importance_chart() -> go.Figure:
    items = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]
    bar_colors = [
        PALETTE["accent_red"] if v >= 0.15
        else PALETTE["accent_amber"] if v >= 0.07
        else PALETTE["accent_teal"]
        for v in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=PALETTE["text_primary"], size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Feature Importance (Global SHAP Weights)",
            font=dict(size=13, color=PALETTE["text_muted"]),
            x=0,
        ),
        xaxis=dict(
            gridcolor="#30363D",
            zerolinecolor="#30363D",
            title=dict(text="Mean |SHAP value|", font=dict(size=11, color=PALETTE["text_muted"])),
            tickformat=".3f",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
        height=420,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# RISK SCORE GAUGE
# ─────────────────────────────────────────────

def risk_gauge(score: float) -> go.Figure:
    color = (
        PALETTE["accent_red"] if score >= 60
        else PALETTE["accent_amber"] if score >= 30
        else PALETTE["accent_teal"]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(color=color, size=36)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=PALETTE["text_muted"],
                    tickfont=dict(color=PALETTE["text_muted"], size=10),
                ),
                bar=dict(color=color, thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30],  color="#1C2333"),
                    dict(range=[30, 60], color="#1C2333"),
                    dict(range=[60, 100], color="#1C2333"),
                ],
                threshold=dict(
                    line=dict(color=color, width=3),
                    thickness=0.8,
                    value=score,
                ),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text_primary"]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    return fig


# ─────────────────────────────────────────────
# MODEL METRICS GAUGES
# ─────────────────────────────────────────────

def render_model_metrics(metrics: dict):
    targets = {"precision": 0.85, "recall": 0.80, "f1": 0.80, "mcc": 0.60}
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "mcc": "MCC",
    }
    for key, label in labels.items():
        val = metrics.get(key, 0.0)
        target = targets[key]
        pct = min(val * 100, 100)
        color = PALETTE["accent_teal"] if val >= target else PALETTE["accent_red"]
        target_str = f"{int(target*100)}%"
        icon = "✓" if val >= target else "✗"
        st.markdown(
            f"""
            <div class="gauge-row">
                <div class="gauge-label">{label}</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <div class="gauge-value" style="color:{color};">{val:.1%}</div>
                <div style="font-size:0.72rem;color:#7D8590;width:80px;text-align:right;">
                    target ≥{target_str} {icon}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# TRANSACTION DETAIL PANEL
# ─────────────────────────────────────────────

def render_transaction_detail(row: pd.Series, features: dict):
    risk_level = row.get("risk_level", "Unknown")
    risk_color = RISK_COLORS.get(risk_level, PALETTE["text_muted"])
    score = float(row.get("numeric_score", 0))
    reason_codes = json.loads(row.get("reason_codes_json", "[]") or "[]")

    st.markdown(
        f"""
        <div class="detail-panel">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div>
                    <div style="font-size:0.72rem;color:#7D8590;text-transform:uppercase;
                                letter-spacing:.08em;">Transaction Detail</div>
                    <div style="font-size:1rem;font-weight:600;font-family:monospace;
                                color:#E6EDF3;margin-top:4px;">{row.get("txn_id","—")}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.8rem;font-weight:700;color:{risk_color};">
                        {score:.0f}<span style="font-size:1rem;">/100</span>
                    </div>
                    <span class="risk-badge risk-{risk_level}">{risk_level} Risk</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Transaction fields ──
    fields = [
        ("Timestamp",        row.get("timestamp", "—")),
        ("Sender (masked)",  row.get("masked_sender", row.get("sender_msisdn","—")[:16]+"****")),
        ("Amount (ZMW)",     f"K {float(row.get('amount',0)):,.2f}"),
        ("Operator",         row.get("operator", "—")),
        ("Province",         row.get("province", "—")),
        ("Transaction Type", row.get("txn_type", "—")),
        ("Channel",          row.get("channel", "—")),
        ("Fraud Probability",f"{float(row.get('fraud_probability',0)):.2%}"),
        ("Ground Truth",     "⚠ Confirmed Fraud" if row.get("is_fraud") else "✓ Legitimate"),
    ]
    for label, val in fields:
        st.markdown(
            f"""<div class="detail-row">
                    <div class="detail-key">{label}</div>
                    <div class="detail-value">{val}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── Reason codes ──
    if reason_codes:
        st.markdown(
            '<div style="margin-top:14px;margin-bottom:6px;font-size:0.72rem;'
            'color:#7D8590;text-transform:uppercase;letter-spacing:.08em;">'
            "Rule Triggers</div>",
            unsafe_allow_html=True,
        )
        chips = "".join(f'<span class="reason-chip">{rc}</span>' for rc in reason_codes)
        st.markdown(chips, unsafe_allow_html=True)

    # ── BoZ-compliant narrative ──
    if reason_codes:
        st.markdown(
            '<div style="margin-top:16px;font-size:0.72rem;color:#7D8590;'
            'text-transform:uppercase;letter-spacing:.08em;">BoZ Compliance Narrative</div>',
            unsafe_allow_html=True,
        )
        narrative = _generate_boz_narrative(reason_codes, score, risk_level)
        st.markdown(
            f'<div style="margin-top:8px;padding:12px 16px;background:#0D1117;'
            f'border-left:3px solid #4D9FEC;border-radius:0 6px 6px 0;'
            f'font-size:0.83rem;color:#C9D1D9;line-height:1.6;">{narrative}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature vector ──
    if features:
        st.markdown(
            '<div class="section-header" style="margin-top:20px;">Engineered Feature Vector</div>',
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(
            [{"Feature": k.replace("_", " ").title(), "Value": v}
             for k, v in features.items()]
        )
        st.dataframe(
            feat_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width=280),
                "Value":   st.column_config.NumberColumn("Value", format="%.4f"),
            },
        )


# ─────────────────────────────────────────────
# ALERTS TABLE
# ─────────────────────────────────────────────

def render_alerts_table(df: pd.DataFrame):
    if df.empty:
        st.info("No high-risk transactions matching current filters.")
        return

    display_df = df[[
        "txn_id", "timestamp", "masked_sender", "amount",
        "operator", "province", "numeric_score", "risk_level",
    ]].copy()
    display_df.columns = [
        "Transaction ID", "Timestamp", "Sender (masked)", "Amount (ZMW)",
        "Operator", "Province", "Risk Score", "Risk Level",
    ]
    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Transaction ID":    st.column_config.TextColumn(width=300),
            "Amount (ZMW)":      st.column_config.NumberColumn(format="K %.2f"),
            "Risk Score":        st.column_config.ProgressColumn(
                                    "Risk Score", min_value=0, max_value=100, format="%.1f"
                                 ),
            "Risk Level":        st.column_config.TextColumn("Risk Level"),
        },
    )


# ─────────────────────────────────────────────
# CONFUSION MATRIX MINI-TABLE
# ─────────────────────────────────────────────

def render_confusion_matrix(metrics: dict):
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tn = metrics.get("tn", 0)
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;text-align:center;">
          <thead>
            <tr style="color:#7D8590;">
              <th></th>
              <th>Predicted Fraud</th>
              <th>Predicted Legit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="color:#7D8590;text-align:left;padding:6px 0;">Actual Fraud</td>
              <td style="background:rgba(0,229,195,.1);color:#00E5C3;
                         border-radius:6px;padding:8px;">{tp:,} TP</td>
              <td style="background:rgba(255,77,109,.1);color:#FF4D6D;
                         border-radius:6px;padding:8px;">{fn:,} FN</td>
            </tr>
            <tr>
              <td style="color:#7D8590;text-align:left;padding:6px 0;">Actual Legit</td>
              <td style="background:rgba(255,77,109,.1);color:#FF4D6D;
                         border-radius:6px;padding:8px;">{fp:,} FP</td>
              <td style="background:rgba(0,229,195,.1);color:#00E5C3;
                         border-radius:6px;padding:8px;">{tn:,} TN</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# SCORE DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────

def score_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No score data available")

    colors = []
    for bucket in df["score_bucket"]:
        if bucket >= 60:
            colors.append(PALETTE["accent_red"])
        elif bucket >= 30:
            colors.append(PALETTE["accent_amber"])
        else:
            colors.append(PALETTE["accent_teal"])

    fig = go.Figure(
        go.Bar(
            x=df["score_bucket"],
            y=df["count"],
            marker=dict(color=colors, line=dict(width=0)),
            width=8,
            hovertemplate="Score bucket: %{x}–%{x}+10<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Risk Score Distribution",
            font=dict(size=13, color=PALETTE["text_muted"]),
            x=0,
        ),
        xaxis=dict(
            title=dict(text="Risk Score", font=dict(size=11, color=PALETTE["text_muted"])),
            gridcolor="#30363D",
            dtick=10,
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(size=11, color=PALETTE["text_muted"])),
            gridcolor="#30363D",
        ),
        height=240,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# BOZ NARRATIVE GENERATOR
# ─────────────────────────────────────────────

_BOZ_TEMPLATES = {
    "R01": (
        "A SIM swap event was detected on the sender's MSISDN within the preceding 72 hours. "
        "Per Bank of Zambia Directive FSD/02/2023, transactions following a SIM swap within "
        "72 hours require mandatory enhanced due diligence and may be subject to a temporary "
        "transaction hold pending identity re-verification."
    ),
    "R02": (
        "This transaction was initiated from an unregistered device to a first-time beneficiary, "
        "representing a dual-vector account-takeover pattern. BoZ AML/CFT guidelines require "
        "multi-factor authentication confirmation before processing."
    ),
    "R03": (
        "The transaction amount significantly exceeds the sender's 30-day average spend profile, "
        "triggering an anomalous value alert consistent with account-takeover or social engineering "
        "scenarios identified in BoZ Financial Intelligence Centre (FIC) typology reports."
    ),
    "R04": (
        "Unusually high transaction velocity within a 24-hour window was detected. This pattern "
        "is consistent with rapid asset liquidation post-compromise and must be reviewed per "
        "BoZ Anti-Money Laundering (AML) Regulation 2023, Section 12(4)."
    ),
    "R05": (
        "A smishing or vishing signal was correlated within 30 minutes preceding this transaction. "
        "This is a high-confidence social engineering indicator. BoZ Consumer Protection Framework "
        "mandates investigation and optional transaction reversal notification."
    ),
    "R06": (
        "The transaction originated from a geographic location significantly deviating from the "
        "sender's established activity centroid. Geographic displacement fraud is a documented "
        "typology in the BoZ Annual Fraud Report. Location verification is recommended."
    ),
    "R07": (
        "The receiving MSISDN appears on the internal fraud blocklist. All transfers to blocklisted "
        "accounts must be suspended pending FIC notification per BoZ Directive FSD/05/2022."
    ),
    "R08": (
        "The transaction was conducted outside the sender's typical operating hours, which combined "
        "with other risk indicators elevates the composite risk score."
    ),
    "R09": (
        "The receiving mobile money agent has an elevated consumer complaint rate. Monitoring of "
        "agent-facilitated fraud is required under BoZ Agent Banking Regulations, Schedule 2."
    ),
    "R10": (
        "The transaction amount exceeds the sender's KYC tier transaction limit as defined under "
        "BoZ National Payment Systems Act 2007 (as amended). Enhanced verification is mandatory "
        "before processing proceeds."
    ),
}


def _generate_boz_narrative(reason_codes: list[str], score: float, risk_level: str) -> str:
    paragraphs = []
    for rc in reason_codes:
        code = rc[:3]
        if code in _BOZ_TEMPLATES:
            paragraphs.append(f"<b>{rc}</b> — {_BOZ_TEMPLATES[code]}")

    if not paragraphs:
        return "No specific BoZ rule triggers identified for this transaction."

    header = (
        f"<b>Risk Assessment Summary</b> (Score: {score:.0f}/100 · {risk_level} Risk)<br><br>"
    )
    return header + "<br><br>".join(paragraphs)


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=PALETTE["text_muted"], size=13),
        )],
        height=300,
    )
    return fig


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)