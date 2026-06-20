"""AI-Agent Failure Predictor — real-time risk dashboard.

A 1 MB calibrated CatBoost that out-ranks Claude Opus & Haiku at predicting whether an
LLM-agent run will fail (Phase-5 head-to-head: AUPRC 0.833 vs 0.74/0.71) — at ~70 us/run
and $0.0001/1k. This app scores a run, explains the risk by telemetry family, and shows
the early-window model raising the alarm several steps before the run actually ends.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import (
    load_champion, load_early_window, predict_run, explain_run,
    early_window_curve, early_warning_lead, risk_band,
)
from src.feature_engineering import synthesize_run, TASK_LEVELS, TIER_LEVELS

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="AI-Agent Failure Predictor", page_icon="🛰️", layout="wide")

BAND_COLOR = {"Low": "#2a9d8f", "Elevated": "#e9c46a", "High": "#f4a261", "Critical": "#e76f51"}
INK = "#264653"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1300px;}
      .hero {font-size: 2.0rem; font-weight: 800; color: #264653; margin-bottom: 0.1rem;}
      .sub  {color: #5b6770; font-size: 1.02rem; margin-bottom: 0.4rem;}
      .pill {display:inline-block; padding:2px 11px; border-radius:11px; font-size:0.82rem;
             font-weight:700; color:white; margin-right:6px;}
      .metric-big {font-size: 2.2rem; font-weight: 800;}
      .verdict-card {border-radius: 14px; padding: 16px 20px; color: white;}
      .small {color:#5b6770; font-size:0.84rem;}
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _champ():
    return load_champion()


@st.cache_resource(show_spinner=False)
def _ew():
    return load_early_window()


@st.cache_data(show_spinner=False)
def _examples():
    p = os.path.join(RESULTS, "ui_examples.json")
    return json.load(open(p)) if os.path.exists(p) else {"examples": [], "threshold": 0.5}


@st.cache_data(show_spinner=False)
def _metrics():
    p = os.path.join(RESULTS, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ---------------------------------------------------------------------------------------
# Guard: model must be trained.
# ---------------------------------------------------------------------------------------
try:
    champ = _champ()
    ew = _ew()
except FileNotFoundError:
    st.error("Model artefacts not found. Build them first:\n\n```\npython -m src.train\n```")
    st.stop()

examples = _examples()
metrics = _metrics()
tm = champ["test_metrics"]
op = champ["operating_point"]


# =======================================================================================
# Sidebar — model card + the LLM head-to-head selling point
# =======================================================================================
with st.sidebar:
    st.markdown("### 🛰️ Model")
    st.caption(champ["champion"])
    c1, c2 = st.columns(2)
    c1.metric("Test AUPRC", f"{tm['auprc']:.3f}")
    c2.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    c1.metric("Brier", f"{tm['brier']:.3f}", help="Calibration error (lower = better)")
    c2.metric("Deployed recall", f"{op['honest_recall']:.2f}",
              help=f"at precision {op['honest_precision']:.2f}, threshold {op['threshold']:.2f}")

    st.divider()
    st.markdown("#### vs. frontier LLMs (zero-shot)")
    st.caption("Phase-5 head-to-head · same 50 runs")
    hh = pd.DataFrame([
        {"Model": "This model", "AUPRC": 0.833, "Latency": "70 µs", "Cost/1k": "$0.0001"},
        {"Model": "Claude Opus", "AUPRC": 0.738, "Latency": "10.3 s", "Cost/1k": "$4.50"},
        {"Model": "Claude Haiku", "AUPRC": 0.709, "Latency": "23.9 s", "Cost/1k": "$0.30"},
    ])
    st.dataframe(hh, hide_index=True, use_container_width=True)
    st.caption("≈147,000× faster · ~45,000× cheaper · and it out-ranks both. "
               "LLMs cry wolf (recall ~0.95, precision ~0.5); the tree is calibrated.")

    st.divider()
    with st.expander("How it works"):
        st.markdown(
            "- **Input**: one agent run's telemetry — steps, tool calls, errors, retries, "
            "context growth, latency, plus per-step traces.\n"
            "- **49 features**: 26 run-level + 16 leading-indicator trajectory features "
            "(velocity/accel/early-warning) + 7 domain interactions.\n"
            "- **Model**: Optuna-tuned CatBoost, sigmoid-calibrated, thresholded at "
            f"**P≥0.80** (operating point {op['threshold']:.2f}).\n"
            "- **Early-window**: separate calibrated models on the first *k* steps only — "
            "k=3 already recovers ~78% of full-run AUPRC. Failure is visible early.\n"
            "- **Ceiling**: the residual is signal-bound (latent capability gap + noise) — "
            "confirmed 5 independent ways across Phases 2–5."
        )


# =======================================================================================
# Header
# =======================================================================================
st.markdown('<div class="hero">AI-Agent Failure Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Predict whether an LLM-agent run will fail — and catch it '
    '<b>several steps before it does</b>. A 1 MB calibrated tree that out-ranks Claude '
    'Opus &amp; Haiku at this task.</div>',
    unsafe_allow_html=True,
)


# =======================================================================================
# Input selector
# =======================================================================================
mode = st.radio("Run source", ["📁 Pick a real agent run", "🎛️ Build a run (what-if)"],
                horizontal=True, label_visibility="collapsed")

run: dict | None = None
truth: dict | None = None

if mode.startswith("📁"):
    if not examples["examples"]:
        st.warning("No curated examples found. Run `python -m src.train` to generate them.")
        st.stop()

    def _label(ex):
        n = ex["num_steps"]
        if ex["_true_label"]:
            return f"❌ FAIL · {ex.get('failure_reason','?')} · {n} steps · {ex['task_type']}/{ex['model_tier']}"
        return f"✅ success · {n} steps · {ex['task_type']}/{ex['model_tier']}"

    opts = list(range(len(examples["examples"])))
    sel = st.selectbox("Choose a held-out run", opts, format_func=lambda i: _label(examples["examples"][i]))
    run = examples["examples"][sel]
    truth = {"label": run["_true_label"], "prob": run["_champion_prob"], "correct": run.get("_correct")}
else:
    cc = st.columns(4)
    task = cc[0].selectbox("Task type", TASK_LEVELS, index=1)
    tier = cc[1].selectbox("Model tier", TIER_LEVELS, index=1)
    steps = cc[2].slider("Steps", 3, 30, 12)
    temp = cc[3].slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    cc2 = st.columns(4)
    err_rate = cc2[0].slider("Tool error rate", 0.0, 1.0, 0.35, 0.05)
    retries = cc2[1].slider("Max consecutive retries", 0, 5, 1)
    ctx_max = cc2[2].slider("Context max (% of budget)", 0.1, 1.0, 0.55, 0.05)
    loops = cc2[3].slider("Reasoning loops", 0, 6, 0)
    run = synthesize_run(num_steps=steps, task_type=task, model_tier=tier,
                         tool_error_rate=err_rate, max_consecutive_retries=retries,
                         context_max_pct=ctx_max, reasoning_loops=loops, temperature=temp)
    st.caption("Hand-built hypothetical — a deterministic trace consistent with these "
               "knobs (not sampled telemetry).")


# =======================================================================================
# Score
# =======================================================================================
pred = predict_run(run, champ)
prob = pred["failure_probability"]
band = pred["risk_band"]
color = BAND_COLOR[band]


# --- Row 1: gauge + verdict ------------------------------------------------------------
left, right = st.columns([1.05, 1.0])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        title={"text": "Failure probability", "font": {"size": 16, "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": INK},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 15], "color": "#eaf4f1"},
                {"range": [15, 40], "color": "#fbf3d8"},
                {"range": [40, 65], "color": "#fde4d4"},
                {"range": [65, 100], "color": "#f9d8cf"},
            ],
            "threshold": {"line": {"color": INK, "width": 4}, "thickness": 0.82,
                          "value": (champ["threshold"] or 0.5) * 100},
        },
    ))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Dark needle = deployed alarm threshold ({(champ['threshold'] or 0.5):.2f}, "
               f"set for precision ≥ {op['operating_precision']:.0%}).")

with right:
    verdict = "WILL LIKELY FAIL" if pred["predicted_failure"] else "likely to succeed"
    st.markdown(
        f'<div class="verdict-card" style="background:{color};">'
        f'<div style="font-size:0.9rem;opacity:0.9;">Verdict · risk band</div>'
        f'<div class="metric-big">{band}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;">{verdict}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if truth is not None:
        tlab = "FAILED" if truth["label"] else "SUCCEEDED"
        ok = (pred["predicted_failure"] == bool(truth["label"]))
        st.metric("Ground truth (held-out run)", tlab,
                  delta="model correct ✓" if ok else "model wrong ✗",
                  delta_color="normal" if ok else "inverse")
        if run.get("failure_reason") and truth["label"]:
            st.caption(f"Failure reason: **{run['failure_reason']}**")
    lead = early_warning_lead(run, alert_prob=0.5, ew=ew)
    if lead["alerted"] and pred["predicted_failure"]:
        st.success(f"⏱️ Early-warning model crosses 50% risk at **step {lead['alert_step']}** — "
                   f"**{lead['steps_early']} step(s) before** the run's {lead['n_steps']}-step end.")
    elif lead["alerted"]:
        st.warning(f"⚠️ Early-warning model crosses 50% risk at step {lead['alert_step']}, but the "
                   f"full run scores **below** the failure threshold — this would be a **false alarm**, "
                   f"not a catch.")
    else:
        st.info("Early-warning model stays below 50% across the observed window.")


# --- Row 2: early-window timeline + contributing factors -------------------------------
st.write("")
c_left, c_right = st.columns([1.05, 1.0])

with c_left:
    st.markdown("##### ⏱️ Risk as the run unfolds (early-window model)")
    curve = early_window_curve(run, ew)
    if curve["points"]:
        ks = [p["k"] for p in curve["points"]]
        ps = [p["prob"] * 100 for p in curve["points"]]
        fline = go.Figure()
        fline.add_trace(go.Scatter(x=ks, y=ps, mode="lines+markers",
                                   line=dict(color=INK, width=3), marker=dict(size=8),
                                   name="early-window risk"))
        fline.add_hline(y=50, line_dash="dot", line_color="#e76f51",
                        annotation_text="alert level 50%", annotation_position="top left")
        fline.add_trace(go.Scatter(x=[curve["n_steps"]], y=[prob * 100], mode="markers+text",
                                    marker=dict(size=13, color=color, symbol="star"),
                                    text=["full-run"], textposition="top center",
                                    name="full-run verdict"))
        fline.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="steps observed (k)", yaxis_title="failure risk (%)",
                            yaxis_range=[0, 100], paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fline, use_container_width=True, config={"displayModeBar": False})
        st.caption("Each point: the model's risk estimate if it had seen only the first *k* "
                   "steps. k=3 already recovers ~78% of full-run AUPRC.")
    else:
        st.caption("Run too short for the early-window grid.")

with c_right:
    st.markdown("##### 🔍 What's driving the risk (SHAP by telemetry family)")
    exp = explain_run(run, champ, top_n=8)
    if exp["available"]:
        g = pd.DataFrame(exp["groups"]).sort_values("shap")
        bar_colors = ["#e76f51" if "Error" in name else "#90a4ae" for name in g["group"]]
        fg = go.Figure(go.Bar(x=g["shap"], y=g["group"], orientation="h",
                              marker_color=bar_colors))
        fg.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="contribution to failure log-odds",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fg.add_vline(x=0, line_color=INK, line_width=1)
        st.plotly_chart(fg, use_container_width=True, config={"displayModeBar": False})
        st.caption("→ pushes toward **failure**, ← toward **success**. The "
                   "**Error/Retry/Loop** family (orange) is the load-bearing signal "
                   "(Phase-5 ablation: −0.022 AUPRC when removed).")
    else:
        st.caption("SHAP twin unavailable in this artefact.")

with st.expander("Top individual feature contributions"):
    if exp["available"]:
        ft = pd.DataFrame(exp["features"])[["feature", "value", "shap"]]
        ft.columns = ["feature", "run value", "SHAP (log-odds)"]
        st.dataframe(ft.round(4), hide_index=True, use_container_width=True)


# --- Row 3: the run's telemetry traces -------------------------------------------------
with st.expander("📈 Per-step telemetry traces"):
    n = run["num_steps"]
    x = list(range(1, n + 1))
    tfig = go.Figure()
    tfig.add_trace(go.Scatter(x=x, y=[c * 100 for c in run["trace_ctx_pct"]],
                              name="context %", line=dict(color="#2a9d8f")))
    tfig.add_trace(go.Bar(x=x, y=run["trace_err"], name="error", marker_color="#e76f51",
                          opacity=0.6, yaxis="y2"))
    tfig.add_trace(go.Bar(x=x, y=run["trace_retry"], name="retries", marker_color="#e9c46a",
                          opacity=0.6, yaxis="y2"))
    tfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="step", yaxis_title="context (%)",
                       yaxis2=dict(title="errors / retries", overlaying="y", side="right",
                                   showgrid=False),
                       legend=dict(orientation="h", y=1.15), barmode="overlay",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(tfig, use_container_width=True, config={"displayModeBar": False})

st.divider()
st.caption("AI-Agent Failure Predictor · Phase 6 production pipeline · synthetic telemetry "
           "calibrated to the MAST agent-failure taxonomy. The ~0.62 AUPRC ceiling is "
           "signal-bound (irreducible latent capability gap) — see README for the 5-phase "
           "research log.")
