"""
Mumbai Monsoon Risk Calculator — Streamlit edition.

Monte Carlo cost and schedule contingency for construction projects exposed to
the south-west monsoon. Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
import textwrap
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

# ----------------------------------------------------------------------
# page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Monsoon Risk Calculator",
                  layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**Monsoon Risk Calculator** — Monte Carlo cost and schedule "
                 "contingency for construction exposed to the south-west monsoon. "
                 "Results follow from the assumptions you enter; they are not forecasts.",
    },
                   )

NAVY, ORANGE, RED, GREEN, MUT = "#1F3864", "#C55A11", "#B00020", "#2E7D32", "#6B7280"

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1, h2, h3 {color: #1F3864;}
  [data-testid="stMetricValue"] {font-size: 1.45rem;}
  [data-testid="stMetricLabel"] {font-size: .80rem; color: #6B7280;}
  .readnote {background:#F4F7FB; border-left:3px solid #1F3864; padding:.7rem .9rem;
             border-radius:4px; font-size:.86rem; color:#374151; margin-top:.4rem;}
  .plain {background:#FFF8E7; border-left:3px solid #C55A11; padding:.9rem 1.1rem;
          border-radius:4px; font-size:.95rem; line-height:1.55;}
  .stTabs [data-baseweb="tab"] {font-size: .93rem; padding: .35rem .85rem;}
  @media (prefers-color-scheme: dark) {
    .readnote {background:#1C2433; color:#D5DBE5;}
    .plain {background:#2A2317; color:#EADFC8;}
  }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_white", height=340,
    margin=dict(l=55, r=20, t=48, b=48),
    font=dict(family="Georgia, serif", size=12),
    hoverlabel=dict(font_size=12),
)

DEFAULT_ROWS = [
    ("A1",  "Site clearance and enabling works",          18,  12,  "",        0,  1,  4,  0.0, 0.2, 1.0),
    ("A2",  "Excavation, shoring and dewatering",         35,  28,  "A1",      0,  2,  8,  0.0, 0.8, 4.5),
    ("A3",  "PCC, footing reinforcement and shuttering",  29,  55,  "A2",      0,  1,  5,  0.0, 0.8, 4.0),
    ("A4",  "Footing and column RCC up to plinth",        23,  65,  "A3",      0,  2,  6,  0.0, 1.0, 5.0),
    ("A5",  "Backfilling and plinth beam RCC",            21,  32,  "A4",      0,  1,  3,  0.1, 0.8, 4.0),
    ("A6",  "Superstructure RCC, ground to 2nd floor",    70, 150,  "A5",      5, 15, 41,  3.0, 10.0, 30.0),
    ("A7",  "Superstructure RCC, 3rd to 4th and terrace", 53,  95,  "A6",      2,  8, 22,  1.5, 6.0, 20.0),
    ("A8",  "Block masonry, all floors",                  47,  60,  "A7",      0,  1,  2,  0.0, 1.0, 5.0),
    ("A9",  "MEP services, rough-in and fit-out",         42, 110,  "A8",      0,  1,  4,  0.0, 1.0, 5.0),
    ("A10", "External plaster and waterproofing",         41,  65,  "A8",      0,  2,  6,  0.3, 3.0, 14.0),
    ("A11", "Internal plaster, flooring and painting",    64, 163,  "A9;A10",  0,  2,  6,  0.0, 1.5, 8.0),
    ("A12", "External development, paving and drainage",  29,  23,  "A10",     0,  1,  5,  0.0, 0.5, 3.0),
]
COLS = ["id", "name", "duration", "cost", "predecessors",
        "delay_O", "delay_M", "delay_P", "cost_O", "cost_M", "cost_P"]


def default_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_ROWS, columns=COLS)


# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------
if "acts_df" not in st.session_state:
    st.session_state.acts_df = default_df()
for k, v in dict(result=None, scenario_label="Typical monsoon", rain_mult=None,
                 ran_once=False, scen=None, tests=None, stab=None).items():
    st.session_state.setdefault(k, v)


def df_to_activities(df: pd.DataFrame):
    acts, bad = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            pre = str(row["predecessors"] or "").replace(",", ";")
            acts.append(E.Activity(
                id=str(row["id"]).strip(),
                name=str(row["name"] or "").strip(),
                dur=float(row["duration"]), cost=float(row["cost"]),
                pre=[p.strip() for p in pre.split(";") if p.strip()],
                dO=float(row["delay_O"]), dM=float(row["delay_M"]), dP=float(row["delay_P"]),
                cO=float(row["cost_O"]), cM=float(row["cost_M"]), cP=float(row["cost_P"])))
        except (TypeError, ValueError):
            bad.append(f"Row {i+1}: every numeric field must contain a number.")
    return acts, bad


def fmt(x, d=1):
    return f"{x:,.{d}f}"


# ======================================================================
# SIDEBAR — settings
# ======================================================================
with st.sidebar:
    st.markdown("### Project profile")
    start = st.date_input("Planned start date", value=date(2027, 2, 1),
                          help="Decides which stages fall inside June to September. "
                               "This one does change the result.")
    c1, c2 = st.columns(2)
    budget = c1.number_input("Budget (₹ lakh)", min_value=1.0, value=900.0, step=10.0)
    buf = c2.number_input("Time buffer (days)", min_value=0.0, value=30.0, step=5.0)
    area = st.number_input("Built-up area (sq ft)", min_value=0, value=60000, step=1000)
    site = st.selectbox("Site type", ["Normal", "Low-lying or reclaimed",
                                      "Near creek or sea", "Tight plot"], index=1)
    fnd = st.selectbox("Foundation", ["Footing or raft", "Piled"])
    bsmt = st.number_input("Basement levels", min_value=0, max_value=5, value=0)

    st.markdown("### Cost of each extra day")
    st.caption("Everything that accrues on a rain-stopped day. Do not repeat anything "
               "already inside an activity's cost O/M/P.")
    dc1 = st.number_input("Idle labour wages (₹ L/day)", value=0.2868, format="%.4f", step=0.01)
    dc2 = st.number_input("Equipment rent or standby", value=0.1400, format="%.4f", step=0.01)
    dc3 = st.number_input("Site staff and overheads", value=0.1400, format="%.4f", step=0.01)
    dc4 = st.number_input("Security, power, water, temp works", value=0.0400, format="%.4f", step=0.01)
    dc5 = st.number_input("Contract penalty (LD)", value=0.0000, format="%.4f", step=0.01)
    lc1, lc2 = st.columns(2)
    loan = lc1.number_input("Loan outstanding (₹ L)", min_value=0.0, value=200.0, step=10.0)
    lint = lc2.number_input("Interest (% p.a.)", min_value=0.0, value=9.0, step=0.5)
    interest_day = loan * lint / 100.0 / 365.0
    rate = dc1 + dc2 + dc3 + dc4 + dc5 + interest_day
    st.info(f"Interest works out to ₹{interest_day:.4f} L/day.  \n"
            f"**Total cost of one extra day: ₹{rate:.4f} lakh** (₹{rate*1e5:,.0f}).")

    st.markdown("### Model settings")
    rho = st.slider("Shared-monsoon correlation ρ", 0.0, 0.9, 0.5, 0.05,
                    help="One monsoon drives every activity. Pairwise correlation "
                         "between activities is ρ², not ρ.")
    n_iter = st.select_slider("Iterations", [1000, 2500, 5000, 10000, 25000, 50000, 100000],
                              value=10000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    eng = st.radio("Sampling engine", ["legacy", "precise"], horizontal=True,
                   help="legacy reproduces the original HTML tool bit for bit, so it "
                        "matches the validation report. precise uses exact beta "
                        "quantiles and a modern generator.")
    st.markdown("### Scenario multipliers")
    m1, m2, m3 = st.columns(3)
    m_mild = m1.number_input("Mild", min_value=0.0, value=float(
        st.session_state.rain_mult["mild"]) if st.session_state.rain_mult else 0.60,
        step=0.05, format="%.2f")
    m_typ = m2.number_input("Typical", min_value=0.0, value=1.00, step=0.05, format="%.2f")
    m_sev = m3.number_input("Severe", min_value=0.0, value=float(
        st.session_state.rain_mult["severe"]) if st.session_state.rain_mult else 1.50,
        step=0.05, format="%.2f")
    if st.session_state.rain_mult:
        st.success("Mild and severe were derived from your rainfall file.")

cfg = E.Config(start=start.isoformat(), budget=budget, buf=buf, rate=rate, rho=rho,
               n=int(n_iter), seed=int(seed), m_mild=m_mild, m_typ=m_typ,
               m_sev=m_sev, engine=eng)

# ======================================================================
# HEADER
# ======================================================================
st.title("🌧️ Mumbai Monsoon Risk Calculator")
st.caption("Monte Carlo cost and schedule contingency from three-point estimates. "
           "Replaces a flat contingency percentage with a distribution you can defend.")

tabs = st.tabs(["Inputs", "Results", "Scenarios", "Validation",
                "Rainfall", "Input guide", "Why rain delays", "Model and trust"])

# ======================================================================
# 1. INPUTS
# ======================================================================
with tabs[0]:
    st.subheader("Construction stages")
    st.markdown(
        "Durations and delays are **calendar days**; costs are **₹ lakh**. "
        "`delay_O/M/P` and `cost_O/M/P` are the monsoon-driven *increments* on top of "
        "the baseline duration and cost, not totals. Separate predecessors with `;`.")

    bc1, bc2, bc3 = st.columns([1, 1, 4])
    if bc1.button("Reset to example", width="stretch"):
        st.session_state.acts_df = default_df()
        st.session_state.result = None
        st.rerun()
    up = bc2.file_uploader("Load CSV", type="csv", label_visibility="collapsed")
    if up is not None:
        try:
            d = pd.read_csv(up)
            missing = [c for c in COLS if c not in d.columns]
            if missing:
                st.error("CSV is missing: " + ", ".join(missing))
            else:
                st.session_state.acts_df = d[COLS].copy()
                st.success(f"Loaded {len(d)} activities.")
        except Exception as ex:
            st.error(f"Could not read that CSV: {ex}")

    edited = st.data_editor(
        st.session_state.acts_df, num_rows="dynamic", width="stretch",
        key="editor", height=460,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "name": st.column_config.TextColumn("Activity", width="large"),
            "duration": st.column_config.NumberColumn("Duration (d)", min_value=0.0, format="%.0f"),
            "cost": st.column_config.NumberColumn("Cost (₹L)", min_value=0.0, format="%.1f"),
            "predecessors": st.column_config.TextColumn("Predecessors", width="small"),
            "delay_O": st.column_config.NumberColumn("Delay O", min_value=0.0, format="%.1f"),
            "delay_M": st.column_config.NumberColumn("Delay M", min_value=0.0, format="%.1f"),
            "delay_P": st.column_config.NumberColumn("Delay P", min_value=0.0, format="%.1f"),
            "cost_O": st.column_config.NumberColumn("Cost O", min_value=0.0, format="%.1f"),
            "cost_M": st.column_config.NumberColumn("Cost M", min_value=0.0, format="%.1f"),
            "cost_P": st.column_config.NumberColumn("Cost P", min_value=0.0, format="%.1f"),
        })
    st.session_state.acts_df = edited

    acts, parse_errors = df_to_activities(edited)
    errors, warnings, order = ([], [], [])
    if parse_errors:
        errors = parse_errors
    elif acts:
        errors, warnings, order = E.validate(acts, cfg)

    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)

    if not errors and acts:
        fl, cp, base = E.floats_and_critical_path(acts, order)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline duration", f"{base:,.0f} days")
        k2.metric("Baseline activity cost", f"₹{fmt(cfg.base)} L")
        k3.metric("Budget headroom", f"₹{fmt(cfg.head)} L",
                  delta=None if cfg.head >= 0 else "baseline already overruns",
                  delta_color="inverse")
        k4.metric("Critical path", f"{len(cp)} of {len(acts)} activities")

        with st.expander("Network, float and monsoon exposure", expanded=False):
            ex = pd.DataFrame([{
                "ID": a.id, "Activity": a.name,
                "Predecessors": "; ".join(a.pre) or "—",
                "Duration": a.dur, "Float (d)": fl[a.id],
                "On critical path": "Yes" if a.id in cp else "No",
                "Monsoon overlap": "—" if a.exposure is None else f"{a.exposure*100:.0f}%",
            } for a in acts])
            st.dataframe(ex, width="stretch", hide_index=True)
            st.markdown(
                f'<div class="readnote"><b>Critical path:</b> {" → ".join(cp)}. '
                "Activities with float can absorb some delay without moving the finish date. "
                "Where two branches reach a merge with similar float, neither is reliably "
                "critical and only the simulation gets the mean right — see the Validation tab."
                "</div>", unsafe_allow_html=True)

        # profile guidance (does not feed the simulation)
        hints = []
        if area > 0:
            hints.append(f"Baseline cost works out to about **₹{cfg.base*1e5/area:,.0f} per sq ft**. "
                         "Compare with current local rates; if it looks wrong, recheck costs or area.")
        if bsmt > 0 or fnd == "Piled":
            hints.append("Basement or piling: excavation and dewatering are more exposed, so "
                         "consider higher pessimistic values for excavation delay and pumping cost.")
        if site == "Low-lying or reclaimed":
            hints.append("Low-lying or reclaimed site: waterlogging and a high water table worsen "
                         "the pessimistic case. Raise pessimistic delay and dewatering cost.")
        if site == "Near creek or sea":
            hints.append("Near creek or sea: heavy rain at high tide drains slowly, so raise "
                         "pessimistic delay values.")
        if site == "Tight plot":
            hints.append("Tight plot: little room to store materials or divert water, so material "
                         "damage and access problems are more likely.")
        with st.expander("Guidance from the project profile", expanded=False):
            for h in hints:
                st.markdown("- " + h)
            st.caption("These profile fields do not change the simulation by themselves. They "
                       "document the job, sense-check the budget and guide how you set O/M/P. "
                       "The planned start date is the exception — it does change the result.")

        st.download_button("Download activities.csv",
                           edited.to_csv(index=False).encode(),
                           "activities.csv", "text/csv")

        st.divider()
        if st.button("▶  Run simulation", type="primary", width="stretch"):
            mult = {"Mild monsoon": cfg.m_mild, "Typical monsoon": cfg.m_typ,
                    "Severe monsoon": cfg.m_sev}[st.session_state.scenario_label]
            with st.spinner(f"Running {cfg.n:,} iterations…"):
                st.session_state.result = E.simulate(
                    acts, order, cfg, dm=mult, cm=mult,
                    label=st.session_state.scenario_label)
            st.session_state.ran_once = True
            st.success("Done — open the **Results** tab.")
        st.session_state.scenario_label = st.radio(
            "Scenario to run", ["Mild monsoon", "Typical monsoon", "Severe monsoon"],
            index=["Mild monsoon", "Typical monsoon", "Severe monsoon"].index(
                st.session_state.scenario_label), horizontal=True)

# ======================================================================
# charts
# ======================================================================
def hist_chart(vals, title, unit, pcts):
    fig = go.Figure()
    fig.add_histogram(x=vals, nbinsx=48, marker_color=NAVY, opacity=.80,
                      name="Simulated", hovertemplate="%{x:.1f}<br>%{y} seasons<extra></extra>")
    for p, lbl in ((.5, "P50"), (.8, "P80"), (.9, "P90")):
        fig.add_vline(x=pcts[p], line=dict(color=RED, dash="dash", width=1.4),
                      annotation_text=f"{lbl} {pcts[p]:,.0f}",
                      annotation_position="top", annotation_font_size=11)
    fig.update_layout(title=title, xaxis_title=unit, yaxis_title="Number of seasons",
                      showlegend=False, bargap=.04, **PLOT_LAYOUT)
    return fig


def cdf_chart(sorted_vals, title, unit, pcts):
    q = np.linspace(0, 1, 301)
    xs = [E.percentile(sorted_vals, float(p)) for p in q]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=q * 100, mode="lines", line=dict(color=NAVY, width=2.4),
                    name="Cumulative", hovertemplate="%{x:.1f} → %{y:.0f}%<extra></extra>")
    for p, lbl in ((.5, "P50"), (.8, "P80"), (.9, "P90")):
        v = pcts[p]
        fig.add_shape(type="line", x0=v, x1=v, y0=0, y1=p * 100,
                      line=dict(color=RED, dash="dot", width=1.2))
        fig.add_shape(type="line", x0=xs[0], x1=v, y0=p * 100, y1=p * 100,
                      line=dict(color=RED, dash="dot", width=1.2))
        fig.add_annotation(x=v, y=p * 100, text=f"{lbl} = {v:,.0f}", showarrow=False,
                           xshift=46, yshift=-12, font=dict(size=11, color=RED))
    fig.update_layout(title=title, xaxis_title=unit,
                      yaxis_title="Chance of being at or below (%)",
                      yaxis_range=[0, 100], showlegend=False, **PLOT_LAYOUT)
    return fig


# ======================================================================
# 2. RESULTS
# ======================================================================
with tabs[1]:
    R = st.session_state.result
    if R is None:
        st.info("Set up the activities on the **Inputs** tab, then press "
                "**Run simulation**.")
    else:
        S = E.summarise(R)
        c = R.cfg
        st.caption(
            f"Scenario **{R.label}** (multiplier ×{R.dm:g}) · {R.n:,} iterations · seed {c.seed} "
            f"· ρ = {c.rho} · {c.engine} engine · baseline {R.base:,.0f} days / ₹{fmt(c.base)} L "
            f"· budget ₹{fmt(c.budget)} L · headroom ₹{fmt(c.head)} L")

        st.markdown(f"""<div class="plain"><b>In plain words.</b>
        With the {R.label.lower()} assumptions there is an 80% chance the monsoon adds no more
        than <b>₹{fmt(S['cost_p'][.8])} lakh</b> ({S['cost_p'][.8]/c.budget*100:.1f}% of budget)
        and no more than <b>{fmt(S['delay_p'][.8])} days</b>. On average it adds
        ₹{fmt(S['mean_cost'])} lakh and {fmt(S['mean_delay'])} days. The budget holds
        ₹{fmt(max(0, c.head))} lakh of headroom over the activity costs, so the chance of
        exceeding the budget is <b>{S['p_over_budget']*100:.1f}%</b>, and the chance of finishing
        beyond the {c.buf:.0f}-day buffer is <b>{S['p_late']*100:.1f}%</b>.
        These are estimates under your inputs and assumptions, not forecasts.</div>""",
                    unsafe_allow_html=True)

        st.markdown("#### Additional monsoon cost and delay, above the baseline plan")
        g = st.columns(4)
        g[0].metric("P50 additional cost", f"₹{fmt(S['cost_p'][.5])} L")
        g[1].metric("P80 additional cost", f"₹{fmt(S['cost_p'][.8])} L",
                    f"{S['cost_p'][.8]/c.budget*100:.1f}% of budget", delta_color="off")
        g[2].metric("P90 additional cost", f"₹{fmt(S['cost_p'][.9])} L",
                    f"{S['cost_p'][.9]/c.budget*100:.1f}% of budget", delta_color="off")
        g[3].metric("Mean additional cost", f"₹{fmt(S['mean_cost'])} L",
                    f"± {fmt(S['se_cost'], 2)} standard error", delta_color="off")
        g = st.columns(4)
        g[0].metric("P50 additional delay", f"{fmt(S['delay_p'][.5])} d")
        g[1].metric("P80 additional delay", f"{fmt(S['delay_p'][.8])} d")
        g[2].metric("P90 additional delay", f"{fmt(S['delay_p'][.9])} d")
        g[3].metric("Mean additional delay", f"{fmt(S['mean_delay'])} d",
                    f"± {fmt(S['se_delay'], 2)} standard error", delta_color="off")
        g = st.columns(2)
        g[0].metric("Chance total cost exceeds budget", f"{S['p_over_budget']*100:.1f}%",
                    f"above ₹{fmt(max(0, c.head))} L headroom", delta_color="off")
        g[1].metric("Chance of finishing later than plan", f"{S['p_late']*100:.1f}%",
                    f"beyond the {c.buf:.0f}-day buffer", delta_color="off")

        st.markdown("#### Charts and how to read them")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(hist_chart(R.extra, "Additional cost: histogram",
                                       "Additional cost (₹ lakh)", S["cost_p"]),
                            width="stretch")
            st.markdown(
                f'<div class="readnote"><b>How to read this.</b> Each bar counts how many of the '
                f'{R.n:,} simulated seasons ended with extra cost in that range. Half cost less '
                f'than ₹{fmt(S["cost_p"][.5])} L, 80% less than ₹{fmt(S["cost_p"][.8])} L, 90% '
                f'less than ₹{fmt(S["cost_p"][.9])} L. The long right tail is the bad-luck '
                f'seasons, which is why budgeting for the average (₹{fmt(S["mean_cost"])} L) '
                "leaves you exposed: pessimistic estimates sit far above most-likely ones, and "
                "all stages share one monsoon, so a bad season hurts several at once.</div>",
                unsafe_allow_html=True)
        with b:
            st.plotly_chart(cdf_chart(S["e"], "Additional cost: cumulative",
                                      "Additional cost (₹ lakh)", S["cost_p"]),
                            width="stretch")
            st.markdown(
                '<div class="readnote"><b>How to read this.</b> Pick an amount on the bottom '
                "axis, go up to the curve, read the height: that is the chance the extra cost "
                f'lands at or below it. At ₹{fmt(S["cost_p"][.8])} L the curve reads 80%. Steep '
                "sections hold many outcomes, flat sections few. To set a contingency, choose "
                "the confidence you want and read off the amount.</div>",
                unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.plotly_chart(hist_chart(R.delay, "Additional delay: histogram",
                                       "Additional delay (days)", S["delay_p"]),
                            width="stretch")
            st.markdown(
                '<div class="readnote"><b>How to read this.</b> The same idea for time. A tall '
                "bar near zero would mean the delay is often absorbed by float: a late stage "
                "that is not holding up the finish date does not move completion. Gaps or "
                "bunching come from that schedule logic, not from errors.</div>",
                unsafe_allow_html=True)
        with b:
            st.plotly_chart(cdf_chart(S["d"], "Additional delay: cumulative",
                                      "Additional delay (days)", S["delay_p"]),
                            width="stretch")
            st.markdown(
                f'<div class="readnote"><b>How to read this.</b> The chance of finishing within '
                f'{c.buf:.0f} days of plan is {(1-S["p_late"])*100:.1f}%. To be 80% confident of '
                f'finishing on time, add {fmt(S["delay_p"][.8])} days to the programme.</div>',
                unsafe_allow_html=True)

        st.markdown("#### Contingency at different confidence levels")
        lv = pd.DataFrame([{
            "Confidence": f"P{int(p*100)}",
            "Cost contingency (₹L)": round(S["cost_p"][p], 1),
            "% of budget": f"{S['cost_p'][p]/c.budget*100:.1f}%",
            "Total cost at that level (₹L)": round(c.base + S["cost_p"][p], 1),
            "Time contingency (days)": round(S["delay_p"][p], 1),
        } for p in (.5, .7, .8, .9, .95)])
        st.dataframe(lv, width="stretch", hide_index=True)
        st.caption("Total cost = baseline activity cost + the additional-cost percentile. "
                   "Each row is a percentile of the additional amount, not of the total.")

        st.markdown("#### Test your own contingency")
        t1, t2 = st.columns(2)
        tx = t1.number_input("Cost contingency (₹ lakh)",
                             value=float(round(S["cost_p"][.8], 1)), step=1.0)
        ty = t2.number_input("Time contingency (days)",
                             value=float(round(S["delay_p"][.8])), step=1.0)
        pc_ = float(np.mean(R.extra > tx)) * 100
        pd_ = float(np.mean(R.delay > ty)) * 100
        t1.markdown(f"Chance extra cost exceeds ₹{tx:,.0f} L: **{pc_:.1f}%**")
        t2.markdown(f"Chance delay exceeds {ty:,.0f} days: **{pd_:.1f}%**")

        st.markdown("#### Which activities drive the risk?")
        tot_cost = sum(float(R.act_cost[i].mean()) for i in range(len(R.acts)))
        drv = pd.DataFrame([{
            "Activity": f"{a.id} — {a.name}",
            "Monsoon overlap": "—" if a.exposure is None else f"{a.exposure*100:.0f}%",
            "Mean extra cost (₹L)": round(float(R.act_cost[i].mean()), 2),
            "Share of extra cost": f"{float(R.act_cost[i].mean())/max(tot_cost,1e-9)*100:.1f}%",
            "Mean delay (d)": round(float(R.act_delay[i].mean()), 2),
            "Corr. with project delay": round(float(np.corrcoef(
                R.act_delay[i], R.delay)[0, 1]), 2),
        } for i, a in enumerate(R.acts)]).sort_values(
            "Mean extra cost (₹L)", ascending=False)
        st.dataframe(drv, width="stretch", hide_index=True)

        share = [float(R.act_cost[i].mean()) for i in range(len(R.acts))]
        o = np.argsort(share)
        fig = go.Figure(go.Bar(x=[share[i] for i in o],
                               y=[R.acts[i].id for i in o], orientation="h",
                               marker_color=ORANGE,
                               hovertemplate="%{y}: ₹%{x:.2f} L<extra></extra>"))
        fig.update_layout(title="Mean extra cost by activity (₹ lakh)",
                          xaxis_title="₹ lakh", **PLOT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            '<div class="readnote">Correlation near zero suggests the activity sits on a path '
            "with float, so its delay rarely moves the finish date. It is an indicator, not a "
            "full criticality analysis. Concentrate estimating effort on the top rows: those "
            "are where a sharper O/M/P changes the answer.</div>", unsafe_allow_html=True)

        buf_out = io.StringIO()
        pd.DataFrame({"additional_cost_lakh": R.extra,
                      "additional_delay_days": R.delay}).to_csv(buf_out, index=False)
        st.download_button("Download the simulated draws (CSV)",
                           buf_out.getvalue().encode(), "simulation_draws.csv", "text/csv")

# ======================================================================
# 3. SCENARIOS
# ======================================================================
with tabs[2]:
    st.subheader("Mild, typical and severe side by side")
    st.caption("Same activities, same seed, three severity multipliers. Multipliers are "
               "assumptions unless you derive them on the Rainfall tab.")
    acts2, bad2 = df_to_activities(st.session_state.acts_df)
    if bad2:
        st.error("Fix the input errors on the Inputs tab first.")
    else:
        err2, _, ord2 = E.validate(acts2, cfg)
        if err2:
            st.error("Fix the input errors on the Inputs tab first.")
        elif st.button("Run all three scenarios", type="primary"):
            rows, store = [], {}
            prog = st.progress(0.0)
            for j, (nm, m) in enumerate([("Mild", cfg.m_mild), ("Typical", cfg.m_typ),
                                         ("Severe", cfg.m_sev)]):
                r = E.simulate(acts2, ord2, cfg, dm=m, cm=m, label=nm)
                s = E.summarise(r)
                store[nm] = s
                rows.append({
                    "Scenario": f"{nm} (×{m:g})",
                    "Mean cost": round(s["mean_cost"], 1),
                    "P50 cost": round(s["cost_p"][.5], 1),
                    "P80 cost": round(s["cost_p"][.8], 1),
                    "P90 cost": round(s["cost_p"][.9], 1),
                    "Mean delay": round(s["mean_delay"], 1),
                    "P50 delay": round(s["delay_p"][.5], 1),
                    "P80 delay": round(s["delay_p"][.8], 1),
                    "P90 delay": round(s["delay_p"][.9], 1),
                    "P(over budget)": f"{s['p_over_budget']*100:.1f}%",
                    "P(late)": f"{s['p_late']*100:.1f}%",
                })
                prog.progress((j + 1) / 3)
            prog.empty()
            st.session_state.scen = (rows, {k: {"cost_p": v["cost_p"]} for k, v in store.items()})
        if st.session_state.scen:
            rows, store = st.session_state.scen
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Additional cost in ₹ lakh, additional delay in days.")

            fig = go.Figure()
            for nm, col in (("Mild", "#7FA6D9"), ("Typical", NAVY), ("Severe", ORANGE)):
                fig.add_scatter(
                    x=[store[nm]["cost_p"][p] for p in (.5, .7, .8, .9, .95)],
                    y=[50, 70, 80, 90, 95], mode="lines+markers", name=nm,
                    line=dict(color=col, width=2.2))
            fig.update_layout(title="Cost contingency by confidence level",
                              xaxis_title="Additional cost (₹ lakh)",
                              yaxis_title="Confidence (%)", **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                '<div class="readnote">Read across at the confidence you need. The gap between '
                "the mild and severe lines is the price of not knowing what kind of monsoon you "
                "will get — often wider than the gap between percentiles within one scenario, "
                "which is why deriving the multipliers from real rainfall matters.</div>",
                unsafe_allow_html=True)

# ======================================================================
# 4. VALIDATION
# ======================================================================
with tabs[3]:
    st.subheader("Does the tool compute what the mathematics says it should?")
    R = st.session_state.result
    if R is None:
        st.info("Run a simulation first, then come back.")
    else:
        m = R.dm
        st.markdown(f"##### Simulated mean against the hand PERT mean (scenario ×{m:g})")
        vr = []
        for i, a in enumerate(R.acts):
            for lbl, O, M, P, arr in (("delay (d)", a.dO, a.dM, a.dP, R.act_delay[i]),
                                      ("extra cost (₹L)", a.cO, a.cM, a.cP, R.act_cost[i])):
                exp = E.pert_mean(O, M, P) * m
                sm = float(arr.mean())
                se = float(arr.std(ddof=1) / math.sqrt(R.n))
                z = (sm - exp) / se if se > 0 else 0.0
                vr.append({"Variable": f"{a.id} {lbl}",
                           "O / M / P": f"{O:g} / {M:g} / {P:g}",
                           "PERT mean": round(exp, 3), "Simulated": round(sm, 3),
                           "Difference": round(sm - exp, 4), "Std error": round(se, 4),
                           "Diff / SE": round(z, 2),
                           "Result": "✓ within 3 SE" if abs(z) <= 3 else "✗ check"})
        vdf = pd.DataFrame(vr)
        npass = int((vdf["Result"].str.startswith("✓")).sum())
        st.dataframe(vdf, width="stretch", hide_index=True, height=360)
        (st.success if npass == len(vdf) else st.error)(
            f"{npass} of {len(vdf)} variables agree with the hand PERT mean within "
            f"3 standard errors. Largest |Diff / SE| = {vdf['Diff / SE'].abs().max():.2f}.")
        st.markdown(
            '<div class="readnote">Agreement here shows the <b>sampler</b> is correct. It does '
            "not show the O/M/P inputs are realistic, and it does not license a hand "
            "critical-path sum as a project estimate — see the next section.</div>",
            unsafe_allow_html=True)

        st.markdown("##### Project level: where a hand calculation legitimately differs")
        fl, cp, base = E.floats_and_critical_path(R.acts, R.order)
        H = E.hand_pert_project(R.acts, cp, R.cfg.rho, R.cfg.rate)
        S = E.summarise(R)
        hm = {k: v * m for k, v in H.items()}
        se_d = S["se_delay"]
        prj = pd.DataFrame([
            {"Quantity": "Mean additional delay (days)", "Manual": round(hm["mean_delay"], 3),
             "Tool": round(S["mean_delay"], 3),
             "Difference": round(S["mean_delay"] - hm["mean_delay"], 3),
             "Diff / SE": round((S["mean_delay"] - hm["mean_delay"]) / se_d, 2),
             "Manual method": f"Sum of PERT means along {'-'.join(cp)}"},
            {"Quantity": "SD of delay, classical (P−O)/6 RSS", "Manual": round(hm["sd_class"], 3),
             "Tool": round(S["sd_delay"], 3),
             "Difference": round(S["sd_delay"] - hm["sd_class"], 3), "Diff / SE": None,
             "Manual method": "Root-sum-of-squares, independence assumed"},
            {"Quantity": "SD of delay, exact beta RSS", "Manual": round(hm["sd_beta"], 3),
             "Tool": round(S["sd_delay"], 3),
             "Difference": round(S["sd_delay"] - hm["sd_beta"], 3), "Diff / SE": None,
             "Manual method": "Exact beta SDs, still independent"},
            {"Quantity": "SD of delay, beta with ρ² cross terms",
             "Manual": round(hm["sd_corr"], 3), "Tool": round(S["sd_delay"], 3),
             "Difference": round(S["sd_delay"] - hm["sd_corr"], 3), "Diff / SE": None,
             "Manual method": "Corrected formula — the one that works"},
            {"Quantity": "Mean additional cost (₹ lakh)", "Manual": round(hm["mean_cost"], 3),
             "Tool": round(S["mean_cost"], 3),
             "Difference": round(S["mean_cost"] - hm["mean_cost"], 3),
             "Diff / SE": round((S["mean_cost"] - hm["mean_cost"]) / S["se_cost"], 2),
             "Manual method": "Cost PERT means + mean delay × rate"},
        ])
        st.dataframe(prj, width="stretch", hide_index=True)

        gap_sd = abs(S["sd_delay"] - hm["sd_corr"]) / max(S["sd_delay"], 1e-9) * 100
        st.markdown(
            f'<div class="readnote"><b>On the standard deviation.</b> The textbook '
            f'(P−O)/6 root-sum-of-squares gives {hm["sd_class"]:.2f} days against '
            f'{S["sd_delay"]:.2f} simulated. Two separate manual errors cause that: '
            "(P−O)/6 understates the true beta SD, and root-sum-of-squares assumes "
            f"independence when ρ = {R.cfg.rho}. Correcting both gives "
            f'{hm["sd_corr"]:.2f} — within {gap_sd:.2f}% of the simulation. The tool was '
            "right; the hand formula was wrong.</div>", unsafe_allow_html=True)

        MD = E.merge_diagnostics(R)
        if MD and MD["spread"] < 5:
            diff = S["mean_delay"] - hm["mean_delay"]
            st.markdown("##### Merge-event bias")
            mc = st.columns(4)
            mc[0].metric(f"Merge at {MD['merge_id']}",
                         " vs ".join(MD["pred_ids"]))
            mc[1].metric("E[max] − max[E]", f"{MD['jensen']:+.3f} d")
            mc[2].metric("Tool − manual", f"{diff:+.3f} d",
                         f"{diff/se_d:+.1f} SE", delta_color="off")
            mc[3].metric("Governs the merge",
                         " / ".join(f"{g*100:.0f}%" for g in MD["governs"]))
            st.markdown(
                f'<div class="readnote"><b>Why the project mean does not reconcile.</b> '
                f'Two branches reach {MD["merge_id"]} with mean finishes '
                + " and ".join(f"{v:,.2f}" for v in MD["pred_mean_finish"])
                + f' days — only {MD["spread"]:.3f} days apart. Because '
                "E[max(X,Y)] ≥ max(E[X],E[Y]), the expected completion is later than any single "
                f'path\'s expectation, here by {MD["jensen"]:.3f} days. A hand critical-path sum '
                "commits to one path before the draws are known and cannot see this, so the "
                "simulated mean is the correct one. Where the branches are far apart this "
                "vanishes and the hand sum is fine.</div>", unsafe_allow_html=True)

        st.markdown("##### Self-tests")
        st.caption("Deterministic cases with known answers, plus checks on the sampler, the "
                   "correlation structure and your own model.")
        if st.button("Run self-tests"):
            acts3, _ = df_to_activities(st.session_state.acts_df)
            _, _, ord3 = E.validate(acts3, cfg)
            with st.spinner("Testing…"):
                st.session_state.tests = E.self_tests(acts3, ord3, cfg)
        if st.session_state.tests:
            res = st.session_state.tests
            for t in res:
                (st.success if t["ok"] else st.error)(
                    ("✓ " if t["ok"] else "✗ ") + t["name"])
                if t["detail"]:
                    st.caption("   " + t["detail"])
            nfail = sum(1 for t in res if not t["ok"])
            if nfail:
                st.error(f"{nfail} test(s) failed — do not trust the output until fixed.")
            else:
                st.success(f"All {len(res)} tests passed.")

        st.markdown("##### Convergence and seed stability")
        if st.button("Check stability"):
            acts4, _ = df_to_activities(st.session_state.acts_df)
            _, _, ord4 = E.validate(acts4, cfg)
            rows = []
            for n in (1000, 5000, 10000, 50000):
                s = E.summarise(E.simulate(acts4, ord4, cfg, n=n, seed=7, dm=m, cm=m))
                rows.append({"Test": f"n = {n:,}", "P80 cost": round(s["cost_p"][.8], 1),
                             "P80 delay": round(s["delay_p"][.8], 1),
                             "Mean delay": round(s["mean_delay"], 2)})
            for sd in (1, 2, 3, 4, 5):
                s = E.summarise(E.simulate(acts4, ord4, cfg, seed=sd, dm=m, cm=m))
                rows.append({"Test": f"seed {sd}", "P80 cost": round(s["cost_p"][.8], 1),
                             "P80 delay": round(s["delay_p"][.8], 1),
                             "Mean delay": round(s["mean_delay"], 2)})
            st.session_state.stab = rows
        if st.session_state.stab:
            sdf = pd.DataFrame(st.session_state.stab)
            st.dataframe(sdf, width="stretch", hide_index=True)
            seeds = sdf[sdf["Test"].str.startswith("seed")]
            st.caption(
                f"Across five seeds, P80 cost spans {seeds['P80 cost'].min():.1f} to "
                f"{seeds['P80 cost'].max():.1f} ₹L and P80 delay "
                f"{seeds['P80 delay'].min():.1f} to {seeds['P80 delay'].max():.1f} days. "
                "Choose iterations so that spread is small next to the decision you are making.")

# ======================================================================
# 5. RAINFALL
# ======================================================================
with tabs[4]:
    st.subheader("Derive the scenario multipliers from real rainfall")
    st.markdown(
        "Upload **daily** June–September rainfall for at least five years: a CSV whose first "
        "column is the date and second is rainfall in mm. IMD station data for Santacruz or "
        "Colaba is the right source for Mumbai.")
    thr = st.number_input("Heavy-rain threshold (mm/day)", value=64.5, step=5.0,
                          help="64.5 mm is the IMD 'heavy rainfall' threshold.")
    rf = st.file_uploader("Daily rainfall CSV", type=["csv", "txt"], key="rfup")
    if rf is not None:
        try:
            raw = pd.read_csv(rf)
            g, info = E.rainfall_multipliers(raw, thr)
            if g is not None:
                show = g.copy()
                show["total_mm"] = show["total_mm"].round(0)
                st.dataframe(show.rename(columns={
                    "year": "Year", "total_mm": "Jun–Sep rain (mm)",
                    "heavy_days": f"Days ≥ {thr:g} mm",
                    "days_with_data": "Days with data (of 122)"}),
                    width="stretch", hide_index=True)
                thin = g[g["days_with_data"] < 120]
                if len(thin):
                    st.warning("Years with missing days: "
                               + ", ".join(str(y) for y in thin["year"])
                               + ". Their totals understate the season.")
                fig = go.Figure(go.Bar(x=g["year"].astype(str), y=g["total_mm"],
                                       marker_color=NAVY))
                if isinstance(info, dict):
                    fig.add_hline(y=info["median_mm"], line=dict(color=RED, dash="dash"),
                                  annotation_text="median year")
                fig.update_layout(title="Seasonal rainfall by year",
                                  xaxis_title="Year", yaxis_title="Jun–Sep total (mm)",
                                  **PLOT_LAYOUT)
                st.plotly_chart(fig, width="stretch")
            if isinstance(info, str):
                st.error(info)
            elif isinstance(info, dict):
                st.success(
                    f"Median seasonal total **{info['median_mm']:,.0f} mm** over "
                    f"{info['years']} years. Mild (20th-percentile year) = "
                    f"**×{info['mild']:.2f}**, severe (90th-percentile year) = "
                    f"**×{info['severe']:.2f}**.")
                if st.button("Use these as scenario multipliers", type="primary"):
                    st.session_state.rain_mult = {"mild": info["mild"],
                                                  "severe": info["severe"]}
                    st.rerun()
                st.markdown(
                    '<div class="readnote"><b>Assumption, not a measurement.</b> This takes '
                    "delay and extra cost to scale in proportion to the seasonal total relative "
                    "to the median year. That is a modelling choice, not an observed "
                    "relationship — a season with the same total but concentrated in fewer, "
                    "heavier events would disrupt a site more. With only a handful of years the "
                    "percentiles are rough.</div>", unsafe_allow_html=True)
        except Exception as ex:
            st.error(f"Could not read that file: {ex}")
    else:
        st.info("No file loaded, so the mild and severe multipliers in the sidebar remain "
                "assumptions. State them as such in any report.")

# ======================================================================
# 6-8. GUIDANCE
# ======================================================================
with tabs[5]:
    st.subheader("How to set the three-point estimates")
    st.markdown(textwrap.dedent("""
    **What O, M and P mean here.** They are the monsoon-driven *extra* days and *extra*
    rupees, on top of a baseline that already assumes fair weather.

    - **O, optimistic** — a season that barely interferes. Near a best case, roughly a
      1-in-10 outcome. Zero is legitimate for an activity that runs outside June–September.
    - **M, most likely** — the single value you would bet on. The modal outcome, not the
      average.
    - **P, pessimistic** — adverse but credible, roughly a 1-in-10 bad outcome. Not the
      worst imaginable: a cyclone landfall belongs in a separate risk register, not here.

    **A defensible way to build the delay figures**

    1. Work out which calendar months each activity spans, from the planned start date and
       the baseline durations.
    2. Take the mean rain days for those months from IMD station normals. For Santacruz the
       Jun–Sep figures are roughly 14, 22, 20 and 13 days.
    3. Multiply by a workable-day loss factor reflecting how rain-sensitive the activity is.
       Something like 0.10 / 0.33 / 0.90 for O / M / P on highly sensitive work such as
       concrete pours and excavation; 0.05 / 0.20 / 0.55 for moderately sensitive work such
       as masonry and external plaster; 0.00 / 0.05 / 0.15 for internal work.
    4. Sanity-check the total. Published Indian studies put monsoon duration overrun around
       5% for Kolkata; Mumbai's seasonal rainfall is roughly double, so somewhere in the
       10–15% range is reasonable for a project whose structure phase crosses one monsoon.

    **For the cost figures.** Keep them to materials, wastage and rework — anything that
    accrues *per idle day* belongs in the sidebar's daily-cost panel instead. Double counting
    idle labour in both places is the commonest error with this kind of model.

    **Where to be generous with P.** Excavation and dewatering, concrete pours, external
    plaster and waterproofing, and anything involving a tower crane. Where to be tight:
    internal finishes under a completed slab.
    """))

with tabs[6]:
    st.subheader("Why rain turns into delay and cost")
    st.markdown(textwrap.dedent("""
    Rain does not stop a site uniformly, which is why a flat percentage fits so badly.

    **Direct stoppage.** Concrete cannot be placed or finished in active rain without risking
    the water–cement ratio and the surface. Excavation in a flooded pit is unsafe and
    unproductive. Crane lifts stop in high wind and poor visibility.

    **Recovery time after the rain stops.** This is the part flat percentages miss. A pit has
    to be pumped out and the softened bottom re-levelled. Compacted backfill that got
    saturated fails density tests and has to be reworked. Access roads churn up. A two-hour
    shower can cost most of a day.

    **Cost that accrues whether or not work happens.** Labour on site still gets paid, or
    worse, migrant crews leave and have to be re-recruited at a premium. Plant hired by the
    month is paid for rain-stopped days. Site staff, security and overheads run on. Interest
    on drawn finance accrues daily.

    **Damage and rework.** Cement sets hard in damp storage. Rebar surface-corrodes. Plywood
    shuttering delaminates. External plaster applied before a wet spell fails and comes off.

    **Why the tail is long.** Two reinforcing effects. Within an activity, pessimistic
    estimates sit far above most-likely ones because bad outcomes compound — a washed-out
    pour needs breaking out, not just redoing. Across activities, every stage shares a single
    monsoon, so a bad season hits several at once. That is what the correlation ρ represents,
    and it is why the distribution is skewed rather than symmetric.

    **Why float matters.** A stage that is delayed but not holding up the finish date does
    not move completion. That is why the delay distribution can bunch near zero, and why
    activities with large float show weak correlation with project delay in the drivers table.
    """))

with tabs[7]:
    st.subheader("What this model does, and what it cannot do")
    st.markdown(textwrap.dedent("""
    **The method.** Each activity's extra delay and extra cost are drawn from a PERT-beta
    distribution on [O, P] with shape parameters α = 1 + 4(M−O)/(P−O) and
    β = 1 + 4(P−M)/(P−O). Because α + β = 6 by construction, the distribution mean is exactly
    (O + 4M + P)/6 — which is what makes the hand check on the Validation tab possible.

    A single shared factor z drives every activity, so the latent correlation between any two
    is ρ², not ρ. Within an activity, cost and delay are coupled at 0.85. Project delay comes
    from a full forward critical-path pass with the sampled delays added, so it is a maximum
    over paths rather than a sum along one. Additional cost is the sum of sampled activity
    costs plus the project delay times the daily rate.

    **What validation establishes.** That the sampler reproduces the intended distributions,
    that the correlation structure is what was specified, and that the schedule logic handles
    float and merges correctly. Nothing more.

    **What it does not establish**

    - *The inputs are not validated.* Whether 5 / 15 / 41 days is a fair description of
      monsoon delay on a superstructure pour is answered by rainfall records and site history,
      not by this tool.
    - *ρ is assumed, not measured.* Moving it from 0 to 0.9 changes the project standard
      deviation by roughly a factor of two — more than any single estimate does. Always state
      the value you used.
    - *The distribution family is assumed.* PERT-beta was chosen for being bounded and
      mode-weighted, not fitted to observed delay data.
    - *The daily rate is treated as constant and certain.* In reality it probably rises with
      delay length as penalties and financing compound.
    - *Slip into the monsoon is not modelled.* Each activity's spread is fixed, so if an early
      stage overruns badly and pushes a later one into June, the model does not widen that
      later activity's distribution in response. This is the most important structural
      limitation.
    - *Nothing here is a forecast.* These are the consequences of your assumptions, expressed
      as probabilities.

    **Reporting honestly.** Quote P50, P80 and P90 from the simulation rather than a
    mean-plus-z×σ rule, since the distribution is skewed. Give the mean with its standard
    error. State ρ, the iteration count and the seed. And say which multipliers were derived
    from rainfall data and which were assumed.
    """))
