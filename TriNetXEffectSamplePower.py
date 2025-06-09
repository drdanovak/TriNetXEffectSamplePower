import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import io

st.set_page_config(layout="wide")
st.title("TriNetX Power & Effect Size Calculator (Auto-Populated)")

def robust_csv_to_df(uploaded_file):
    raw = uploaded_file.read().decode('utf-8').splitlines()
    rows = []
    max_cols = 0
    for line in raw:
        comma_split = line.split(',')
        tab_split = line.split('\t')
        row = comma_split if len(comma_split) >= len(tab_split) else tab_split
        rows.append(row)
        if len(row) > max_cols:
            max_cols = len(row)
    rows = [r + [''] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(rows)
    return df

def extract_trinetx_stats(df):
    # Map exact cell locations (0-indexed)
    # Note: df.iloc[row, col] = (row, col)
    try:
        group1_n = int(float(df.iat[10,2]))
        group2_n = int(float(df.iat[11,2]))
        group1_events = int(float(df.iat[10,3]))
        group2_events = int(float(df.iat[11,3]))
        group1_risk = float(df.iat[10,4])
        group2_risk = float(df.iat[11,4])
        risk_diff = float(df.iat[16,0])
        risk_diff_lci = float(df.iat[16,1])
        risk_diff_uci = float(df.iat[16,2])
        risk_diff_p = float(df.iat[16,4]) if df.iat[16,4] != "" else None

        risk_ratio = float(df.iat[21,0])
        risk_ratio_lci = float(df.iat[21,1])
        risk_ratio_uci = float(df.iat[21,2])
        risk_ratio_p = float(df.iat[21,4]) if df.iat[21,4] != "" else None

        odds_ratio = float(df.iat[26,0])
        odds_ratio_lci = float(df.iat[26,1])
        odds_ratio_uci = float(df.iat[26,2])
        odds_ratio_p = float(df.iat[26,4]) if df.iat[26,4] != "" else None

        return {
            "group1_n": group1_n,
            "group2_n": group2_n,
            "group1_events": group1_events,
            "group2_events": group2_events,
            "group1_risk": group1_risk,
            "group2_risk": group2_risk,
            "risk_diff": risk_diff,
            "risk_diff_lci": risk_diff_lci,
            "risk_diff_uci": risk_diff_uci,
            "risk_diff_p": risk_diff_p,
            "risk_ratio": risk_ratio,
            "risk_ratio_lci": risk_ratio_lci,
            "risk_ratio_uci": risk_ratio_uci,
            "risk_ratio_p": risk_ratio_p,
            "odds_ratio": odds_ratio,
            "odds_ratio_lci": odds_ratio_lci,
            "odds_ratio_uci": odds_ratio_uci,
            "odds_ratio_p": odds_ratio_p
        }
    except Exception as e:
        st.error(f"Failed to extract stats from CSV: {e}")
        return None

# ========== Upload + Extract ==========
uploaded_file = st.sidebar.file_uploader("📂 Upload TriNetX Outcome CSV", type=["csv"])
trinetx_stats = None
df_csv = None
if uploaded_file:
    df_csv = robust_csv_to_df(uploaded_file)
    trinetx_stats = extract_trinetx_stats(df_csv)

# ========== Power Calculator ==========
def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1-p_bar) * (1/n1 + 1/n2))
    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z = diff / pooled_se
    power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    return power

def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1):
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    num = (z_alpha * np.sqrt(2 * p_bar * q_bar) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)/ratio)) ** 2
    denom = (p1 - p2) ** 2
    n1 = num / denom
    n2 = n1 * ratio
    return int(np.ceil(n1)), int(np.ceil(n2))

st.header("Power & Sample Size Calculator")
if trinetx_stats:
    st.info("Power calculator is pre-filled with values from your uploaded TriNetX outcome file.")
else:
    st.warning("No TriNetX CSV uploaded. Enter values manually.")

with st.form("power_form"):
    c1, c2 = st.columns(2)
    with c1:
        n1 = st.number_input("Group 1 sample size (n1)", min_value=1, value=int(trinetx_stats["group1_n"]) if trinetx_stats else 100)
        p1 = st.number_input("Group 1 risk (proportion)", min_value=0.0, max_value=1.0, value=float(trinetx_stats["group1_risk"]) if trinetx_stats else 0.10, format="%.5f")
    with c2:
        n2 = st.number_input("Group 2 sample size (n2)", min_value=1, value=int(trinetx_stats["group2_n"]) if trinetx_stats else 100)
        p2 = st.number_input("Group 2 risk (proportion)", min_value=0.0, max_value=1.0, value=float(trinetx_stats["group2_risk"]) if trinetx_stats else 0.20, format="%.5f")

    alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05)
    power_goal = st.number_input("Target power (for sample size calc)", min_value=0.01, max_value=0.99, value=0.8)
    two_sided = st.checkbox("Two-sided test", value=True)
    mode = st.radio("What do you want to calculate?", ["Calculate Power (given N)", "Calculate Sample Size (given power)"])

    submitted = st.form_submit_button("Calculate")

if submitted:
    if mode == "Calculate Power (given N)":
        if n1 and n2 and 0 <= p1 < 1 and 0 <= p2 < 1:
            power = calc_power(n1, n2, p1, p2, alpha, two_sided)
            st.success(f"Power: **{power:.3f}** for n1={n1}, n2={n2}, p1={p1:.3f}, p2={p2:.3f}")
    else:
        if 0 <= p1 < 1 and 0 <= p2 < 1 and power_goal > 0:
            n1_needed, n2_needed = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio=n2/n1 if n1 else 1)
            st.success(f"Sample Size Needed: **Group 1: {n1_needed}**, **Group 2: {n2_needed}** for power={power_goal}, alpha={alpha}")

st.caption("Assumes a z-test for two independent proportions.")

# ========== Effect Size Table/Plot Auto-Population ==========

st.header("Effect Size Calculator & Forest Plot (auto-filled from CSV)")
ratio_choices = ["Risk Ratio", "Odds Ratio"]
default_ratio_type = "Risk Ratio"
if trinetx_stats:
    # If odds ratio is present, let them pick which to show
    if trinetx_stats["odds_ratio"] and trinetx_stats["risk_ratio"]:
        default_ratio_type = st.radio("Which effect size to show?", ["Risk Ratio", "Odds Ratio"], index=0)
    elif trinetx_stats["odds_ratio"]:
        default_ratio_type = "Odds Ratio"
    else:
        default_ratio_type = "Risk Ratio"
else:
    default_ratio_type = "Risk Ratio"

if trinetx_stats:
    if default_ratio_type == "Risk Ratio":
        ratio_val = trinetx_stats["risk_ratio"]
        lci = trinetx_stats["risk_ratio_lci"]
        uci = trinetx_stats["risk_ratio_uci"]
        pval = trinetx_stats["risk_ratio_p"]
    else:
        ratio_val = trinetx_stats["odds_ratio"]
        lci = trinetx_stats["odds_ratio_lci"]
        uci = trinetx_stats["odds_ratio_uci"]
        pval = trinetx_stats["odds_ratio_p"]
else:
    ratio_val, lci, uci, pval = 1.0, None, None, None

# Default outcome name from CSV or generic
outcome_name = "Outcome"
if uploaded_file:
    import os
    outcome_name = os.path.splitext(uploaded_file.name)[0]

st.markdown("### Effect Sizes Table (editable)")

table_data = [{
    "Outcome": outcome_name,
    default_ratio_type: ratio_val,
    "Lower CI (Ratio)": lci,
    "Upper CI (Ratio)": uci,
    "p-value": pval,
}]

effect_df = pd.DataFrame(table_data)

edited_df = st.data_editor(
    effect_df,
    num_rows="dynamic",
    use_container_width=True,
    key="effect_table",
    column_config={
        "Outcome": st.column_config.TextColumn("Outcome"),
        default_ratio_type: st.column_config.NumberColumn(default_ratio_type),
        "Lower CI (Ratio)": st.column_config.NumberColumn("Lower CI (Ratio)", required=False),
        "Upper CI (Ratio)": st.column_config.NumberColumn("Upper CI (Ratio)", required=False),
        "p-value": st.column_config.NumberColumn("p-value", required=False)
    }
)

# Compute effect size (Chinn's method)
edited_df[default_ratio_type] = pd.to_numeric(edited_df[default_ratio_type], errors='coerce')
edited_df['Effect Size'] = np.log(np.abs(edited_df[default_ratio_type])) * (np.sqrt(3) / np.pi) * np.sign(edited_df[default_ratio_type])
if "Lower CI (Ratio)" in edited_df and "Upper CI (Ratio)" in edited_df:
    edited_df['Lower CI (Effect Size)'] = np.log(np.abs(edited_df['Lower CI (Ratio)'])) * (np.sqrt(3) / np.pi) * np.sign(edited_df['Lower CI (Ratio)'])
    edited_df['Upper CI (Effect Size)'] = np.log(np.abs(edited_df['Upper CI (Ratio)'])) * (np.sqrt(3) / np.pi) * np.sign(edited_df['Upper CI (Ratio)'])

# Render table
def ama_table_html(df, ratio_label="Risk Ratio"):
    if df.empty:
        return ""
    html = f"""
    <style>
    .ama-table {{ border-collapse:collapse; font-family:Arial,sans-serif; font-size:14px; }}
    .ama-table th, .ama-table td {{ border:1px solid #222; padding:6px 12px; }}
    .ama-table th {{ background:#f8f8f8; font-weight:bold; text-align:center; }}
    .ama-table td {{ text-align:right; }}
    .ama-table td.left {{ text-align:left; }}
    </style>
    <table class="ama-table">
        <tr>
            <th>Outcome</th>
            <th>{ratio_label}</th>
            <th>Lower CI (Ratio)</th>
            <th>Upper CI (Ratio)</th>
            <th>Effect Size</th>
            <th>Lower CI (Effect Size)</th>
            <th>Upper CI (Effect Size)</th>
            <th>p-value</th>
        </tr>
    """
    for _, row in df.iterrows():
        html += (
            f"<tr><td class='left'>{row['Outcome']}</td>"
            f"<td>{row[ratio_label]}</td>"
            f"<td>{row.get('Lower CI (Ratio)','')}</td>"
            f"<td>{row.get('Upper CI (Ratio)','')}</td>"
            f"<td>{row['Effect Size']}</td>"
            f"<td>{row.get('Lower CI (Effect Size)','')}</td>"
            f"<td>{row.get('Upper CI (Effect Size)','')}</td>"
            f"<td>{row.get('p-value','')}</td></tr>"
        )
    html += "</table>"
    return html

st.markdown(ama_table_html(edited_df.round(6), ratio_label=default_ratio_type), unsafe_allow_html=True)

# Forest Plot (optional)
with st.expander("Show Forest Plot"):
    if st.button("Generate Forest Plot"):
        fig, ax = plt.subplots(figsize=(6, 2))
        row = edited_df.iloc[0]
        effect = row["Effect Size"]
        lci = row.get("Lower CI (Effect Size)", None)
        uci = row.get("Upper CI (Effect Size)", None)
        ax.errorbar(effect, 0, xerr=[[effect-lci], [uci-effect]], fmt='o', capsize=6, color='tab:blue')
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_yticks([0])
        ax.set_yticklabels([row["Outcome"]])
        ax.set_xlabel("Effect Size")
        ax.set_title("Forest Plot")
        plt.tight_layout()
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("Download Forest Plot (PNG)", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")

st.markdown(
    "> **Effect sizes are computed using Chinn S. (2000). [A simple method for converting an odds ratio to effect size for use in meta-analysis](https://doi.org/10.1002/1097-0258(20001130)19:22<3127::aid-sim784>3.0.co;2-m). Stat Med, 19(22), 3127-3131.**"
)
