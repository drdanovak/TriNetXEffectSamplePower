import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd

st.set_page_config(layout="wide")
st.title("Power & Sample Size Adequacy Checker (TriNetX Compatible)")

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
    df = np.array(rows)
    return df

def extract_trinetx_stats(df):
    try:
        group1_n = int(float(df[10,2]))
        group2_n = int(float(df[11,2]))
        group1_risk = float(df[10,4])
        group2_risk = float(df[11,4])
        return group1_n, group2_n, group1_risk, group2_risk
    except Exception as e:
        st.warning(f"Could not extract all required values from the CSV: {e}")
        return 100, 100, 0.10, 0.20

def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    # Avoid division by zero for very small groups
    if n1 <= 0 or n2 <= 0 or p1 == p2:
        return 0.0
    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1 - p_bar) * (1/n1 + 1/n2))
    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z = diff / pooled_se
    if two_sided:
        power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    else:
        power = 1 - norm.cdf(z_alpha - z)
    return float(np.clip(power, 0, 1))

def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1):
    if abs(p1 - p2) < 1e-9:
        return None, None
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    num = (z_alpha * np.sqrt(2 * p_bar * q_bar) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)/ratio)) ** 2
    denom = (p1 - p2) ** 2
    n1 = num / denom
    n2 = n1 * ratio
    return int(np.ceil(n1)), int(np.ceil(n2))

uploaded_file = st.file_uploader("📂 Upload a TriNetX Outcome CSV (auto-fills fields below)", type=["csv"])
default_n1, default_n2, default_p1, default_p2 = 100, 100, 0.10, 0.20

if uploaded_file:
    df = robust_csv_to_df(uploaded_file)
    n1, n2, p1, p2 = extract_trinetx_stats(df)
    st.success("Sample sizes and risks auto-filled from uploaded TriNetX CSV.")
else:
    n1, n2, p1, p2 = default_n1, default_n2, default_p1, default_p2

st.markdown("#### Study Parameters")
c1, c2, c3, c4 = st.columns(4)
with c1:
    n1 = st.number_input("Group 1 sample size (n1)", min_value=1, value=n1)
with c2:
    n2 = st.number_input("Group 2 sample size (n2)", min_value=1, value=n2)
with c3:
    p1 = st.number_input("Group 1 risk (proportion)", min_value=0.0, max_value=1.0, value=p1, format="%.4f")
with c4:
    p2 = st.number_input("Group 2 risk (proportion)", min_value=0.0, max_value=1.0, value=p2, format="%.4f")

alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05)
power_goal = st.number_input("Target power", min_value=0.01, max_value=0.99, value=0.8)
ratio = n2 / n1 if n1 else 1.0
two_sided = st.checkbox("Two-sided test", value=True)

# Calculate everything up front
power = calc_power(n1, n2, p1, p2, alpha, two_sided)
n1_needed, n2_needed = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio)

adequacy = "Adequate" if power >= power_goal else "Not Adequate"
annotation = "✅ Adequate" if adequacy == "Adequate" else "❌ Not Adequate"

# Prepare summary table
summary = pd.DataFrame({
    "Group 1 N": [n1],
    "Group 2 N": [n2],
    "Risk 1": [p1],
    "Risk 2": [p2],
    "Estimated Power": [round(power,3)],
    "Required N1": [n1_needed if n1_needed else "N/A"],
    "Required N2": [n2_needed if n2_needed else "N/A"],
    "Adequacy": [annotation]
})

st.markdown("#### Summary Table")
st.dataframe(summary, hide_index=True)

# Clear statement
if adequacy == "Adequate":
    st.info(f"Your study is adequately powered (power = {power:.3f} ≥ {power_goal:.2f}).")
else:
    st.warning(f"Your study is **not adequately powered** (power = {power:.3f} < {power_goal:.2f}). Consider increasing sample size or adjusting effect size.")

st.caption("""
- Upload a TriNetX outcome CSV to auto-fill sample sizes and risks, or enter them manually.
- Estimated power and required sample sizes are always shown.
- A power of 0.8 or higher is considered adequate for most studies.
""")
