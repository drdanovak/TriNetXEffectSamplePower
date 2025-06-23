import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import io

st.set_page_config(layout="wide")
st.title("Multi-Finding Power & Sample Size Adequacy Calculator")

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

def extract_trinetx_stats(df, label=""):
    try:
        group1_n = int(float(df[10,2]))
        group2_n = int(float(df[11,2]))
        group1_risk = float(df[10,4])
        group2_risk = float(df[11,4])
        name = label if label else "Outcome"
        return {
            "Finding": name,
            "Group 1 N": group1_n,
            "Group 2 N": group2_n,
            "Risk 1": group1_risk,
            "Risk 2": group2_risk,
        }
    except Exception:
        return None

def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    if n1 <= 0 or n2 <= 0 or abs(p1 - p2) < 1e-9:
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

st.info("Upload one or more TriNetX CSVs (each for a different outcome), or enter findings manually in the table below.")

uploaded_files = st.file_uploader("📂 Upload TriNetX Outcome CSV(s)", type=["csv"], accept_multiple_files=True)

findings = []
if uploaded_files:
    for f in uploaded_files:
        label = f.name.rsplit(".", 1)[0]
        df = robust_csv_to_df(f)
        stats = extract_trinetx_stats(df, label)
        if stats is not None:
            findings.append(stats)

default_rows = [
    {"Finding": "Example Outcome", "Group 1 N": 100, "Group 2 N": 100, "Risk 1": 0.10, "Risk 2": 0.20}
]

if not findings:
    findings = default_rows

st.markdown("#### Add/Edit Your Findings")
edited_findings = st.data_editor(
    pd.DataFrame(findings),
    num_rows="dynamic",
    key="editable_table",
    use_container_width=True,
    column_config={
        "Finding": st.column_config.TextColumn("Finding Name"),
        "Group 1 N": st.column_config.NumberColumn("Group 1 N", min_value=1),
        "Group 2 N": st.column_config.NumberColumn("Group 2 N", min_value=1),
        "Risk 1": st.column_config.NumberColumn("Risk 1", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
        "Risk 2": st.column_config.NumberColumn("Risk 2", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"),
    }
)

alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05)
power_goal = st.number_input("Target power (for all findings)", min_value=0.01, max_value=0.99, value=0.8)
two_sided = st.checkbox("Two-sided test", value=True)

summary_rows = []
for idx, row in edited_findings.iterrows():
    n1 = int(row["Group 1 N"])
    n2 = int(row["Group 2 N"])
    p1 = float(row["Risk 1"])
    p2 = float(row["Risk 2"])
    ratio = n2 / n1 if n1 else 1.0
    est_power = calc_power(n1, n2, p1, p2, alpha, two_sided)
    req_n1, req_n2 = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio)
    adequacy = "Adequate" if est_power >= power_goal else "Not Adequate"
    annotation = "✅ Adequate" if adequacy == "Adequate" else "❌ Not Adequate"
    summary_rows.append({
        "Finding": row["Finding"],
        "Group 1 N": n1,
        "Group 2 N": n2,
        "Risk 1": p1,
        "Risk 2": p2,
        "Estimated Power": round(est_power, 3),
        "Required N1": req_n1 if req_n1 else "N/A",
        "Required N2": req_n2 if req_n2 else "N/A",
        "Adequacy": annotation
    })

summary = pd.DataFrame(summary_rows)

st.markdown("#### Summary Table")
st.dataframe(summary, hide_index=True)

csv = summary.to_csv(index=False)
st.download_button("Download Summary Table as CSV", csv, "power_adequacy_summary.csv")

# Clear group-level status
if (summary["Adequacy"] == "✅ Adequate").all():
    st.info("All findings are adequately powered.")
elif (summary["Adequacy"] == "❌ Not Adequate").all():
    st.warning("None of the findings are adequately powered. Consider increasing sample sizes or adjusting effect sizes.")
else:
    st.warning("Some findings are not adequately powered. See the table above for details.")

st.caption("""
- Upload multiple TriNetX outcome CSVs or add findings manually.
- Edit the table to change parameters or add new findings.
- All calculations update instantly.  
- Download your summary as a CSV for documentation or reporting.
""")
