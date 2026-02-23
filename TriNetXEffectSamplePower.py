import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import io

st.set_page_config(layout="wide")
st.title("Multi-Finding Power & Sample Size Adequacy Calculator")

# ----------------------------
# CSV ingestion helpers
# ----------------------------
def robust_csv_to_df(uploaded_file):
    raw = uploaded_file.read().decode("utf-8").splitlines()
    rows = []
    max_cols = 0
    for line in raw:
        comma_split = line.split(",")
        tab_split = line.split("\t")
        row = comma_split if len(comma_split) >= len(tab_split) else tab_split
        rows.append(row)
        max_cols = max(max_cols, len(row))
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return np.array(rows)

def extract_trinetx_stats(df, label=""):
    """
    This keeps your existing fixed-index extraction for TriNetX CSVs.
    Adjust indices if TriNetX changes their export layout.
    """
    try:
        group1_n = int(float(df[10, 2]))
        group2_n = int(float(df[11, 2]))
        group1_risk = float(df[10, 4])
        group2_risk = float(df[11, 4])
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

# ----------------------------
# Power / sample size
# ----------------------------
def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    if n1 <= 0 or n2 <= 0 or abs(p1 - p2) < 1e-12:
        return 0.0
    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z = diff / pooled_se
    if two_sided:
        power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    else:
        power = 1 - norm.cdf(z_alpha - z)
    return float(np.clip(power, 0, 1))

def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1):
    if abs(p1 - p2) < 1e-12:
        return None, None
    z_alpha = norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    num = (
        z_alpha * np.sqrt(2 * p_bar * q_bar)
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    ) ** 2
    denom = (p1 - p2) ** 2
    n1 = num / denom
    n2 = n1 * ratio
    return int(np.ceil(n1)), int(np.ceil(n2))

# ----------------------------
# E-value (sensitivity to unmeasured confounding)
# ----------------------------
def e_value_from_rr(rr: float) -> float | None:
    """
    Standard E-value for a risk ratio / rate ratio / hazard ratio.
    Handles protective effects by inverting RR < 1.
    Returns None for invalid inputs.
    """
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return None
    rr_use = rr if rr >= 1 else 1.0 / rr
    if rr_use <= 1:
        return 1.0
    return float(rr_use + np.sqrt(rr_use * (rr_use - 1.0)))

def rr_and_ci_from_risks(
    n1: int, n2: int, p1: float, p2: float, alpha: float = 0.05
) -> tuple[float | None, float | None, float | None]:
    """
    Computes RR = (p_exposed / p_unexposed) from risks plus an approximate Wald CI on log(RR).
    Uses a small continuity correction if event counts are 0 to avoid infinities.
    """
    if n1 <= 0 or n2 <= 0:
        return None, None, None
    if p1 < 0 or p2 < 0 or p1 > 1 or p2 > 1:
        return None, None, None
    if p1 == 0:
        return None, None, None  # RR undefined

    rr = (p2 / p1) if p1 > 0 else None
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return None, None, None

    # approximate event counts
    a = p2 * n2  # exposed events
    c = p1 * n1  # unexposed events

    # continuity correction if needed
    if a <= 0 or c <= 0:
        a = max(a, 0) + 0.5
        c = max(c, 0) + 0.5
        n2_cc = n2 + 1.0
        n1_cc = n1 + 1.0
    else:
        n2_cc = float(n2)
        n1_cc = float(n1)

    # SE(log RR) ≈ sqrt(1/a - 1/n2 + 1/c - 1/n1)
    try:
        se = np.sqrt((1.0 / a) - (1.0 / n2_cc) + (1.0 / c) - (1.0 / n1_cc))
        if not np.isfinite(se) or se <= 0:
            return rr, None, None
        z = norm.ppf(1 - alpha / 2)
        log_rr = np.log(rr)
        lo = float(np.exp(log_rr - z * se))
        hi = float(np.exp(log_rr + z * se))
        return float(rr), lo, hi
    except Exception:
        return float(rr), None, None

def e_value_for_ci_limit(rr: float | None, lo: float | None, hi: float | None) -> float | None:
    """
    E-value for the CI limit closest to the null (RR=1).
    If CI crosses 1, returns 1.0 (because a confounder of arbitrarily small strength could move to null).
    """
    if rr is None or lo is None or hi is None:
        return None
    if lo <= 1.0 <= hi:
        return 1.0
    # choose the bound closer to 1 on the RR scale, but respecting direction
    # for rr > 1 use lower bound; for rr < 1 use upper bound
    limit = lo if rr >= 1.0 else hi
    return e_value_from_rr(limit)

# ----------------------------
# UI
# ----------------------------
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

st.markdown("#### E-value settings")
rr_direction = st.radio(
    "How should the Risk Ratio be computed for the E-value?",
    options=["RR = Risk 2 / Risk 1 (treat Group 2 as exposed)", "RR = Risk 1 / Risk 2 (treat Group 1 as exposed)"],
    index=0,
    horizontal=False,
)

with st.expander("What does the E-value mean?", expanded=False):
    st.write(
        "The E-value is a sensitivity-analysis metric for unmeasured confounding. "
        "It tells you how strong an unmeasured confounder would have to be—"
        "in terms of its association with BOTH the exposure (group assignment) and the outcome—"
        "to fully explain away the observed association on the risk-ratio scale, "
        "after accounting for the covariates you already measured."
    )
    st.write(
        "Interpretation rules of thumb:\n"
        "- E-value = 1.0: no robustness; even very weak unmeasured confounding could move the estimate to the null.\n"
        "- Larger E-values: more robustness; you would need a stronger unmeasured confounder to explain the finding away.\n"
        "This does not address other threats (selection bias, measurement error, reverse causality), and it is not a causal proof—"
        "it is a quantified 'how strong would confounding have to be?' check."
    )

summary_rows = []
for idx, row in edited_findings.iterrows():
    n1 = int(row["Group 1 N"])
    n2 = int(row["Group 2 N"])
    p1 = float(row["Risk 1"])
    p2 = float(row["Risk 2"])

    # Power / sample size
    ratio = n2 / n1 if n1 else 1.0
    est_power = calc_power(n1, n2, p1, p2, alpha, two_sided)
    req_n1, req_n2 = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio)
    adequacy = "Adequate" if est_power >= power_goal else "Not Adequate"
    annotation = "✅ Adequate" if adequacy == "Adequate" else "❌ Not Adequate"

    # E-value uses RR. Choose direction.
    if "Risk 2 / Risk 1" in rr_direction:
        rr, lo, hi = rr_and_ci_from_risks(n1, n2, p1, p2, alpha=alpha)
    else:
        rr, lo, hi = rr_and_ci_from_risks(n2, n1, p2, p1, alpha=alpha)  # swapped

    e_pt = e_value_from_rr(rr) if rr is not None else None
    e_ci = e_value_for_ci_limit(rr, lo, hi) if (rr is not None and lo is not None and hi is not None) else None

    summary_rows.append(
        {
            "Finding": row["Finding"],
            "Group 1 N": n1,
            "Group 2 N": n2,
            "Risk 1": p1,
            "Risk 2": p2,
            "Estimated Power": round(est_power, 3),
            "Required N1": req_n1 if req_n1 else "N/A",
            "Required N2": req_n2 if req_n2 else "N/A",
            "Adequacy": annotation,
            "RR (for E-value)": (round(rr, 3) if rr is not None else "N/A"),
            "RR 95% CI": (f"{lo:.3f}–{hi:.3f}" if (lo is not None and hi is not None) else "N/A"),
            "E-value (point)": (round(e_pt, 2) if e_pt is not None else "N/A"),
            "E-value (CI)": (round(e_ci, 2) if e_ci is not None else "N/A"),
        }
    )

summary = pd.DataFrame(summary_rows)

st.markdown("#### Summary Table")
st.dataframe(summary, hide_index=True)

csv = summary.to_csv(index=False)
st.download_button("Download Summary Table as CSV", csv, "power_adequacy_summary_with_evalues.csv")

# Clear group-level status
if (summary["Adequacy"] == "✅ Adequate").all():
    st.info("All findings are adequately powered.")
elif (summary["Adequacy"] == "❌ Not Adequate").all():
    st.warning("None of the findings are adequately powered. Consider increasing sample sizes or adjusting effect sizes.")
else:
    st.warning("Some findings are not adequately powered. See the table above for details.")

st.caption(
    "- Upload multiple TriNetX outcome CSVs or add findings manually.\n"
    "- Edit the table to change parameters or add new findings.\n"
    "- Power/sample size calculations update instantly.\n"
    "- E-values are computed from the risk ratio derived from Risk 1 and Risk 2.\n"
    "- Download your summary as a CSV for documentation or reporting."
)
