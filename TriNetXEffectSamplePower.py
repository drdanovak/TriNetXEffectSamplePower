import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("TriNetX Outcomes: Power + E-value + NNT/NNH Calculator")

# ----------------------------
# CSV ingestion helpers
# ----------------------------
def robust_csv_to_array(uploaded_file) -> np.ndarray:
    raw = uploaded_file.read().decode("utf-8", errors="replace").splitlines()
    rows = []
    max_cols = 0
    for line in raw:
        comma_split = line.split(",")
        tab_split = line.split("\t")
        row = comma_split if len(comma_split) >= len(tab_split) else tab_split
        rows.append(row)
        max_cols = max(max_cols, len(row))
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return np.array(rows, dtype=object)

def extract_trinetx_stats(arr: np.ndarray, label: str = ""):
    """
    Matches the layout you’ve been using:
      Group 1 row at [10,*], Group 2 row at [11,*]
      N at col 2, Risk at col 4
    Adjust indices if TriNetX changes their export layout.
    """
    try:
        group1_n = int(float(arr[10, 2]))
        group2_n = int(float(arr[11, 2]))
        group1_risk = float(arr[10, 4])
        group2_risk = float(arr[11, 4])
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

def safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

# ----------------------------
# Power / sample size (two-proportion)
# ----------------------------
def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    if n1 <= 0 or n2 <= 0 or p1 is None or p2 is None:
        return 0.0
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        return 0.0
    if abs(p1 - p2) < 1e-12:
        return 0.0

    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    if pooled_se <= 0 or not np.isfinite(pooled_se):
        return 0.0

    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z = diff / pooled_se

    if two_sided:
        power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    else:
        power = 1 - norm.cdf(z_alpha - z)

    return float(np.clip(power, 0, 1))

def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1.0):
    if p1 is None or p2 is None:
        return None, None
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        return None, None
    if abs(p1 - p2) < 1e-12:
        return None, None
    if ratio <= 0 or not np.isfinite(ratio):
        ratio = 1.0

    z_alpha = norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar

    num = (
        z_alpha * np.sqrt(2 * p_bar * q_bar)
        + z_beta * np.sqrt(p1 * (1 - p1) + (p2 * (1 - p2) / ratio))
    ) ** 2
    denom = (p1 - p2) ** 2

    n1 = num / denom
    n2 = n1 * ratio
    return int(np.ceil(n1)), int(np.ceil(n2))

# ----------------------------
# E-value (risk ratio scale)
# ----------------------------
def e_value_from_rr(rr: float):
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return None
    rr_use = rr if rr >= 1 else 1.0 / rr
    if rr_use <= 1:
        return 1.0
    return float(rr_use + np.sqrt(rr_use * (rr_use - 1.0)))

def rr_and_ci_from_risks(n_t: int, n_c: int, p_t: float, p_c: float, alpha: float = 0.05):
    """
    RR = p_t / p_c with approximate Katz/Wald CI on log scale.
    Uses a small continuity correction if events are 0.
    """
    if n_t <= 0 or n_c <= 0:
        return None, None, None
    if p_t is None or p_c is None:
        return None, None, None
    if not (0 <= p_t <= 1 and 0 <= p_c <= 1):
        return None, None, None
    if p_c == 0:
        return None, None, None  # RR undefined/infinite on risk-ratio scale

    rr = p_t / p_c
    if rr <= 0 or not np.isfinite(rr):
        return None, None, None

    # Approximate event counts from risks
    a = p_t * n_t  # treated events
    c = p_c * n_c  # control events

    # Continuity correction if either count is 0
    if a <= 0 or c <= 0:
        a = max(a, 0) + 0.5
        c = max(c, 0) + 0.5
        n_t_cc = n_t + 1.0
        n_c_cc = n_c + 1.0
    else:
        n_t_cc = float(n_t)
        n_c_cc = float(n_c)

    try:
        se = np.sqrt((1.0 / a) - (1.0 / n_t_cc) + (1.0 / c) - (1.0 / n_c_cc))
        if not np.isfinite(se) or se <= 0:
            return float(rr), None, None
        z = norm.ppf(1 - alpha / 2)
        log_rr = np.log(rr)
        lo = float(np.exp(log_rr - z * se))
        hi = float(np.exp(log_rr + z * se))
        return float(rr), lo, hi
    except Exception:
        return float(rr), None, None

def e_value_for_ci_limit(rr: float, lo: float, hi: float):
    """
    E-value for the CI limit closest to the null (RR=1).
    If CI crosses 1, return 1.0.
    """
    if rr is None or lo is None or hi is None:
        return None
    if lo <= 1.0 <= hi:
        return 1.0
    limit = lo if rr >= 1.0 else hi
    return e_value_from_rr(limit)

# ----------------------------
# NNT / NNH (risk difference scale)
# ----------------------------
def risk_diff_and_ci(n_t: int, n_c: int, p_t: float, p_c: float, alpha: float = 0.05):
    if n_t <= 0 or n_c <= 0:
        return None, None, None
    if p_t is None or p_c is None:
        return None, None, None
    if not (0 <= p_t <= 1 and 0 <= p_c <= 1):
        return None, None, None

    rd = p_t - p_c
    se = np.sqrt((p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c))
    if not np.isfinite(se) or se < 0:
        return float(rd), None, None
    z = norm.ppf(1 - alpha / 2)
    lo = rd - z * se
    hi = rd + z * se
    return float(rd), float(lo), float(hi)

def nnt_nnh_from_rd(rd, rd_lo, rd_hi, outcome_is_adverse=True):
    """
    outcome_is_adverse:
      True  -> lower risk is better (benefit if RD < 0)
      False -> higher risk is better (benefit if RD > 0)

    Returns: (label, point, ci_text)
    """
    if rd is None or not np.isfinite(rd):
        return "NNT/NNH", None, "N/A"

    if abs(rd) < 1e-12:
        return "NNT/NNH", np.inf, "RD≈0 → ∞"

    # Benefit magnitude (positive means benefit)
    benefit_mag = (-rd) if outcome_is_adverse else rd

    if benefit_mag > 0:
        label = "NNT"
        eff = benefit_mag
        if rd_lo is not None and rd_hi is not None:
            eff_lo, eff_hi = ((-rd_hi), (-rd_lo)) if outcome_is_adverse else (rd_lo, rd_hi)
        else:
            eff_lo, eff_hi = None, None
    else:
        label = "NNH"
        eff = -benefit_mag
        if rd_lo is not None and rd_hi is not None:
            b_lo, b_hi = ((-rd_hi), (-rd_lo)) if outcome_is_adverse else (rd_lo, rd_hi)  # benefit_mag CI
            eff_lo, eff_hi = (-b_hi, -b_lo)  # harm_mag = -benefit_mag
        else:
            eff_lo, eff_hi = None, None

    point = (1.0 / eff) if eff > 0 else np.inf

    # CI by inversion: only meaningful if effect CI does NOT cross 0 on that effect scale
    if eff_lo is None or eff_hi is None:
        return label, float(point), "N/A"

    eff_lo, eff_hi = min(eff_lo, eff_hi), max(eff_lo, eff_hi)
    if eff_lo <= 0 <= eff_hi:
        return label, float(point), "CI crosses null"

    ci_lo = 1.0 / eff_hi
    ci_hi = 1.0 / eff_lo
    return label, float(point), f"{ci_lo:.1f}–{ci_hi:.1f}"

# ----------------------------
# UI
# ----------------------------
st.info("Upload one or more TriNetX CSVs (each for a different outcome), or enter findings manually.")

uploaded_files = st.file_uploader(
    "📂 Upload TriNetX Outcome CSV(s)",
    type=["csv"],
    accept_multiple_files=True,
)

findings = []
if uploaded_files:
    for f in uploaded_files:
        label = f.name.rsplit(".", 1)[0]
        arr = robust_csv_to_array(f)
        stats = extract_trinetx_stats(arr, label=label)
        if stats is not None:
            findings.append(stats)

if not findings:
    findings = [
        {"Finding": "Example Outcome", "Group 1 N": 100, "Group 2 N": 100, "Risk 1": 0.10, "Risk 2": 0.20}
    ]

st.write("Add/edit outcomes below (you can paste rows, add rows, and change risks and sample sizes).")
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
    },
)

st.write("Settings")
alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05)
power_goal = st.number_input("Target power (for all findings)", min_value=0.01, max_value=0.99, value=0.80)
two_sided = st.checkbox("Two-sided test", value=True)

st.write("E-value settings")
rr_direction = st.radio(
    "Define “treated/exposed” group for RR/E-value and RD/NNT calculations:",
    options=[
        "Treat Group 2 as treated/exposed (RR = Risk 2 / Risk 1; RD = Risk 2 - Risk 1)",
        "Treat Group 1 as treated/exposed (RR = Risk 1 / Risk 2; RD = Risk 1 - Risk 2)",
    ],
    index=0,
)

st.write("NNT/NNH settings")
outcome_type = st.radio(
    "How should NNT/NNH interpret the outcome?",
    options=[
        "Adverse event (lower risk is better)",
        "Beneficial event (higher risk is better)",
    ],
    index=0,
)
outcome_is_adverse = outcome_type.startswith("Adverse")

with st.expander("What does the E-value mean?", expanded=False):
    st.write(
        "The E-value is a sensitivity-analysis metric for unmeasured confounding. "
        "It tells you how strong an unmeasured confounder would have to be—"
        "in terms of its association with BOTH the exposure (group assignment) and the outcome—"
        "to fully explain away the observed association on the risk-ratio scale, after accounting for measured covariates."
    )
    st.write(
        "Interpretation rules of thumb:\n"
        "• E-value = 1.0: minimal robustness (very weak confounding could move the estimate to the null).\n"
        "• Larger E-values: more robustness (a stronger unmeasured confounder would be needed).\n"
        "This does not address other threats (selection bias, measurement error, reverse causality) and is not causal proof."
    )

with st.expander("What does NNT/NNH mean?", expanded=False):
    st.write(
        "NNT/NNH is computed from the absolute risk difference (RD) between treated/exposed and control/unexposed.\n"
        "• If the outcome is adverse and treatment reduces risk, NNT is how many patients must be treated to prevent 1 additional event.\n"
        "• If treatment increases risk, NNH is how many patients treated would produce 1 additional harm.\n"
        "If RD is near 0, NNT/NNH becomes very large (approaches infinity). If the RD CI crosses 0, the NNT/NNH CI is not interpretable."
    )

# ----------------------------
# Compute summary table
# ----------------------------
summary_rows = []
for _, row in edited_findings.iterrows():
    name = str(row.get("Finding", "")).strip() if row.get("Finding", "") is not None else "Outcome"

    n1 = safe_int(row.get("Group 1 N"))
    n2 = safe_int(row.get("Group 2 N"))
    p1 = safe_float(row.get("Risk 1"))
    p2 = safe_float(row.get("Risk 2"))

    notes = []
    if n1 is None or n2 is None or n1 <= 0 or n2 <= 0:
        notes.append("Invalid N")
    if p1 is None or p2 is None or not (0 <= (p1 if p1 is not None else -1) <= 1) or not (0 <= (p2 if p2 is not None else -1) <= 1):
        notes.append("Invalid risk")

    # Defaults for computed fields
    est_power = None
    req_n1 = None
    req_n2 = None
    adequacy = "N/A"
    annotation = "N/A"

    rr = lo_rr = hi_rr = None
    e_pt = e_ci = None

    rd = rd_lo = rd_hi = None
    nnt_label = "NNT/NNH"
    nnt_point = None
    nnt_ci = "N/A"

    if not notes:
        ratio = (n2 / n1) if n1 else 1.0
        est_power = calc_power(n1, n2, p1, p2, alpha, two_sided)
        req_n1, req_n2 = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio)

        adequacy = "Adequate" if est_power >= power_goal else "Not Adequate"
        annotation = "✅ Adequate" if adequacy == "Adequate" else "❌ Not Adequate"

        # Define treated/exposed vs control to keep RR/E-value and RD/NNT consistent
        if rr_direction.startswith("Treat Group 2"):
            n_t, n_c, p_t, p_c = n2, n1, p2, p1
        else:
            n_t, n_c, p_t, p_c = n1, n2, p1, p2

        rr, lo_rr, hi_rr = rr_and_ci_from_risks(n_t, n_c, p_t, p_c, alpha=alpha)
        e_pt = e_value_from_rr(rr) if rr is not None else None
        e_ci = e_value_for_ci_limit(rr, lo_rr, hi_rr) if (rr is not None and lo_rr is not None and hi_rr is not None) else None

        rd, rd_lo, rd_hi = risk_diff_and_ci(n_t, n_c, p_t, p_c, alpha=alpha)
        nnt_label, nnt_point, nnt_ci = nnt_nnh_from_rd(rd, rd_lo, rd_hi, outcome_is_adverse=outcome_is_adverse)

    summary_rows.append(
        {
            "Finding": name,
            "Group 1 N": n1 if n1 is not None else "N/A",
            "Group 2 N": n2 if n2 is not None else "N/A",
            "Risk 1": round(p1, 4) if isinstance(p1, (float, int)) else "N/A",
            "Risk 2": round(p2, 4) if isinstance(p2, (float, int)) else "N/A",
            "Estimated Power": round(est_power, 3) if est_power is not None else "N/A",
            "Required N1": req_n1 if req_n1 is not None else "N/A",
            "Required N2": req_n2 if req_n2 is not None else "N/A",
            "Adequacy": annotation,
            "RR (treated/control)": round(rr, 3) if rr is not None else "N/A",
            "RR 95% CI": f"{lo_rr:.3f}–{hi_rr:.3f}" if (lo_rr is not None and hi_rr is not None) else "N/A",
            "E-value (point)": round(e_pt, 2) if e_pt is not None else "N/A",
            "E-value (CI)": round(e_ci, 2) if e_ci is not None else "N/A",
            "Risk Difference (T-C)": round(rd, 4) if rd is not None else "N/A",
            "RD 95% CI": f"{rd_lo:.4f}–{rd_hi:.4f}" if (rd_lo is not None and rd_hi is not None) else "N/A",
            "NNT/NNH": (
                f"{nnt_label} = {nnt_point:.1f}" if (nnt_point is not None and np.isfinite(nnt_point)) else f"{nnt_label} = ∞"
            ),
            "NNT/NNH 95% CI": nnt_ci if nnt_ci is not None else "N/A",
            "Notes": "; ".join(notes) if notes else "",
        }
    )

summary = pd.DataFrame(summary_rows)

st.write("Summary Table")
st.dataframe(summary, hide_index=True, use_container_width=True)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Summary Table as CSV",
    data=csv_bytes,
    file_name="trinetx_power_evalue_nnt_summary.csv",
    mime="text/csv",
)

# Overall status
if summary.shape[0] > 0 and "Adequacy" in summary.columns:
    adequacy_vals = summary["Adequacy"].astype(str).tolist()
    if all(v == "✅ Adequate" for v in adequacy_vals if v != "N/A"):
        st.info("All valid findings are adequately powered for the target power.")
    elif all(v == "❌ Not Adequate" for v in adequacy_vals if v != "N/A") and any(v != "N/A" for v in adequacy_vals):
        st.warning("None of the valid findings are adequately powered. Consider larger samples or different effect sizes.")
    else:
        st.warning("Some findings are not adequately powered (or have invalid inputs). See the Notes column where applicable.")

st.caption(
    "Notes: E-values here are computed from an RR derived from the absolute risks you provide. "
    "NNT/NNH is computed from the absolute risk difference and is inherently tied to the follow-up window used to compute those risks."
)
