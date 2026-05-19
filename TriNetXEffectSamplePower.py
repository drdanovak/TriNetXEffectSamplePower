import numpy as np
import pandas as pd
import streamlit as st
import html
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="TriNetX Outcomes Interpretation Tool")
st.title("TriNetX Outcomes: Power, E-value, NNT/NNH, and Standardized Effect Size")

st.caption(
    "Upload TriNetX outcome CSVs or enter values manually. The tool separates statistical power, "
    "E-value sensitivity, absolute clinical impact, and standardized effect magnitude so users can interpret each finding correctly."
)

# ============================================================
# CSV ingestion helpers
# ============================================================
def robust_csv_to_array(uploaded_file) -> np.ndarray:
    """Parse irregular TriNetX CSV/TXT-style exports into a rectangular array."""
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
    Extracts the common TriNetX outcomes table layout used in this toolkit.

    Assumed layout:
      Group 1 row: index 10
      Group 2 row: index 11
      N column:     index 2
      Risk column:  index 4

    If TriNetX changes its export layout, adjust these indices or add additional parsers.
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


# ============================================================
# Formatting helpers
# ============================================================
def fmt_float(x, digits=3, na="N/A"):
    if x is None or not np.isfinite(x):
        return na
    return f"{x:.{digits}f}"


def fmt_percent(x, digits=2, na="N/A"):
    if x is None or not np.isfinite(x):
        return na
    return f"{100 * x:.{digits}f}%"


def fmt_ci(lo, hi, digits=3, na="N/A"):
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return na
    return f"{lo:.{digits}f} to {hi:.{digits}f}"


def fmt_nnt(x):
    if x is None:
        return "N/A"
    if not np.isfinite(x):
        return "∞"
    return f"{x:.1f}"


# ============================================================
# Power / sample size: two-proportion approximation
# ============================================================
def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    """Approximate power for a two-sample comparison of proportions."""
    if n1 <= 0 or n2 <= 0 or p1 is None or p2 is None:
        return None
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        return None
    if abs(p1 - p2) < 1e-12:
        return 0.0

    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
    if pooled_se <= 0 or not np.isfinite(pooled_se):
        return None

    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z = diff / pooled_se

    if two_sided:
        power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    else:
        power = 1 - norm.cdf(z_alpha - z)

    return float(np.clip(power, 0, 1))


def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1.0):
    """Approximate required sample size for a two-sample comparison of proportions."""
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


def classify_power(power_value, target=0.80):
    if power_value is None or not np.isfinite(power_value):
        return "Not interpretable"
    if power_value >= 0.90:
        return "High power"
    if power_value >= target:
        return "Adequate power"
    if power_value >= 0.50:
        return "Underpowered"
    return "Very underpowered"


def explain_power(power_value, target=0.80):
    label = classify_power(power_value, target)
    if power_value is None or not np.isfinite(power_value):
        return "Power could not be estimated from these inputs. Check sample sizes and risks."

    pct = fmt_percent(power_value, 1)
    target_pct = fmt_percent(target, 0)

    if power_value >= 0.90:
        return (
            f"Estimated power is {pct}, which is high. Under the assumptions of this two-proportion test, "
            f"the available sample is very likely to detect a difference of the observed magnitude at the selected alpha level."
        )
    if power_value >= target:
        return (
            f"Estimated power is {pct}, which meets the target of {target_pct}. "
            f"This finding is adequately powered for a difference of the observed magnitude."
        )
    if power_value >= 0.50:
        return (
            f"Estimated power is {pct}, which is below the target of {target_pct}. "
            f"This finding may be vulnerable to false-negative interpretation if the observed association is not statistically significant."
        )
    return (
        f"Estimated power is {pct}, which is very low. The current sample is unlikely to reliably detect a difference "
        f"of the observed magnitude. Interpret null or borderline findings cautiously."
    )


# ============================================================
# E-value: risk-ratio scale sensitivity to unmeasured confounding
# ============================================================
def e_value_from_rr(rr: float):
    """
    E-value for a point estimate on the risk-ratio scale.
    For protective effects, the reciprocal RR is used.
    """
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return None
    rr_use = rr if rr >= 1 else 1.0 / rr
    if rr_use <= 1:
        return 1.0
    return float(rr_use + np.sqrt(rr_use * (rr_use - 1.0)))


def rr_and_ci_from_risks(n_t: int, n_c: int, p_t: float, p_c: float, alpha: float = 0.05):
    """
    RR = treated/exposed risk divided by control/unexposed risk.
    Approximate Katz/Wald CI on the log scale.
    """
    if n_t <= 0 or n_c <= 0:
        return None, None, None
    if p_t is None or p_c is None:
        return None, None, None
    if not (0 <= p_t <= 1 and 0 <= p_c <= 1):
        return None, None, None
    if p_c == 0:
        return None, None, None

    rr = p_t / p_c
    if rr <= 0 or not np.isfinite(rr):
        return None, None, None

    a = p_t * n_t
    c = p_c * n_c

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
    E-value for the confidence-limit closest to the null.
    If the CI crosses 1, the CI-limit E-value is 1.0.
    """
    if rr is None or lo is None or hi is None:
        return None
    if lo <= 1.0 <= hi:
        return 1.0
    limit = lo if rr >= 1.0 else hi
    return e_value_from_rr(limit)


def classify_e_value(e_value):
    """
    Rule-of-thumb classification for robustness to unmeasured confounding.
    These are not universal thresholds and should not be treated as causal proof.
    """
    if e_value is None or not np.isfinite(e_value):
        return "Not interpretable"
    if e_value < 1.25:
        return "Minimal robustness"
    if e_value < 1.50:
        return "Small robustness"
    if e_value < 2.00:
        return "Moderate robustness"
    if e_value < 3.00:
        return "Large robustness"
    return "Very large robustness"


def explain_e_value(e_value, e_value_ci=None):
    label = classify_e_value(e_value)
    if e_value is None or not np.isfinite(e_value):
        return "The E-value could not be calculated. This usually occurs when the risk ratio is undefined."

    base = (
        f"An E-value of {e_value:.2f} indicates {label.lower()} to unmeasured confounding. "
        f"An unmeasured confounder would need to be associated with both the exposure and outcome by a risk ratio of at least {e_value:.2f}, "
        f"above and beyond measured covariates, to fully explain away the observed association."
    )

    if e_value_ci is not None and np.isfinite(e_value_ci):
        if e_value_ci <= 1.0:
            base += (
                " The confidence interval crosses or approaches the null, so the CI-limit E-value is 1.00; "
                "the precision-adjusted robustness of this finding is weak."
            )
        else:
            base += (
                f" The CI-limit E-value is {e_value_ci:.2f}, meaning that confounding of at least that magnitude "
                f"would be needed to move the confidence interval to include the null."
            )

    base += " This is a sensitivity metric, not a conventional effect size and not proof of causality."
    return base


# ============================================================
# NNT / NNH: absolute risk difference scale
# ============================================================
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
    Converts absolute risk difference into NNT or NNH.

    outcome_is_adverse=True:
      lower risk is better; benefit if RD < 0.

    outcome_is_adverse=False:
      higher risk is better; benefit if RD > 0.
    """
    if rd is None or not np.isfinite(rd):
        return "NNT/NNH", None, "N/A", None, None

    if abs(rd) < 1e-12:
        return "NNT/NNH", np.inf, "RD≈0 → ∞", None, None

    benefit_mag = (-rd) if outcome_is_adverse else rd

    if benefit_mag > 0:
        label = "NNT"
        effect_abs = benefit_mag
        if rd_lo is not None and rd_hi is not None:
            effect_lo, effect_hi = ((-rd_hi), (-rd_lo)) if outcome_is_adverse else (rd_lo, rd_hi)
        else:
            effect_lo, effect_hi = None, None
    else:
        label = "NNH"
        effect_abs = -benefit_mag
        if rd_lo is not None and rd_hi is not None:
            b_lo, b_hi = ((-rd_hi), (-rd_lo)) if outcome_is_adverse else (rd_lo, rd_hi)
            effect_lo, effect_hi = (-b_hi, -b_lo)
        else:
            effect_lo, effect_hi = None, None

    point = (1.0 / effect_abs) if effect_abs > 0 else np.inf

    if effect_lo is None or effect_hi is None:
        return label, float(point), "N/A", None, None

    effect_lo, effect_hi = min(effect_lo, effect_hi), max(effect_lo, effect_hi)
    if effect_lo <= 0 <= effect_hi:
        return label, float(point), "CI crosses null", None, None

    ci_lo = 1.0 / effect_hi
    ci_hi = 1.0 / effect_lo
    return label, float(point), f"{ci_lo:.1f} to {ci_hi:.1f}", ci_lo, ci_hi


def classify_nnt(nnt_point):
    """
    Context-dependent rule of thumb for absolute clinical impact.
    Smaller NNT/NNH values indicate larger absolute impact.
    """
    if nnt_point is None or not np.isfinite(nnt_point):
        return "Not interpretable"
    if nnt_point < 10:
        return "Large absolute impact"
    if nnt_point < 25:
        return "Moderate absolute impact"
    if nnt_point < 50:
        return "Small-to-moderate absolute impact"
    return "Small absolute impact"


def explain_nnt(label, nnt_point, rd, rd_lo, rd_hi, outcome_is_adverse=True):
    if nnt_point is None:
        return "NNT/NNH could not be calculated from these inputs."

    if not np.isfinite(nnt_point):
        return (
            "The absolute risk difference is approximately zero, so the NNT/NNH approaches infinity. "
            "This indicates little or no absolute difference between groups on the observed risk scale."
        )

    impact = classify_nnt(nnt_point).lower()
    rd_text = fmt_percent(abs(rd), 2)

    if label == "NNT":
        if outcome_is_adverse:
            return (
                f"The absolute risk reduction is {rd_text}, giving an NNT of {nnt_point:.1f}. "
                f"This means approximately {nnt_point:.1f} exposed/treated patients would be needed to prevent one additional adverse event "
                f"over the outcome window. This suggests {impact}, though NNT interpretation depends on outcome severity, follow-up time, cost, and competing risks."
            )
        return (
            f"The absolute increase in beneficial outcome is {rd_text}, giving an NNT of {nnt_point:.1f}. "
            f"This means approximately {nnt_point:.1f} exposed/treated patients would be needed for one additional beneficial outcome "
            f"over the outcome window. This suggests {impact}, though interpretation depends on clinical context."
        )

    if label == "NNH":
        if outcome_is_adverse:
            return (
                f"The absolute risk increase is {rd_text}, giving an NNH of {nnt_point:.1f}. "
                f"This means approximately one additional adverse event would occur for every {nnt_point:.1f} exposed/treated patients "
                f"over the outcome window. This suggests {impact}; smaller NNH values indicate more clinically important harm."
            )
        return (
            f"The absolute decrease in beneficial outcome is {rd_text}, giving an NNH of {nnt_point:.1f}. "
            f"This means approximately one fewer beneficial outcome would occur for every {nnt_point:.1f} exposed/treated patients "
            f"over the outcome window. This suggests {impact}; smaller NNH values indicate more clinically important loss of benefit."
        )

    return "NNT/NNH interpretation was not available."


# ============================================================
# Standardized effect size for binary/proportion outcomes
# ============================================================
def cohens_h_from_proportions(p_t, p_c):
    """
    Cohen's h for two proportions.

    For binary/proportion outcomes, Cohen's h is the appropriate Cohen-family standardized effect size.
    It is often more appropriate than Cohen's d, which is designed for continuous outcomes.
    """
    if p_t is None or p_c is None:
        return None
    if not (0 <= p_t <= 1 and 0 <= p_c <= 1):
        return None
    return float(2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c)))


def classify_cohens_h(h):
    if h is None or not np.isfinite(h):
        return "Not interpretable"
    ah = abs(h)
    if ah < 0.20:
        return "Very small/trivial standardized effect"
    if ah < 0.50:
        return "Small standardized effect"
    if ah < 0.80:
        return "Medium standardized effect"
    return "Large standardized effect"


def explain_cohens_h(h, p_t, p_c):
    if h is None or not np.isfinite(h):
        return "The standardized effect size could not be calculated from these inputs."

    label = classify_cohens_h(h).lower()
    direction = "higher" if h > 0 else "lower"
    return (
        f"The Cohen-family standardized effect size for these two risks is h = {h:.3f}, which is a {label}. "
        f"Because these are binary risk outcomes, Cohen's h is used rather than Cohen's d. "
        f"The exposed/treated group has a {direction} risk than the comparison group "
        f"({fmt_percent(p_t, 2)} vs. {fmt_percent(p_c, 2)})."
    )


# ============================================================
# Narrative synthesis
# ============================================================
def generate_overall_interpretation(row):
    name = row["Finding"]
    parts = [f"For {name}, the observed treated/exposed risk was {row['Treated Risk']} compared with {row['Control Risk']} in the comparison group."]

    parts.append(row["Power Explanation"])
    parts.append(row["E-value Explanation"])
    parts.append(row["NNT/NNH Explanation"])
    parts.append(row["Standardized Effect Explanation"])

    if row.get("Notes"):
        parts.append(f"Input note: {row['Notes']}")

    return " ".join(parts)


# ============================================================
# UI: data input
# ============================================================
with st.sidebar:
    st.header("Settings")
    alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01)
    power_goal = st.number_input("Target power", min_value=0.01, max_value=0.99, value=0.80, step=0.01)
    two_sided = st.checkbox("Two-sided test", value=True)

    st.divider()
    st.subheader("Group direction")
    rr_direction = st.radio(
        "Define treated/exposed group:",
        options=[
            "Treat Group 2 as treated/exposed",
            "Treat Group 1 as treated/exposed",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Outcome interpretation")
    outcome_type = st.radio(
        "How should NNT/NNH interpret the outcome?",
        options=[
            "Adverse event (lower risk is better)",
            "Beneficial event (higher risk is better)",
        ],
        index=0,
    )
    outcome_is_adverse = outcome_type.startswith("Adverse")

    st.divider()
    st.caption(
        "Note: E-values are sensitivity metrics for unmeasured confounding. "
        "For binary/proportion outcomes, this app reports Cohen's h rather than Cohen's d."
    )

uploaded_files = st.file_uploader(
    "Upload TriNetX Outcome CSV(s)",
    type=["csv", "txt"],
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
        else:
            st.warning(f"Could not automatically parse {f.name}. You can still enter the values manually below.")

if not findings:
    findings = [
        {"Finding": "Example Outcome", "Group 1 N": 100, "Group 2 N": 100, "Risk 1": 0.10, "Risk 2": 0.20}
    ]

st.subheader("1. Add or edit findings")
st.write("Enter risks as proportions, not percentages. For example, enter 0.12 for 12%.")

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


# ============================================================
# Compute component-specific results
# ============================================================
summary_rows = []
component_rows = []

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

    if rr_direction == "Treat Group 2 as treated/exposed":
        n_t, n_c, p_t, p_c = n2, n1, p2, p1
        treated_label = "Group 2"
        control_label = "Group 1"
    else:
        n_t, n_c, p_t, p_c = n1, n2, p1, p2
        treated_label = "Group 1"
        control_label = "Group 2"

    est_power = req_n1 = req_n2 = None
    power_label = "Not interpretable"
    power_explanation = "Power could not be interpreted because one or more inputs were invalid."

    rr = lo_rr = hi_rr = None
    e_pt = e_ci = None
    e_label = "Not interpretable"
    e_explanation = "E-value could not be interpreted because one or more inputs were invalid."

    rd = rd_lo = rd_hi = None
    nnt_label = "NNT/NNH"
    nnt_point = None
    nnt_ci = "N/A"
    nnt_ci_lo = nnt_ci_hi = None
    nnt_impact = "Not interpretable"
    nnt_explanation = "NNT/NNH could not be interpreted because one or more inputs were invalid."

    cohens_h = None
    cohens_h_label = "Not interpretable"
    cohens_h_explanation = "Standardized effect size could not be interpreted because one or more inputs were invalid."

    if not notes:
        ratio = (n2 / n1) if n1 else 1.0

        est_power = calc_power(n1, n2, p1, p2, alpha=alpha, two_sided=two_sided)
        req_n1, req_n2 = calc_sample_size(p1, p2, alpha=alpha, power=power_goal, two_sided=two_sided, ratio=ratio)
        power_label = classify_power(est_power, target=power_goal)
        power_explanation = explain_power(est_power, target=power_goal)

        rr, lo_rr, hi_rr = rr_and_ci_from_risks(n_t, n_c, p_t, p_c, alpha=alpha)
        e_pt = e_value_from_rr(rr) if rr is not None else None
        e_ci = e_value_for_ci_limit(rr, lo_rr, hi_rr) if (rr is not None and lo_rr is not None and hi_rr is not None) else None
        e_label = classify_e_value(e_pt)
        e_explanation = explain_e_value(e_pt, e_ci)

        rd, rd_lo, rd_hi = risk_diff_and_ci(n_t, n_c, p_t, p_c, alpha=alpha)
        nnt_label, nnt_point, nnt_ci, nnt_ci_lo, nnt_ci_hi = nnt_nnh_from_rd(
            rd, rd_lo, rd_hi, outcome_is_adverse=outcome_is_adverse
        )
        nnt_impact = classify_nnt(nnt_point)
        nnt_explanation = explain_nnt(nnt_label, nnt_point, rd, rd_lo, rd_hi, outcome_is_adverse=outcome_is_adverse)

        cohens_h = cohens_h_from_proportions(p_t, p_c)
        cohens_h_label = classify_cohens_h(cohens_h)
        cohens_h_explanation = explain_cohens_h(cohens_h, p_t, p_c)

    summary_row = {
        "Finding": name,
        "Treated/Exposed Group": treated_label,
        "Control/Comparison Group": control_label,
        "Group 1 N": n1 if n1 is not None else "N/A",
        "Group 2 N": n2 if n2 is not None else "N/A",
        "Risk 1": fmt_percent(p1, 2) if p1 is not None else "N/A",
        "Risk 2": fmt_percent(p2, 2) if p2 is not None else "N/A",
        "Treated Risk": fmt_percent(p_t, 2) if p_t is not None else "N/A",
        "Control Risk": fmt_percent(p_c, 2) if p_c is not None else "N/A",
        "Power": fmt_percent(est_power, 1) if est_power is not None else "N/A",
        "Power Interpretation": power_label,
        "Required N1": req_n1 if req_n1 is not None else "N/A",
        "Required N2": req_n2 if req_n2 is not None else "N/A",
        "RR": fmt_float(rr, 3),
        "RR 95% CI": fmt_ci(lo_rr, hi_rr, 3),
        "E-value": fmt_float(e_pt, 2),
        "E-value CI-limit": fmt_float(e_ci, 2),
        "E-value Interpretation": e_label,
        "Risk Difference": fmt_percent(rd, 2) if rd is not None else "N/A",
        "RD 95% CI": f"{fmt_percent(rd_lo, 2)} to {fmt_percent(rd_hi, 2)}" if rd_lo is not None and rd_hi is not None else "N/A",
        "NNT/NNH": f"{nnt_label} = {fmt_nnt(nnt_point)}",
        "NNT/NNH 95% CI": nnt_ci,
        "NNT/NNH Interpretation": nnt_impact,
        "Cohen's h": fmt_float(cohens_h, 3),
        "Cohen-style Effect Interpretation": cohens_h_label,
        "Power Explanation": power_explanation,
        "E-value Explanation": e_explanation,
        "NNT/NNH Explanation": nnt_explanation,
        "Standardized Effect Explanation": cohens_h_explanation,
        "Notes": "; ".join(notes) if notes else "",
    }

    summary_row["Narrative Interpretation"] = generate_overall_interpretation(summary_row)
    summary_rows.append(summary_row)

    component_rows.extend(
        [
            {
                "Finding": name,
                "Component": "Power",
                "Value": fmt_percent(est_power, 1) if est_power is not None else "N/A",
                "Interpretation": power_label,
                "Explanation": power_explanation,
            },
            {
                "Finding": name,
                "Component": "E-value",
                "Value": fmt_float(e_pt, 2),
                "Interpretation": e_label,
                "Explanation": e_explanation,
            },
            {
                "Finding": name,
                "Component": "NNT/NNH",
                "Value": f"{nnt_label} = {fmt_nnt(nnt_point)}",
                "Interpretation": nnt_impact,
                "Explanation": nnt_explanation,
            },
            {
                "Finding": name,
                "Component": "Cohen-style standardized effect size",
                "Value": fmt_float(cohens_h, 3),
                "Interpretation": cohens_h_label,
                "Explanation": cohens_h_explanation,
            },
        ]
    )

summary = pd.DataFrame(summary_rows)
components = pd.DataFrame(component_rows)


# ============================================================
# UI: results display
# ============================================================
st.subheader("2. Review component cards")
st.write(
    "Start here. Each card separates one interpretive metric and uses color-coded thresholds to help users quickly distinguish "
    "strong, adequate, borderline, low, or potentially concerning findings."
)

st.markdown(
    """
<style>
.metric-card {
    border-radius: 18px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 0.9rem;
    border: 1.5px solid #d0d7de;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}
.metric-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.metric-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111827;
}
.status-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
}
.metric-value {
    font-size: 2.0rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 0.35rem;
    color: #111827;
}
.metric-interpretation {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
    color: #1f2937;
}
.metric-explanation {
    font-size: 0.9rem;
    line-height: 1.45;
    color: #374151;
}
.metric-success {
    background: #ecfdf5;
    border-color: #10b981;
}
.metric-success .status-pill {
    background: #d1fae5;
    color: #065f46;
}
.metric-warning {
    background: #fffbeb;
    border-color: #f59e0b;
}
.metric-warning .status-pill {
    background: #fef3c7;
    color: #92400e;
}
.metric-danger {
    background: #fef2f2;
    border-color: #ef4444;
}
.metric-danger .status-pill {
    background: #fee2e2;
    color: #991b1b;
}
.metric-info {
    background: #eff6ff;
    border-color: #3b82f6;
}
.metric-info .status-pill {
    background: #dbeafe;
    color: #1e40af;
}
.metric-magnitude {
    background: #f5f3ff;
    border-color: #8b5cf6;
}
.metric-magnitude .status-pill {
    background: #ede9fe;
    color: #5b21b6;
}
.metric-neutral {
    background: #f8fafc;
    border-color: #cbd5e1;
}
.metric-neutral .status-pill {
    background: #e2e8f0;
    color: #334155;
}
.threshold-note {
    font-size: 0.85rem;
    color: #475569;
    margin-top: -0.25rem;
    margin-bottom: 0.9rem;
}
</style>
    """,
    unsafe_allow_html=True,
)


def escape_text(value):
    """Safely escape values shown inside custom HTML cards."""
    if value is None:
        return "N/A"
    return html.escape(str(value))


def power_card_status(interpretation):
    interp = str(interpretation).lower()
    if "high" in interp:
        return "metric-success", "High / exceeds target"
    if "adequate" in interp:
        return "metric-success", "Meets target"
    if "very underpowered" in interp:
        return "metric-danger", "Low / below threshold"
    if "underpowered" in interp:
        return "metric-warning", "Borderline / below target"
    return "metric-neutral", "Not interpretable"


def evalue_card_status(interpretation):
    interp = str(interpretation).lower()
    if "very large" in interp:
        return "metric-success", "High robustness"
    if "large" in interp:
        return "metric-success", "High robustness"
    if "moderate" in interp:
        return "metric-info", "Moderate robustness"
    if "small" in interp:
        return "metric-warning", "Low robustness"
    if "minimal" in interp:
        return "metric-danger", "Very low robustness"
    return "metric-neutral", "Not interpretable"


def nnt_card_status(value, interpretation):
    interp = str(interpretation).lower()
    value_text = str(value).upper()
    is_harm = value_text.startswith("NNH")

    if "not interpretable" in interp:
        return "metric-neutral", "Not interpretable"

    if is_harm:
        if "large absolute" in interp:
            return "metric-danger", "High harm signal"
        if "moderate" in interp:
            return "metric-warning", "Moderate harm signal"
        if "small" in interp:
            return "metric-info", "Low harm signal"
        return "metric-warning", "Potential harm"

    if "large absolute" in interp:
        return "metric-success", "High benefit / impact"
    if "moderate" in interp:
        return "metric-info", "Moderate impact"
    if "small" in interp:
        return "metric-neutral", "Low absolute impact"
    return "metric-neutral", "Not interpretable"


def standardized_effect_card_status(interpretation):
    interp = str(interpretation).lower()
    if "large" in interp:
        return "metric-magnitude", "High magnitude"
    if "medium" in interp:
        return "metric-info", "Medium magnitude"
    if "small standardized" in interp:
        return "metric-warning", "Small magnitude"
    if "very small" in interp or "trivial" in interp:
        return "metric-neutral", "Very small magnitude"
    return "metric-neutral", "Not interpretable"


def colored_metric_card(title, value, interpretation, explanation, status_class, status_label):
    st.markdown(
        f"""
<div class="metric-card {status_class}">
    <div class="metric-card-header">
        <span class="metric-title">{escape_text(title)}</span>
        <span class="status-pill">{escape_text(status_label)}</span>
    </div>
    <div class="metric-value">{escape_text(value)}</div>
    <div class="metric-interpretation">{escape_text(interpretation)}</div>
    <div class="metric-explanation">{escape_text(explanation)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def style_component_rows(row):
    """Apply light background colors to the component-level table based on interpretation."""
    component = str(row.get("Component", "")).lower()
    interpretation = str(row.get("Interpretation", ""))
    value = str(row.get("Value", ""))

    if component == "power":
        status_class, _ = power_card_status(interpretation)
    elif component == "e-value":
        status_class, _ = evalue_card_status(interpretation)
    elif component == "nnt/nnh":
        status_class, _ = nnt_card_status(value, interpretation)
    elif "standardized" in component:
        status_class, _ = standardized_effect_card_status(interpretation)
    else:
        status_class = "metric-neutral"

    color_map = {
        "metric-success": "background-color: #ecfdf5;",
        "metric-warning": "background-color: #fffbeb;",
        "metric-danger": "background-color: #fef2f2;",
        "metric-info": "background-color: #eff6ff;",
        "metric-magnitude": "background-color: #f5f3ff;",
        "metric-neutral": "background-color: #f8fafc;",
    }
    style = color_map.get(status_class, "")
    return [style] * len(row)


# These tabs must be created before any `with tab_components:`, `with tab_summary:`,
# `with tab_report:`, or `with tab_methods:` blocks. Do not paste an individual
# tab block at the top of the file by itself.
tab_components, tab_summary, tab_report, tab_methods = st.tabs(
    ["Component cards", "Metric summary tables", "Narrative report", "Methods notes"]
)

with tab_components:
    if summary.empty:
        st.info("No valid findings to display.")
    else:
        selected_finding = st.selectbox("Choose a finding", summary["Finding"].tolist())
        r = summary[summary["Finding"] == selected_finding].iloc[0]

        st.markdown(f"### {selected_finding}")
        st.caption(
            f"Orientation check: {r['Treated/Exposed Group']} is treated/exposed and {r['Control/Comparison Group']} is the comparison group. "
            f"Observed risks: {r['Treated Risk']} vs. {r['Control Risk']}."
        )

        col1, col2 = st.columns(2)

        with col1:
            status_class, status_label = power_card_status(r["Power Interpretation"])
            colored_metric_card(
                "Power",
                r["Power"],
                r["Power Interpretation"],
                r["Power Explanation"],
                status_class,
                status_label,
            )

            status_class, status_label = nnt_card_status(r["NNT/NNH"], r["NNT/NNH Interpretation"])
            colored_metric_card(
                "Absolute impact: NNT/NNH",
                r["NNT/NNH"],
                r["NNT/NNH Interpretation"],
                r["NNT/NNH Explanation"],
                status_class,
                status_label,
            )

        with col2:
            status_class, status_label = evalue_card_status(r["E-value Interpretation"])
            colored_metric_card(
                "E-value sensitivity",
                r["E-value"],
                r["E-value Interpretation"],
                r["E-value Explanation"],
                status_class,
                status_label,
            )

            status_class, status_label = standardized_effect_card_status(r["Cohen-style Effect Interpretation"])
            colored_metric_card(
                "Standardized effect size",
                r["Cohen's h"],
                r["Cohen-style Effect Interpretation"],
                r["Standardized Effect Explanation"],
                status_class,
                status_label,
            )

        st.markdown(
            "<div class='threshold-note'>Color key: green = meets target or high benefit/robustness; yellow = low or borderline; "
            "red = very low or concerning; blue/purple = magnitude or moderate robustness rather than inherently good or bad.</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### Metric-level table for selected finding")
        selected_components = components[components["Finding"] == selected_finding]
        st.dataframe(
            selected_components.style.apply(style_component_rows, axis=1),
            hide_index=True,
            use_container_width=True,
        )

with tab_summary:
    st.write(
        "The results are separated by metric so users can review each finding without scrolling across one very wide table. "
        "Each table answers a different interpretive question."
    )

    def show_metric_table(title, description, columns, dataframe=summary):
        """Display a focused table for one interpretive metric."""
        existing_columns = [c for c in columns if c in dataframe.columns]
        if not existing_columns:
            return

        st.markdown(f"### {title}")
        st.caption(description)
        st.dataframe(dataframe[existing_columns], hide_index=True, use_container_width=True)

    show_metric_table(
        "Cohort and risk overview",
        "Use this table first to confirm that the treated/exposed group, comparison group, sample sizes, and risks are oriented correctly.",
        [
            "Finding",
            "Treated/Exposed Group",
            "Control/Comparison Group",
            "Group 1 N",
            "Group 2 N",
            "Risk 1",
            "Risk 2",
            "Treated Risk",
            "Control Risk",
            "Notes",
        ],
    )

    show_metric_table(
        "Power analysis",
        "Power addresses whether the available sample is likely to detect a difference of the observed magnitude. It should not be interpreted as clinical importance.",
        [
            "Finding",
            "Power",
            "Power Interpretation",
            "Required N1",
            "Required N2",
            "Power Explanation",
        ],
    )

    show_metric_table(
        "E-value sensitivity analysis",
        "The E-value describes robustness to unmeasured confounding on the risk-ratio scale. It is not a conventional effect size and is not causal proof.",
        [
            "Finding",
            "RR",
            "RR 95% CI",
            "E-value",
            "E-value CI-limit",
            "E-value Interpretation",
            "E-value Explanation",
        ],
    )

    show_metric_table(
        "Absolute impact: risk difference and NNT/NNH",
        "NNT/NNH translates the absolute risk difference into an interpretable clinical impact metric. Smaller NNT or NNH values indicate larger absolute impact.",
        [
            "Finding",
            "Risk Difference",
            "RD 95% CI",
            "NNT/NNH",
            "NNT/NNH 95% CI",
            "NNT/NNH Interpretation",
            "NNT/NNH Explanation",
        ],
    )

    show_metric_table(
        "Standardized effect size for binary outcomes",
        "For two risk/proportion outcomes, the Cohen-family standardized effect size is Cohen's h rather than Cohen's d.",
        [
            "Finding",
            "Cohen's h",
            "Cohen-style Effect Interpretation",
            "Standardized Effect Explanation",
        ],
    )

    with st.expander("Show full combined results table"):
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.divider()
    col_download1, col_download2 = st.columns(2)

    with col_download1:
        csv_bytes = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full results as CSV",
            data=csv_bytes,
            file_name="trinetx_interpreted_outcomes_summary.csv",
            mime="text/csv",
        )

    with col_download2:
        component_csv_bytes = components.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download metric-level results as CSV",
            data=component_csv_bytes,
            file_name="trinetx_metric_level_interpretations.csv",
            mime="text/csv",
        )

with tab_report:
    st.write("This report is intended to give users language they can adapt for methods checks, internal review, or manuscript planning.")

    report_lines = []
    line_break = chr(10)

    for _, r in summary.iterrows():
        finding_name = str(r.get("Finding", "Outcome"))
        narrative_text = str(r.get("Narrative Interpretation", ""))

        st.markdown("### " + finding_name)
        st.write(narrative_text)

        report_block = "## " + finding_name + line_break + line_break + narrative_text + line_break
        report_lines.append(report_block)

    report_text = line_break.join(report_lines)
    st.download_button(
        "Download narrative report as Markdown",
        data=report_text.encode("utf-8"),
        file_name="trinetx_interpretive_report.md",
        mime="text/markdown",
    )

with tab_methods:
    st.markdown("### Interpretation rules used by this tool")

    st.markdown(
        """
**Power** estimates the probability of detecting a difference of the observed magnitude under a two-proportion test approximation. Values at or above the selected target are labeled adequate; values below the target are flagged as underpowered.

**E-value** is a sensitivity analysis metric for unmeasured confounding on the risk-ratio scale. Larger values indicate that a stronger unmeasured confounder would be needed to fully explain away the observed association. The E-value is not a conventional effect size and is not causal proof.

**NNT/NNH** is calculated from the absolute risk difference. For adverse outcomes, lower risk in the treated/exposed group is interpreted as benefit and higher risk as harm. For beneficial outcomes, this direction is reversed. Smaller NNT or NNH values indicate larger absolute clinical impact, but clinical meaning depends on outcome severity, follow-up duration, baseline risk, and intervention burden.

**Cohen-style standardized effect size** is reported as Cohen's h because TriNetX outcome risks are proportions. Cohen's d is generally used for continuous outcomes; Cohen's h is the appropriate Cohen-family standardized effect size for comparing two proportions. Rule-of-thumb thresholds are: <0.20 very small/trivial, 0.20 to <0.50 small, 0.50 to <0.80 medium, and ≥0.80 large.

**Color coding** is used as an interpretive aid. Green indicates that a metric meets its target or suggests high benefit/robustness. Yellow indicates borderline or low values. Red indicates very low robustness, insufficient power, or a potentially concerning harm signal. Blue and purple indicate magnitude or moderate robustness rather than inherently good or bad findings.
        """
    )

st.caption(
    "Statistical notes: This app uses approximate Wald/Katz intervals and two-proportion power calculations. "
    "For manuscript-grade analyses, verify against the exact TriNetX model outputs, study design, matching strategy, censoring structure, and outcome time window."
)
