import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("TriNetX Effect Size & Power Calculator")

# ========== CSV UPLOAD & ROBUST READER ==========

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

st.sidebar.header("Data Import & Options")
uploaded_file = st.sidebar.file_uploader("📂 Upload TriNetX CSV (for auto fill)", type=["csv"])

csv_data = None
csv_extracted = False
csv_info = {}
if uploaded_file:
    df_csv = robust_csv_to_df(uploaded_file)
    # Try to extract key numbers for binary outcome
    try:
        group1_n = int(float(df_csv.iat[10,2]))
        group2_n = int(float(df_csv.iat[11,2]))
        group1_events = int(float(df_csv.iat[10,3]))
        group2_events = int(float(df_csv.iat[11,3]))
        group1_risk = float(df_csv.iat[10,4])
        group2_risk = float(df_csv.iat[11,4])
        risk_diff = float(df_csv.iat[16,0])
        risk_ratio = float(df_csv.iat[21,0])
        odds_ratio = float(df_csv.iat[26,0])
        csv_info = {
            "Group 1 n": group1_n,
            "Group 2 n": group2_n,
            "Group 1 events": group1_events,
            "Group 2 events": group2_events,
            "Group 1 risk": group1_risk,
            "Group 2 risk": group2_risk,
            "Risk difference": risk_diff,
            "Risk ratio": risk_ratio,
            "Odds ratio": odds_ratio
        }
        csv_extracted = True
        st.sidebar.success("Extracted values from CSV!")
    except Exception as e:
        st.sidebar.error(f"Could not extract key values from CSV ({e})")

# ========== POWER CALCULATOR ==========

def calc_power(n1, n2, p1, p2, alpha=0.05, two_sided=True):
    """Calculate power for two-group proportions (z-test)"""
    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    pooled_se = np.sqrt(p_bar * (1-p_bar) * (1/n1 + 1/n2))
    diff = abs(p1 - p2)
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z = diff / pooled_se
    power = norm.cdf(z - z_alpha) + (1 - norm.cdf(z + z_alpha))
    return power

def calc_sample_size(p1, p2, alpha=0.05, power=0.8, two_sided=True, ratio=1):
    """Calculate sample size per group for two-group proportions"""
    z_alpha = norm.ppf(1 - alpha/2 if two_sided else 1 - alpha)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    num = (z_alpha * np.sqrt(2 * p_bar * q_bar) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)/ratio)) ** 2
    denom = (p1 - p2) ** 2
    n1 = num / denom
    n2 = n1 * ratio
    return int(np.ceil(n1)), int(np.ceil(n2))

st.markdown("## Power Calculator for Binary Outcomes")

with st.expander("Power/Sample Size Calculator", expanded=True):
    st.write("Enter group sizes and risks below, or import them automatically by uploading a TriNetX outcomes CSV.")

    col1, col2 = st.columns(2)
    with col1:
        n1 = st.number_input("Group 1 sample size (n1)", min_value=1, value=csv_info.get("Group 1 n", 100))
        p1 = st.number_input("Group 1 risk (proportion)", min_value=0.0, max_value=1.0, value=csv_info.get("Group 1 risk", 0.1))
    with col2:
        n2 = st.number_input("Group 2 sample size (n2)", min_value=1, value=csv_info.get("Group 2 n", 100))
        p2 = st.number_input("Group 2 risk (proportion)", min_value=0.0, max_value=1.0, value=csv_info.get("Group 2 risk", 0.2))
    alpha = st.number_input("Significance level (alpha)", min_value=0.0001, max_value=0.5, value=0.05)
    power_goal = st.number_input("Target power (for sample size calc)", min_value=0.01, max_value=0.99, value=0.8)
    two_sided = st.checkbox("Two-sided test", value=True)

    mode = st.radio("What do you want to calculate?", ["Calculate Power (given N)", "Calculate Sample Size (given power)"])

    if mode == "Calculate Power (given N)":
        if n1 and n2 and 0 <= p1 < 1 and 0 <= p2 < 1:
            power = calc_power(n1, n2, p1, p2, alpha, two_sided)
            st.success(f"Power: **{power:.3f}** for n1={n1}, n2={n2}, p1={p1:.3f}, p2={p2:.3f}")
    else:
        if 0 <= p1 < 1 and 0 <= p2 < 1 and power_goal > 0:
            n1_needed, n2_needed = calc_sample_size(p1, p2, alpha, power_goal, two_sided, ratio=n2/n1 if n1 else 1)
            st.success(f"Sample Size Needed: **Group 1: {n1_needed}**, **Group 2: {n2_needed}** for power={power_goal}, alpha={alpha}")

    st.caption("Assumes a z-test for two independent proportions.")

# ========== EFFECT SIZE CALCULATOR AND FOREST PLOT (Original Functionality) ==========

st.markdown("---")
st.markdown("## Effect Size Calculator and Forest Plot")

st.sidebar.header("🛠️ Table and Plot Options")
ratio_type = st.sidebar.selectbox(
    "Type of Ratio Used",
    ["Risk Ratio", "Odds Ratio", "Hazard Ratio"],
    index=0
)
add_p = st.sidebar.checkbox("Add p-value column")
add_ci = st.sidebar.checkbox("Add confidence interval columns (for ratios and effect sizes)")
add_forest = st.sidebar.checkbox("Show forest plot of effect sizes")

if add_forest:
    st.sidebar.subheader("Forest Plot Settings")
    plot_title = st.sidebar.text_input("Plot Title", "Forest Plot")
    x_axis_label = st.sidebar.text_input("X-axis Label", f"Effect Size ({ratio_type})")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    show_values = st.sidebar.checkbox("Show Numerical Annotations", value=False)
    use_groups = st.sidebar.checkbox("Treat rows starting with '##' as section headers", value=True)
    with st.sidebar.expander("🎨 Advanced Visual Controls", expanded=False):
        color_scheme = st.selectbox("Color Scheme", ["Color", "Black & White"])
        point_size = st.slider("Marker Size", 6, 20, 10)
        line_width = st.slider("CI Line Width", 1, 4, 2)
        font_size = st.slider("Font Size", 10, 20, 12)
        label_offset = st.slider("Label Horizontal Offset", 0.01, 0.3, 0.05)
        use_log = st.checkbox("Use Log Scale for X-axis", value=False)
        axis_padding = st.slider("X-axis Padding (%)", 2, 40, 10)
        y_axis_padding = st.slider("Y-axis Padding (Rows)", 0.0, 5.0, 1.0, step=0.5)
        if color_scheme == "Color":
            ci_color = st.color_picker("CI Color", "#1f77b4")
            marker_color = st.color_picker("Point Color", "#d62728")
        else:
            ci_color = "black"
            marker_color = "black"

columns = ['Outcome', ratio_type]
defaults = {"Outcome": [""], ratio_type: [1.0]}
if add_ci:
    columns += ['Lower CI (Ratio)', 'Upper CI (Ratio)']
    defaults['Lower CI (Ratio)'] = [""]
    defaults['Upper CI (Ratio)'] = [""]
if add_p:
    columns += ['p-value']
    defaults['p-value'] = [""]

df = pd.DataFrame({col: defaults[col] for col in columns})

edited_df = st.data_editor(df, num_rows="dynamic", key="input_table", use_container_width=True)

results_df = edited_df.copy()
results_df = results_df[results_df['Outcome'].astype(str).str.strip() != ""]
results_df[ratio_type] = pd.to_numeric(results_df[ratio_type], errors='coerce')
results_df['Effect Size'] = np.log(np.abs(results_df[ratio_type])) * (np.sqrt(3) / np.pi) * np.sign(results_df[ratio_type])

if add_ci:
    results_df['Lower CI (Ratio)'] = pd.to_numeric(results_df['Lower CI (Ratio)'], errors='coerce')
    results_df['Upper CI (Ratio)'] = pd.to_numeric(results_df['Upper CI (Ratio)'], errors='coerce')
    results_df['Lower CI (Effect Size)'] = np.log(np.abs(results_df['Lower CI (Ratio)'])) * (np.sqrt(3) / np.pi) * np.sign(results_df['Lower CI (Ratio)'])
    results_df['Upper CI (Effect Size)'] = np.log(np.abs(results_df['Upper CI (Ratio)'])) * (np.sqrt(3) / np.pi) * np.sign(results_df['Upper CI (Ratio)'])

if add_p:
    results_df['p-value'] = pd.to_numeric(results_df['p-value'], errors='coerce')

def ama_table_html(df, ratio_label="Risk Ratio", ci=False, pval=False):
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
            <th>{ratio_label}</th>"""
    if ci:
        html += "<th>Lower CI (Ratio)</th><th>Upper CI (Ratio)</th>"
    html += "<th>Effect Size</th>"
    if ci:
        html += "<th>Lower CI (Effect Size)</th><th>Upper CI (Effect Size)</th>"
    if pval:
        html += "<th>p-value</th>"
    html += "</tr>"
    for _, row in df.iterrows():
        html += f"<tr><td class='left'>{row['Outcome']}</td><td>{row[ratio_label]}</td>"
        if ci:
            html += f"<td>{row.get('Lower CI (Ratio)','')}</td><td>{row.get('Upper CI (Ratio)','')}</td>"
        html += f"<td>{row['Effect Size']}</td>"
        if ci:
            html += f"<td>{row.get('Lower CI (Effect Size)','')}</td><td>{row.get('Upper CI (Effect Size)','')}</td>"
        if pval:
            html += f"<td>{row.get('p-value','')}</td>"
        html += "</tr>"
    html += "</table>"
    return html

st.markdown("### Calculated Effect Sizes Table")
if not results_df.empty:
    components.html(ama_table_html(results_df.round(6), ratio_label=ratio_type, ci=add_ci, pval=add_p), height=350, scrolling=True)
else:
    st.info("Enter at least one Outcome and Ratio to see results.")

def generate_forest_plot(
    df,
    plot_title="Forest Plot",
    x_axis_label="Effect Size (RR / OR / HR)",
    show_grid=True,
    show_values=False,
    use_groups=True,
    color_scheme="Color",
    point_size=10,
    line_width=2,
    font_size=12,
    label_offset=0.05,
    use_log=False,
    axis_padding=10,
    y_axis_padding=1.0,
    ci_color="#1f77b4",
    marker_color="#d62728"
):
    rows = []
    y_labels = []
    text_styles = []
    indent = "\u00A0" * 4
    group_mode = False

    for i, row in df.iterrows():
        if use_groups and isinstance(row["Outcome"], str) and row["Outcome"].startswith("##"):
            header = row["Outcome"][3:].strip()
            y_labels.append(header)
            text_styles.append("bold")
            rows.append(None)
            group_mode = True
        else:
            display_name = f"{indent}{row['Outcome']}" if group_mode else row["Outcome"]
            y_labels.append(display_name)
            text_styles.append("normal")
            rows.append(row)

    fig, ax = plt.subplots(figsize=(10, max(3, len(y_labels) * 0.7)))
    if (df['Lower CI (Effect Size)'].notnull().any() and df['Upper CI (Effect Size)'].notnull().any()):
        ci_vals = pd.concat([df['Lower CI (Effect Size)'].dropna(), df['Upper CI (Effect Size)'].dropna()])
        x_min, x_max = ci_vals.min(), ci_vals.max()
        x_pad = (x_max - x_min) * (axis_padding / 100)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    for i, row in enumerate(rows):
        if row is None:
            continue
        effect = row["Effect Size"]
        lci = row.get("Lower CI (Effect Size)", None) if "Lower CI (Effect Size)" in row else None
        uci = row.get("Upper CI (Effect Size)", None) if "Upper CI (Effect Size)" in row else None
        if pd.notnull(effect) and pd.notnull(lci) and pd.notnull(uci):
            ax.hlines(i, xmin=lci, xmax=uci, color=ci_color, linewidth=line_width, capstyle='round')
            ax.plot(effect, i, 'o', color=marker_color, markersize=point_size)
            if show_values:
                label = f"{effect:.2f} [{lci:.2f}, {uci:.2f}]"
                ax.text(uci + label_offset, i, label, va='center', fontsize=font_size - 2)

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(y_labels)))
    for tick_label, style in zip(ax.set_yticklabels(y_labels), text_styles):
        if style == "bold":
            tick_label.set_fontweight("bold")
        tick_label.set_fontsize(font_size)

    if use_log:
        ax.set_xscale('log')
    if show_grid:
        ax.grid(True, axis='x', linestyle=':', linewidth=0.6)
    else:
        ax.grid(False)

    ax.set_ylim(len(y_labels) - 1 + y_axis_padding, -1 - y_axis_padding)
    ax.set_xlabel(x_axis_label, fontsize=font_size)
    ax.set_title(plot_title, fontsize=font_size + 2, weight='bold')
    fig.tight_layout()
    return fig

if add_forest and not results_df.empty and add_ci:
    if st.button("📊 Generate Forest Plot"):
        fig = generate_forest_plot(
            results_df,
            plot_title=plot_title,
            x_axis_label=x_axis_label,
            show_grid=show_grid,
            show_values=show_values,
            use_groups=use_groups,
            color_scheme=color_scheme,
            point_size=point_size,
            line_width=line_width,
            font_size=font_size,
            label_offset=label_offset,
            use_log=use_log,
            axis_padding=axis_padding,
            y_axis_padding=y_axis_padding,
            ci_color=ci_color,
            marker_color=marker_color,
        )
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="forest_plot.png", mime="image/png")

st.markdown("---")
st.markdown(
    "**Citation for Original Study on Computing Effect Sizes from Odds Ratios:**  \n"
    "Chinn S. A simple method for converting an odds ratio to effect size for use in meta-analysis. "
    "Stat Med. 2000;19(22):3127-3131. "
    "[doi:10.1002/1097-0258(20001130)19:22<3127::aid-sim784>3.0.co;2-m](https://doi.org/10.1002/1097-0258(20001130)19:22<3127::aid-sim784>3.0.co;2-m), PMID: 11113947"
)
