
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
from statsmodels.stats.power import NormalIndPower

st.set_page_config(layout="wide")
st.title("🧪 TriNetX Effect Size, Power Calculator & Forest Plot Generator")

st.markdown("""
Use this all-in-one tool to:
- Estimate **statistical power** for binary outcomes from TriNetX exports.
- Calculate **effect sizes** from Risk, Odds, or Hazard Ratios.
- Create **publication-ready forest plots**.
""")

# ----------------------------
# Power Calculator Section
# ----------------------------
st.header("1️⃣ Binary Outcome Power Calculator")

with st.expander("ℹ️ What is Power?", expanded=False):
    st.markdown("""
    **Power** is the probability that your study will detect a true effect when one exists (i.e., avoid a false negative).
    - A power of **0.8** (80%) or more is typically considered adequate.
    - This section calculates power using a **Z-test for two proportions**.
    """)

col1, col2 = st.columns(2)
with col1:
    n1 = st.number_input("Cohort 1 Sample Size", min_value=1, value=1437546)
    events1 = st.number_input("Cohort 1 Event Count", min_value=0, value=6263)
with col2:
    n2 = st.number_input("Cohort 2 Sample Size", min_value=1, value=1437546)
    events2 = st.number_input("Cohort 2 Event Count", min_value=0, value=1901)

alpha = st.slider("Significance Level (α)", min_value=0.001, max_value=0.1, value=0.05, step=0.001)

p1 = events1 / n1
p2 = events2 / n2
pooled_prob = (p1 + p2) / 2

if pooled_prob * (1 - pooled_prob) == 0:
    st.error("Invalid event rates. Must be between 0 and 1.")
else:
    effect_size = (p1 - p2) / np.sqrt(pooled_prob * (1 - pooled_prob))
    analysis = NormalIndPower()
    power = analysis.power(effect_size=effect_size, nobs1=n1, alpha=alpha, ratio=n2/n1, alternative='two-sided')

    st.markdown(f"""
    - **Cohort 1 Event Rate:** {p1:.4%}  
    - **Cohort 2 Event Rate:** {p2:.4%}  
    - **Risk Difference:** {(p1 - p2):.4%}  
    - **Effect Size (Z-based):** {effect_size:.3f}  
    - **Calculated Power:** **{power:.3f}**
    """)

# ----------------------------
# Effect Size & Table Section
# ----------------------------
st.header("2️⃣ Effect Size Calculator and Table")

st.sidebar.header("🛠️ Effect Size & Plot Options")
ratio_type = st.sidebar.selectbox("Type of Ratio Used", ["Risk Ratio", "Odds Ratio", "Hazard Ratio"], index=0)
add_p = st.sidebar.checkbox("Add p-value column")
add_ci = st.sidebar.checkbox("Add confidence interval columns (for ratios and effect sizes)")

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

# Process effect sizes
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

# Display Table
st.markdown("### 📋 Calculated Effect Sizes Table")
if not results_df.empty:
    html = results_df.to_html(index=False, escape=False)
    components.html(html, height=400, scrolling=True)
else:
    st.info("Enter at least one outcome and ratio value to compute effect sizes.")
