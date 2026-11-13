"""Run `streamlit run theme_preview.py` to see the new look before touching app.py."""
import streamlit as st
from theme import inject_theme, hero, grade_card, verdict_card, balance_beam

st.set_page_config(page_title="FairLens", page_icon="🔎", layout="wide")
inject_theme()
hero()

grade_card(42, "Needs work: women are selected far less often than men",
           "Selection rates differ by 38 percentage points. Try reweighing or group thresholds below.")

st.write("")
st.subheader("Where the model breaks down")
left, right = st.columns([3, 2])
with left:
    balance_beam("Men", 0.584, "Women", 0.204)
with right:
    verdict_card("Demographic parity gap", "38.0 pts", "fail", "Rule of thumb: under 10 pts.")
    st.write("")
    verdict_card("Disparate impact", "0.35", "fail", "The 80% rule wants 0.80 or higher.")

st.write("")
c1, c2 = st.columns(2)
with c1:
    verdict_card("Equal opportunity gap", "7.2 pts", "warn", "Qualified people are caught at slightly different rates.")
with c2:
    verdict_card("Accuracy", "94.1%", "pass", "Overall accuracy says nothing about fairness by itself.")

tab1, tab2 = st.tabs(["Audit", "Mitigation"])
with tab1:
    st.button("Run audit")
