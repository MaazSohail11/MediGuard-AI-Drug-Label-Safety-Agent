"""
app.py  —  MediGuard AI Medication Safety Agent
Run: streamlit run app.py
"""

import sys
from pathlib import Path

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="MediGuard — AI Medication Safety",
    page_icon="💊",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💊 MediGuard — AI Medication Safety Agent")
st.markdown("*RAG · Forward Chaining · Random Forest · FDA Drug Labels*")
st.divider()


# ── Lazy-load heavy components with Streamlit caching ─────────────────────────
@st.cache_resource(show_spinner="Loading AI components…")
def _load():
    from decision_manager      import analyze
    from explanation_generator import generate_explanation
    return analyze, generate_explanation


try:
    analyze, generate_explanation = _load()
    _ready = True
except Exception as _err:
    _ready     = False
    _err_msg   = str(_err)


# ── Sidebar — input form ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient & Medication Details")

    medicine_1 = st.text_input("Medicine 1 *",         placeholder="e.g. buspirone")
    medicine_2 = st.text_input("Medicine 2",            placeholder="e.g. MAOI / phenelzine")
    age        = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    symptoms   = st.text_area("Symptoms",               placeholder="e.g. anxiety, dizziness", height=70)
    condition  = st.text_area("Medical Condition",      placeholder="e.g. depression, hypertension", height=70)
    allergy    = st.text_input("Known Allergies",        placeholder="e.g. penicillin  (or 'none')")
    food       = st.text_input("Food / Drink",           placeholder="e.g. grapefruit juice, alcohol")

    check_btn = st.button("🔍 Check Safety", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**About MediGuard**")
    st.caption(
        "MediGuard queries 260 000+ FDA drug-label records via vector-based RAG, "
        "applies forward-chaining safety rules, and uses a trained Random Forest "
        "classifier to produce a Low / Moderate / High risk assessment."
    )


# ── Main area ─────────────────────────────────────────────────────────────────
if not _ready:
    st.error(f"Failed to load AI components:\n\n`{_err_msg}`")
    st.info(
        "Ensure the following setup steps are complete:\n"
        "1. `python src/preprocess_json.py`\n"
        "2. `python src/train_ml_model.py`\n"
        "3. `python src/build_vector_db.py`"
    )
    st.stop()

if not check_btn:
    st.info("👈 Fill in the medication details on the left and click **Check Safety**.")

    with st.expander("ℹ️ How MediGuard Works"):
        st.markdown("""
**Pipeline overview:**

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | **RAG Retrieval** | Searches ChromaDB vector store of 260 000+ FDA drug labels |
| 2 | **Forward Chaining** | Applies IF→THEN safety rules over retrieved text |
| 3 | **Random Forest** | ML model trained on FDA label features predicts risk level |
| 4 | **Decision Manager** | Fuses signals — rule engine has highest priority |
| 5 | **Explanation** | Generates structured advice, sources, and disclaimer |

**PEAS Description:**
- **Performance** — Correct risk classification, low false-negative rate for High-risk cases
- **Environment** — FDA drug labels, user inputs (medicines, age, allergies, food)
- **Actuators** — Risk card, warning, advice, source list, follow-up questions
- **Sensors** — Medicine names, age, symptoms, allergies, medical conditions, diet
        """)
    st.stop()

# ── Validate input ─────────────────────────────────────────────────────────────
if not medicine_1.strip():
    st.warning("Please enter at least Medicine 1.")
    st.stop()

# ── Run full analysis ──────────────────────────────────────────────────────────
with st.spinner("Analysing medication safety…"):
    try:
        decision    = analyze(
            medicine_1 = medicine_1.strip(),
            medicine_2 = medicine_2.strip(),
            age        = int(age),
            symptoms   = symptoms.strip(),
            condition  = condition.strip(),
            allergy    = allergy.strip(),
            food       = food.strip(),
        )
        explanation = generate_explanation(
            medicine_1 = medicine_1.strip(),
            medicine_2 = medicine_2.strip(),
            decision   = decision,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ── Risk banner ────────────────────────────────────────────────────────────────
risk  = explanation["risk_level"]
icon  = explanation["risk_icon"]
COLORS = {"High": "#ff4b4b", "Moderate": "#ffa500", "Low": "#21c354"}

c1, c2, c3 = st.columns(3)
c1.metric("Risk Level",     f"{icon}  {risk}")
c2.metric("Risk Score",     f"{explanation['risk_score']} / 100")
c3.metric("Confidence",     f"{explanation['confidence']} %")

st.divider()

if risk == "High":
    st.error(f"⚠️  **{explanation['warning']}**")
elif risk == "Moderate":
    st.warning(f"⚡  **{explanation['warning']}**")
else:
    st.success(f"✅  {explanation['warning']}")

st.divider()

# ── Detail columns ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("🔍 Why This Happens")
    st.write(explanation["why"])

    st.subheader("📌 Advice")
    st.info(explanation["advice"])

    st.subheader("📚 Retrieved FDA Sources")
    for src in explanation["sources"] or ["No sources retrieved."]:
        st.markdown(f"- {src}")

with right:
    st.subheader("⚙️ System Analysis")
    st.markdown(f"**Rule-Based Result :** `{explanation['rule_result']}`")
    st.markdown(f"**ML Prediction     :** `{explanation['ml_result']}`")

    if explanation["triggered_rules"]:
        with st.expander(f"Triggered Rules ({len(explanation['triggered_rules'])})"):
            for r in explanation["triggered_rules"]:
                st.markdown(f"- `{r}`")

    st.subheader("🤖 ML Class Probabilities")
    proba = decision.get("ml_probabilities", {})
    for cls, prob in sorted(proba.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(prob * 20))
        st.markdown(f"`{cls:<10}` {bar}  {prob*100:.1f} %")

    if explanation["follow_up_questions"]:
        st.subheader("❓ Follow-Up Questions")
        for q in explanation["follow_up_questions"]:
            st.markdown(f"- {q}")

# ── Expanded raw context ───────────────────────────────────────────────────────
st.divider()
with st.expander("📄 View Retrieved FDA Label Sections"):
    chunks = decision.get("context", [])
    if not chunks:
        st.write("No context retrieved.")
    for i, chunk in enumerate(chunks, 1):
        st.markdown(
            f"**[{i}]  {chunk['drug_name']}  —  {chunk['section_name']}**  "
            f"*(cosine distance: {chunk['distance']})*"
        )
        text = chunk["text"]
        st.markdown(f"> {text[:450]}{'…' if len(text) > 450 else ''}")
        st.divider()

# ── Feedback ───────────────────────────────────────────────────────────────────
st.subheader("💬 Was this helpful?")
fb_col1, fb_col2 = st.columns(2)
with fb_col1:
    if st.button("👍  Helpful", use_container_width=True):
        st.success("Thank you for your feedback!")
with fb_col2:
    if st.button("👎  Not Helpful", use_container_width=True):
        st.info("Thank you. Your feedback helps us improve MediGuard.")

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"⚠️  {explanation['disclaimer']}")
