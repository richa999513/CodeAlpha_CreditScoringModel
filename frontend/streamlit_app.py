# python -m streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# ==========================================================
# 1️⃣ DATA & SAMPLE MAPPING
# ==========================================================
CAT_COLS = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
]

# Sample data from your raw dataset (Cleaned for the model)
SAMPLES = {
    "Low Risk (sample)": {
        "person_age": 30, "person_income": 80000, "person_home_ownership": "RENT",
        "person_emp_length": 5, "loan_intent": "PERSONAL", "loan_grade": "B",
        "loan_amnt": 5000, "loan_int_rate": 8.5, "loan_percent_income": 0.06,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 3
    },
    "High Risk (sample)": {
        "person_age": 22, "person_income": 12000, "person_home_ownership": "RENT",
        "person_emp_length": 1, "loan_intent": "PERSONAL", "loan_grade": "D",
        "loan_amnt": 20000, "loan_int_rate": 18.0, "loan_percent_income": 0.4,
        "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 1
    }
}

@st.cache_data
def get_feature_schema():
    """Return numeric feature list and mapping of categorical options.

    We read a slice of the loan data so the front end can offer
    selectboxes for low-cardinality categorical columns. """
    try:
        df = pd.read_csv('../loan/credit_risk_dataset.csv', nrows=5000)
        all_num = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude IDs and Target
        num_features = [c for c in all_num if c not in ['target_bin', 'loan_status', 'id', 'member_id']]

        cat_values = {}
        for c in CAT_COLS:
            if c in df.columns:
                # dropna helps avoid NaN entries
                values = df[c].dropna().unique().tolist()
                # sort for consistency
                cat_values[c] = sorted(values)
        return num_features, cat_values
    except Exception:
        return [], {}

num_features, cat_values = get_feature_schema()
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# ==========================================================
# 2️⃣ HEADER & SAMPLE SELECTOR
# ==========================================================
st.title("💳 Loan Default Risk Dashboard")

st.subheader("Step 1: Load Data or Enter Manually")
selected_sample = st.selectbox("Choose a sample to auto-fill:", ["None"] + list(SAMPLES.keys()))

if selected_sample != "None":
    st.session_state.inputs = SAMPLES[selected_sample]
    st.toast(f"Form updated with {selected_sample}")

# ==========================================================
# 3️⃣ TAB-BASED INPUT FORM
# ==========================================================
user_data = {}

with st.form("risk_form"):
    tab1, tab2 = st.tabs(["📋 Categorical Features (10)", "🔢 Numeric Features (51)"])

    with tab1:
        cols = st.columns(2)
        for i, col in enumerate(CAT_COLS):
            with cols[i % 2]:
                default_val = st.session_state.inputs.get(col, "")
                options = cat_values.get(col, [])
                if options:
                    user_data[col] = st.selectbox(
                        col,
                        options=["", *options],
                        index=0 if default_val == "" else (options.index(default_val) + 1 if default_val in options else 0),
                        key=f"cat_{col}"
                    )
                else:
                    user_data[col] = st.text_input(col, value=str(default_val), key=f"cat_{col}")

    with tab2:
        st.caption("All numeric variables required by the model")
        cols = st.columns(4)
        for i, col in enumerate(num_features):
            with cols[i % 4]:
                default_val = float(st.session_state.inputs.get(col, 0.0))
                user_data[col] = st.number_input(col, value=default_val, key=f"num_{col}")

    submit = st.form_submit_button("Analyze Loan Risk", type="primary")

# ==========================================================
# 4️⃣ PREDICTION RESULTS
# ==========================================================
if submit:
    try:
        r = requests.post("http://localhost:8000/predict", json={"data": user_data})
        if r.ok:
            res = r.json()
            prob = res.get('probability') or res.get('probability_of_default')
            pred_text = res.get('prediction', '')

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                if prob is not None:
                    if prob > 0.5:
                        st.error(f"### Result: REJECT\nProbability of default: {prob:.2%}")
                    else:
                        st.success(f"### Result: APPROVE\nProbability of default: {prob:.2%}")
                    if pred_text:
                        st.write(f"**Model says:** {pred_text}")
                else:
                    st.warning("Response did not contain a probability field.")
            with c2:
                if prob is not None:
                    st.write("**Risk Analysis Visualization**")
                    st.progress(prob)
                    st.caption("The progress bar represents the likelihood of default (0% to 100%).")
        else:
            st.error(f"API Error: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")