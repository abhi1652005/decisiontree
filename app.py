import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1e3a8a;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Campus Placement Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter academic and professional background details to check eligibility</div>', unsafe_allow_html=True)

# Load the Model Safely
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'decisionc.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), None
    else:
        return None, f"Model file 'decisionc.pkl' not found at {model_path}"

model, model_error = load_model()

if model_error:
    st.error(f"**Critical Error:** {model_error}\nPlease make sure `decisionc.pkl` is pushed to your GitHub repository.")
    st.stop()

# Input Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        workex = st.selectbox("Work Experience", options=["No", "Yes"])
        ssc_p = st.number_input("10th (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
        ssc_b = st.selectbox("10th Board", options=["Central", "Others"])
        hsc_p = st.number_input("12th (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
        hsc_b = st.selectbox("12th Board", options=["Central", "Others"])

    with col2:
        hsc_s = st.selectbox("12th Specialization", options=["Commerce", "Science", "Arts"])
        degree_p = st.number_input("Degree (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
        degree_t = st.selectbox("Degree Field", options=["Comm&Mgmt", "Sci&Tech", "Others"])
        etest_p = st.number_input("Employability Test (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        specialisation = st.selectbox("MBA Specialization", options=["Mkt&HR", "Mkt&Fin"])
        mba_p = st.number_input("MBA (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    submit_button = st.form_submit_button(label="Predict Placement Status")

if submit_button:
    try:
        # Mapping inputs to standard model-friendly numerical encodings
        gender_map = {'Male': 1, 'Female': 0}
        board_map = {'Central': 1, 'Others': 0}
        hsc_s_map = {'Commerce': 1, 'Science': 2, 'Arts': 0}
        degree_t_map = {'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2}
        workex_map = {'Yes': 1, 'No': 0}
        specialisation_map = {'Mkt&Fin': 0, 'Mkt&HR': 1}

        input_data = pd.DataFrame([{
            'gender': gender_map.get(gender, 1),
            'ssc_p': float(ssc_p),
            'ssc_b': board_map.get(ssc_b, 1),
            'hsc_p': float(hsc_p),
            'hsc_b': board_map.get(hsc_b, 1),
            'hsc_s': hsc_s_map.get(hsc_s, 1),
            'degree_p': float(degree_p),
            'degree_t': degree_t_map.get(degree_t, 0),
            'workex': workex_map.get(workex, 0),
            'etest_p': float(etest_p),
            'specialisation': specialisation_map.get(specialisation, 0),
            'mba_p': float(mba_p)
        }])

        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

        is_placed = str(pred) in ['1', 'Placed', 'placed', 'True']

        if is_placed:
            st.success("🎉 **Result: Student is Likely to be PLACED!**")
        else:
            st.warning("⚠️ **Result: Student is NOT PLACED**")

        if proba is not None:
            confidence = np.max(proba) * 100
            st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
