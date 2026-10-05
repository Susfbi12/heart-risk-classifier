"""
Heart Disease Risk Classifier - web app
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Classifier", page_icon="❤️", layout="centered")

@st.cache_resource
def load():
    return joblib.load("model.joblib")

bundle = load()
pipe, LOW, HIGH, COLS = bundle["pipeline"], bundle["low"], bundle["high"], bundle["columns"]

def risk_label(p):
    if p < LOW:
        return "Low Risk", "success"
    if p > HIGH:
        return "High Risk", "error"
    return "Moderate Risk", "warning"

st.title("❤️ Heart Disease Risk Classifier")
st.caption("College PBL prototype - Logistic Regression on the UCI/Kaggle Heart Disease dataset. "
           "Not a medical device; for learning purposes only.")

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 20, 100, 55)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female")
        cp = st.selectbox("Chest pain type", [0, 1, 2, 3],
                          format_func=lambda v: ["0 - Typical angina", "1 - Atypical angina",
                                                 "2 - Non-anginal pain", "3 - Asymptomatic"][v])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                           format_func=lambda v: "Yes" if v else "No")
        restecg = st.selectbox("Resting ECG result", [0, 1, 2])
    with c2:
        thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina", [0, 1],
                             format_func=lambda v: "Yes" if v else "No")
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
        ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia result", [0, 1, 2, 3])
    go = st.form_submit_button("Predict risk", use_container_width=True)

if go:
    row = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]], columns=COLS)
    p = float(pipe.predict_proba(row)[0, 1])
    label, kind = risk_label(p)
    getattr(st, kind)(f"**{label}**  -  predicted disease probability: {p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Thresholds: below {LOW} = Low, above {HIGH} = High, in between = Moderate "
               "(flag for closer follow-up).")

with st.expander("Model performance (from training)"):
    st.image("confusion_matrix.png", caption="Confusion matrix")
    st.image("correlation_heatmap.png", caption="Feature correlation heatmap")
