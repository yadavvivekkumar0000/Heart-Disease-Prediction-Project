import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.caption("Educational ML demonstration only—not a medical diagnosis or substitute for clinical care.")

model=joblib.load("heart_model.joblib")

with st.form("prediction_form"):
    col1,col2=st.columns(2)
    age=col1.number_input("Age",18,100,45)
    sex=col2.selectbox("Sex",["Female","Male"])
    cp=st.selectbox("Chest pain type",["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"])
    trestbps=col1.number_input("Resting blood pressure (mm Hg)",80,250,120)
    chol=col2.number_input("Cholesterol (mg/dl)",100,600,200)
    fbs=st.selectbox("Fasting blood sugar > 120 mg/dl",["No","Yes"])
    restecg=st.selectbox("Resting ECG",["Normal","ST-T abnormality","Left ventricular hypertrophy"])
    thalach=col1.number_input("Maximum heart rate achieved",60,250,150)
    exang=col2.selectbox("Exercise-induced angina",["No","Yes"])
    oldpeak=col1.number_input("ST depression (oldpeak)",0.0,10.0,0.0,step=0.1)
    slope=st.selectbox("ST slope",["Upsloping","Flat","Downsloping"])
    ca=st.selectbox("Major vessels (0–4)",[0,1,2,3,4])
    thal=st.selectbox("Thalassemia",["Normal","Fixed defect","Reversible defect"])
    submitted=st.form_submit_button("Predict")

if submitted:
    values = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,

    "cp": [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ].index(cp) + 1,

    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,

    "restecg": [
        "Normal",
        "ST-T abnormality",
        "Left ventricular hypertrophy"
    ].index(restecg),

    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,

    "slope": [
        "Upsloping",
        "Flat",
        "Downsloping"
    ].index(slope) + 1,

    "ca": ca,

    "thal": [
        3, 6, 7
    ][
        ["Normal", "Fixed defect", "Reversible defect"].index(thal)
    ]
}
    X=pd.DataFrame([values])
    prediction=int(model.predict(X)[0])
    st.subheader("Prediction")
    if prediction==1:
        st.warning("The model predicts a potential presence of heart disease.")
    else:
        st.success("The model predicts no heart disease detected in this input.")
    st.write("This prediction is statistical and can be wrong. Seek a qualified medical professional for health concerns.")
