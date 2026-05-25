import streamlit as st
import pandas as pd
import joblib as jb
model = jb.load('svm.pkl')
scalar = jb.load('scalar.pkl')
cols = jb.load('cols.pkl')   
# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details to predict the likelihood of heart disease.")

# -----------------------------
# Input Fields
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=25)

resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)

cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["TA", "ATA", "NAP", "ASY"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):

    # Create dataframe (same format as training data)
    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "MaxHR": [max_hr],
        "Oldpeak": [oldpeak],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingECG": [resting_ecg],
        "ExerciseAngina": [exercise_angina],
        "ST_Slope": [st_slope]
    })
    # Convert categorical variables
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(columns=cols, fill_value=0)
    for col in cols:
        if col not in input_data.columns:
            input_data[0]= 0
    input_data = input_data[cols]
    input_data_scale = scalar.fit_transform(input_data)
    # -----------------------------
    # 🔴 CONNECT TO YOUR BACKEND HERE
    # -----------------------------
    
    # Example 1: If you have a function
    prediction = model.predict(input_data_scale)[0]

    # Example 2: If using API
    # import requests
    # prediction = requests.post("http://127.0.0.1:5000/predict", json=input_data.to_dict()).json()

    # TEMP (dummy output)
    prediction = [1]  # replace this

    # -----------------------------
    # Output
    # -----------------------------
    if prediction[0] == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")