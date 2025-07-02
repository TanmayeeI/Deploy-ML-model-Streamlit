import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('heart_model.pkl')

st.title("Heart Disease Risk Prediction App")
st.write("Predict heart disease risk based on patient data.")

st.header("Enter Patient Data")

age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1, value=0, step=1)
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0, step=1)
trestbps = st.number_input("Resting BP", min_value=80, max_value=200, value=120, step=1)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200, step=1)
fbs = st.number_input("Fasting Blood Sugar > 120 (0 or 1)", min_value=0, max_value=1, value=0, step=1)
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0, step=1)
thalach = st.number_input("Max Heart Rate", min_value=70, max_value=210, value=150, step=1)
exang = st.number_input("Exercise Induced Angina (0 or 1)", min_value=0, max_value=1, value=0, step=1)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=0, step=1)
ca = st.number_input("Major vessels colored (0-3)", min_value=0, max_value=3, value=0, step=1)
thal = st.number_input("Thal (1-3)", min_value=1, max_value=3, value=2, step=1)

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    risk = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    st.subheader(f"Prediction: {risk}")

    proba = model.predict_proba(input_data)[0]
    st.write(f"Probability of No Disease: {proba[0]:.2f}")
    st.write(f"Probability of Disease: {proba[1]:.2f}")

st.header("Model Insights")
if st.checkbox("Show Feature Importances"):
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                     'ca', 'thal']
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)

    st.bar_chart(fi_df.set_index('feature'))

    fig, ax = plt.subplots()
    ax.barh(fi_df['feature'], fi_df['importance'])
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importances from RandomForest")
    st.pyplot(fig)
