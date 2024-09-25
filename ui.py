import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app title
st.title("Heart Disease Risk Assessment")

# Input fields for user data
age = st.slider("Age", min_value=0, max_value=100, step=1)
sex = st.radio("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # Assuming 0, 1, 2, 3 are the encoded values
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # 0 = False, 1 = True
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])  # Assuming 0, 1, 2 are the encoded values
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
exang = st.radio("Exercise Induced Angina", [0, 1])  # 0 = No, 1 = Yes
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])  # Assuming 0, 1, 2 are the encoded values
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3)
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])  # Assuming 0, 1, 2 are the encoded values

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale input data
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("High likelihood of heart disease.")
    else:
        st.success("Low likelihood of heart disease.")
