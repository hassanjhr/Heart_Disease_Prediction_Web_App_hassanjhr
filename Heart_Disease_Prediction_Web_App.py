# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:24:18 2024

@author: Precision
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model (relative path)
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Function for Heart Disease Prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return '🎉 The person is **not at risk of heart disease**.', 'success'
    else:
        return '⚠️ The person is **at risk of heart disease**.', 'danger'

# Main function for Streamlit app
def main():
    # Streamlit Page Configurations
    st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")
    
    # Custom CSS for styling and animation
    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 2.8rem;
                
            }
            .result-success {
                background-color: #D4EDDA;
                color: #155724;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.2rem;
                animation: fadeIn 2s;
            }
            .result-danger {
                background-color: #F8D7DA;
                color: #721C24;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.2rem;
                animation: blink 1.5s infinite;
            }
            @keyframes blink {
                0% {opacity: 1;}
                50% {opacity: 0.5;}
                100% {opacity: 1;}
            }
            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }
        </style>
    """, unsafe_allow_html=True)
    
    # App Title
    st.markdown('<div class="title">❤️ Heart Disease Prediction Web App</div>', unsafe_allow_html=True)

    # Sidebar for Inputs
    st.sidebar.title("🔧 Input Features")
    st.sidebar.write("Please provide the following details:")

    # Input fields
    age = st.sidebar.number_input('Age', min_value=0, step=1)
    sex = st.sidebar.number_input('Sex (0 = Female, 1 = Male)', min_value=0, max_value=1, step=1)
    cp = st.sidebar.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, step=1)
    trestbps = st.sidebar.number_input('Resting Blood Pressure', min_value=0)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dL)', min_value=0)
    fbs = st.sidebar.number_input('Fasting Blood Sugar (>120 mg/dL: 1, else 0)', min_value=0, max_value=1, step=1)
    restecg = st.sidebar.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2, step=1)
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0)
    exang = st.sidebar.number_input('Exercise Induced Angina (1 = Yes, 0 = No)', min_value=0, max_value=1, step=1)
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, format="%.2f")
    slope = st.sidebar.number_input('Slope of the Peak Exercise ST Segment (0-2)', min_value=0, max_value=2, step=1)
    ca = st.sidebar.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4, step=1)
    thal = st.sidebar.number_input('Thalassemia (0-3)', min_value=0, max_value=3, step=1)
    
    # Code for Prediction
    diagnosis = ''
    alert_type = 'success'

    # Main Section for Results
    st.write("## Prediction Results")
    if st.sidebar.button('🧪 Get Heart Disease Test Result'):
        # Run the prediction
        diagnosis, alert_type = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, 
                                                          thalach, exang, oldpeak, slope, ca, thal])
        
        # Display results attractively
        if alert_type == 'success':
            st.markdown(f'<div class="result-success">{diagnosis}</div>', unsafe_allow_html=True)
            st.balloons()  # Show balloons for "no risk" cases
        else:
            st.markdown(f'<div class="result-danger">{diagnosis}</div>', unsafe_allow_html=True)
            st.error("⚠️ **Warning: High Risk Detected! Please consult a cardiologist immediately.**")

if __name__ == '__main__':
    main()
