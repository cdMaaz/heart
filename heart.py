import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Health", page_icon="❤️")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('heart.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler
model, scaler = load_assets()

st.title("❤️ Heart Disease Prediction BY Maaz")


age = st.slider("Age", min_value=10, max_value=100, value=25)
sex = st.selectbox("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=250)
restecg = st.selectbox("Resting ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", ["yes", "no"])
oldpeak = st.number_input("ST Depression (induced by exercise relative to rest)", value=1.0)
slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])


# --- DATA PREPROCESSING ---
def preprocess_input():
    # 1. Map text back to the exact numbers the original dataset used
    thal_map = {"normal": 0, "fixed defect": 1, "reversable defect": 2}
    exang_map = {"yes": 1, "no": 0}
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    restecg_map = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
    
    # 2. Create a dictionary using the EXACT column names from your notebook before OHE
    input_dict = {
        'age': [age],
        'sex': [sex_map[sex]], 
        'cp': [cp_map[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'restecg': [restecg_map[restecg]],
        'thalach': [thalach],
        'exang': [exang_map[exang]],
        'oldpeak': [oldpeak],
        'slope': [slope_map[slope]],
        'ca': [ca],
        'thal': [thal_map[thal]]
    }
    
    # 2. Turn it into a DataFrame
    df_input = pd.DataFrame(input_dict)
    
    # 3. Scale the data
    scaled_features = scaler.transform(df_input)
    return scaled_features

# --- PREDICTION ---
if st.button("Predict Health Status"):
    try:
        input_data = preprocess_input()
        prediction = model.predict(input_data)
        
        st.divider()
        if prediction[0] == 1:
            st.error("### High Risk Detected")
            st.write("Please consult a healthcare professional.")
        else:
            st.success("### Low Risk / Healthy")
            st.write("Keep up the good lifestyle!")
        st.info("### JUST A PREDICTION")
            
    except Exception as e:
        st.error(f"Error in prediction: {e}")