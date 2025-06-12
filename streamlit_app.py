import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\DELL\Downloads\model (2).save")
    scaler = joblib.load(r"C:\Users\DELL\Downloads\scaler (2).save")
    return model, scaler

model, scaler = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it's fraudulent.")

# List of feature names (excluding 'Class')
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Create input fields dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.5f")
    user_input.append(value)

# Predict button
if st.button("Predict Fraud"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Probability of Fraud: {probability:.2f})")
