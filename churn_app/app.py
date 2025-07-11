import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models and scaler
@st.cache_resource
def load_artifacts():
    return {
        'xgb': joblib.load('./models/xgb_grid.pkl'),
        'ensemble': joblib.load('./models/ensemble.pkl'),
        'scaler': joblib.load('./models/scaler.pkl')  # Save this during model training
    }

artifacts = load_artifacts()

# UI
st.title("Customer Churn Predictor")
model_choice = st.radio("Select model:", 
                       ["XGBoost (Best Accuracy)", "Expert Ensemble"])

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        streaming = st.selectbox("Streaming Services User", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    # Additional service questions
    with st.expander("Additional Services"):
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    submitted = st.form_submit_button("Predict Churn Risk")

# Prediction
if submitted:
    # Create DataFrame with all features initialized to 0
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'StreamingUser', 'MultipleLines_No', 'MultipleLines_No phone service',
        'MultipleLines_Yes', 'InternetService_DSL',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service',
        'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No internet service', 'TechSupport_Yes'
    ]
    
    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    # Map categorical features
    input_data['gender'] = 1 if gender == "Male" else 0
    input_data['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    input_data['Partner'] = 1 if partner == "Yes" else 0
    input_data['Dependents'] = 1 if dependents == "Yes" else 0
    input_data['PhoneService'] = 1 if phone_service == "Yes" else 0
    input_data['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
    input_data['StreamingUser'] = 1 if streaming == "Yes" else 0
    
    # Handle multiple categorical features
    input_data[f'MultipleLines_No'] = 1 if multiple_lines == "No" else 0
    input_data[f'MultipleLines_No phone service'] = 1 if multiple_lines == "No phone service" else 0
    input_data[f'MultipleLines_Yes'] = 1 if multiple_lines == "Yes" else 0
    
    # Contract type
    input_data[f'Contract_{contract.replace(" ", " ")}'] = 1
    
    # Payment method
    input_data[f'PaymentMethod_{payment_method}'] = 1
    
    # Internet services
    input_data[f'InternetService_{internet_service.replace(" ", " ")}'] = 1
    input_data[f'OnlineSecurity_{online_security.replace(" ", " ")}'] = 1
    input_data[f'OnlineBackup_{online_backup.replace(" ", " ")}'] = 1
    input_data[f'DeviceProtection_{device_protection.replace(" ", " ")}'] = 1
    input_data[f'TechSupport_{tech_support.replace(" ", " ")}'] = 1
    
    # Standardize numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_data[numerical_cols] = artifacts['scaler'].transform(input_data[numerical_cols])
    
    # Make prediction
    if model_choice == "XGBoost (Best Accuracy)":
        proba = artifacts['xgb'].predict_proba(input_data)[0][1]
    else:
        proba = artifacts['ensemble'].predict_proba(input_data)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Churn Probability", f"{proba*100:.1f}%")
    
    with col2:
        st.metric("Prediction", "Churn" if proba > 0.5 else "No Churn")
    
    st.progress(float(proba))  # Ensure it's a float between 0.0 and 1.0
    
    # Threshold explanation
    st.caption(f"Using threshold = 0.5 (adjustable in advanced settings)")
    
    # Model explanation
    if model_choice == "Expert Ensemble":
        st.info("""
        **Ensemble Model**: Combines predictions from XGBoost, LightGBM and Random Forest 
        for more robust performance. May be slower but generally more accurate.
        """)
    else:
        st.info("""
        **XGBoost Model**: Optimized for best accuracy on our test data. 
        Fast predictions with good overall performance.
        """)
    
    # Feature importance button
    # if st.button("Explain Prediction"):
    #     st.write("Feature importance analysis would go here")
        # You would add SHAP or LIME explanations here