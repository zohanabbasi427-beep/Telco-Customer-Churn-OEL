
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration & Title
st.set_page_config(page_title="Telco Customer Analytics", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction & Segmentation System")
st.markdown("### IQRA University — Introduction to Machine Learning Lab (AIC-221L)")
st.write("Enter customer details below to predict churn probability and identify their behavioral segment.")

# 2. Models aur Transformers Load Karna
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("Logistic_Regression.pkl") # Best performing baseline
        kmeans = joblib.load("KMeans_Clustering.pkl")
        scaler = joblib.load("Standard_Scaler.pkl")
        return model, kmeans, scaler
    except:
        # Agar folder ke andar files hain to local path check karein
        import os
        path = "oel_final_models/" if os.path.exists("oel_final_models/") else ""
        model = joblib.load(f"{path}Logistic_Regression.pkl")
        kmeans = joblib.load(f"{path}KMeans_Clustering.pkl")
        scaler = joblib.load(f"{path}Standard_Scaler.pkl")
        return model, kmeans, scaler

model, kmeans, scaler = load_assets()

# 3. User Input Layout (UI Sections)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics & Contract")
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    st.subheader("📞 Services Subscribed")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    st.subheader("💳 Financials & History")
    tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=780.0)

# 4. Input Data Preprocessing Pipeline
if st.button("🚀 Analyze Customer Status", type="primary"):
    
    # Text Inputs ko Numeric Binary/Dummy formats mein convert karna (as per Training Structure)
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_encoded = 1 if phone_service == "Yes" else 0
    paperless_encoded = 1 if paperless == "Yes" else 0
    
    # Internet Service Dummies
    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0
    
    # Online Security Dummies
    security_no_internet = 1 if online_security == "No internet service" else 0
    security_yes = 1 if online_security == "Yes" else 0
    
    # Contract Dummies
    contract_one = 1 if contract == "One year" else 0
    contract_two = 1 if contract == "Two year" else 0
    
    # Numerical Features Scale karna
    scaled_nums = scaler.transform([[tenure, monthly_charges, total_charges]])
    s_tenure, s_monthly, s_total = scaled_nums[0][0], scaled_nums[0][1], scaled_nums[0][2]
    
    # Final Feature Array Structure (Must match X_train columns exactly)
    # Model variables structure:
    # [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling, MonthlyCharges, TotalCharges, ...dummies]
    features = [
        gender_encoded, 0, partner_encoded, dependents_encoded, s_tenure, phone_encoded, 
        paperless_encoded, s_monthly, s_total, internet_fiber, internet_no,
        security_no_internet, security_yes, contract_one, contract_two
    ]
    
    # 5. Model Predictions
    prediction = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1] * 100
    
    # Unsupervised Clustering Prediction
    cluster_pred = kmeans.predict([[s_tenure, s_monthly, s_total]])[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk!** Probability: {prob:.2f}%")
            st.write("This customer is highly likely to leave the services.")
        else:
            st.success(f"✅ **Safe Customer.** Churn Probability: {prob:.2f}%")
            st.write("This customer shows high loyalty patterns.")
            
    with res_col2:
        st.subheader("🎯 Customer Segment (Clustering)")
        st.info(f"📁 **Assigned to Cluster #{cluster_pred}**")
        if cluster_pred == 0:
            st.write("**Profile:** New/Short-term customers with low packages.")
        elif cluster_pred == 1:
            st.write("**Profile:** Long-term, highly loyal premium service users.")
        elif cluster_pred == 2:
            st.write("**Profile:** High monthly payers but moderate tenure risk.")
        else:
            st.write("**Profile:** Standard internet bundle subscribers.")
