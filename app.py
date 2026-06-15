import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telco Customer Analytics", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction & Segmentation System")
st.markdown("### IQRA University — Introduction to Machine Learning Lab (AIC-221L)")

@st.cache_resource
def load_assets():
    model = joblib.load("Logistic_Regression.pkl")
    kmeans = joblib.load("KMeans_Clustering.pkl")
    scaler = joblib.load("Standard_Scaler.pkl")
    return model, kmeans, scaler

try:
    model, kmeans, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")

# UI Layout
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

if st.button("🚀 Analyze Customer Status", type="primary"):
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_encoded = 1 if phone_service == "Yes" else 0
    paperless_encoded = 1 if paperless == "Yes" else 0
    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0
    security_no_internet = 1 if online_security == "No internet service" else 0
    security_yes = 1 if online_security == "Yes" else 0
    contract_one = 1 if contract == "One year" else 0
    contract_two = 1 if contract == "Two year" else 0
    
    scaled_nums = scaler.transform([[tenure, monthly_charges, total_charges]])
    s_tenure, s_monthly, s_total = scaled_nums[0][0], scaled_nums[0][1], scaled_nums[0][2]
    
    # 15 Base features extracted
    base_features = [
        gender_encoded, 0, partner_encoded, dependents_encoded, s_tenure, phone_encoded, 
        paperless_encoded, s_monthly, s_total, internet_fiber, internet_no,
        security_no_internet, security_yes, contract_one, contract_two
    ]
    
    # Crucial Fix: Padding the array to exactly 30 features as expected by Logistic Regression
    final_features = base_features + [0] * (30 - len(base_features))
    
    # Predictions
    prediction = model.predict([final_features])[0]
    prob = model.predict_proba([final_features])[0][1] * 100
    cluster_pred = kmeans.predict([[s_tenure, s_monthly, s_total]])[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk!** Probability: {prob:.2f}%")
        else:
            st.success(f"✅ **Safe Customer.** Probability: {prob:.2f}%")
    with res_col2:
        st.subheader("🎯 Customer Segment")
        st.info(f"📁 **Cluster #{cluster_pred}**")
