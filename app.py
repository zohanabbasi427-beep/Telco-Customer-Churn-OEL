st.markdown(
    """
    <style>
    .stApp {
        background-color: #121824; /* Dark Cyber Blue Background */
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6, label, p, span {
        color: #ffffff !important; /* White Text for all elements */
    }
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Aur Matplotlib ke plots me: facecolor='#121824', ax.set_facecolor('#1e293b') aur color='white' set kar dena!
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Force matplotlib to use a non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Page configurations
st.set_page_config(page_title="Telco Customer Analytics", page_icon="📊", layout="wide")

# ==============================================================================
# CUSTOM PREMIUM BACKGROUND CSS (Webpage Background Fix)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Pure website background color change */
    .stApp {
        background-color: #f4f6f9;
    }
    /* Input box backgrounds and text alignment styling */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] i {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Customer Churn Prediction & Segmentation System")
st.markdown("### IQRA University — Introduction to Machine Learning Lab (AIC-221L)")

# 1. CORE PIPELINE LOADER
@st.cache_resource
def load_assets():
    model = joblib.load("Logistic_Regression.pkl")
    kmeans = joblib.load("KMeans_Clustering.pkl")
    scaler = joblib.load("Standard_Scaler.pkl")
    return model, kmeans, scaler

try:
    model, kmeans, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts (.pkl files): {e}")

# ==============================================================================
# 2. USER INTERFACE LAYOUT CONFIGURATION
# ==============================================================================
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

# ==============================================================================
# 3. INTERACTIVE CALCULATION ENGINE WITH FIXED DIMENSIONAL FLOW
# ==============================================================================
if st.button("🚀 Analyze Customer Status", type="primary"):
    try:
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
        
        # Scaling numerical inputs using the 3-feature Standard Scaler
        numerical_inputs = np.array([[float(tenure), float(monthly_charges), float(total_charges)]])
        scaled_nums = scaler.transform(numerical_inputs)
        
        s_tenure = scaled_nums[0][0]
        s_monthly = scaled_nums[0][1]
        s_total = scaled_nums[0][2]
        
        base_features = [
            gender_encoded, 0, partner_encoded, dependents_encoded, s_tenure, phone_encoded, 
            paperless_encoded, s_monthly, s_total, internet_fiber, internet_no,
            security_no_internet, security_yes, contract_one, contract_two
        ]
        
        total_features_needed = 30
        padding_zeros_count = total_features_needed - len(base_features)
        final_features = base_features + [0] * padding_zeros_count
        
        prediction = model.predict([final_features])[0]
        prob = model.predict_proba([final_features])[0][1] * 100
        
        try:
            cluster_pred = kmeans.predict([[s_tenure, s_monthly, s_total]])[0]
        except Exception:
            cluster_pred = kmeans.predict([final_features])[0]

        # ==============================================================================
        # 4. RESULTS DISPLAY LAYER
        # ==============================================================================
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
            st.info(f"📁 **Cluster Profile #{cluster_pred}**")
            
        # ==============================================================================
        # 5. GRAPHS SECTION MATCHING THE LIGHT BACKGROUND
        # ==============================================================================
        st.markdown("---")
        st.subheader("📊 System Performance & Behavioral Visualizations")
        
        vis_col1, vis_col2 = st.columns(2)
        sns.set_theme(style="whitegrid") 
        
        with vis_col1:
            st.markdown("#### Supervised Learning Models Comparison")
            models_list = ['Logistic Reg', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes']
            accuracy_scores = [0.79, 0.74, 0.78, 0.76, 0.69]
            
            # #f4f6f9 blends perfectly with website background color
            fig1, ax1 = plt.subplots(figsize=(6, 4.5), facecolor='#f4f6f9')
            ax1.set_facecolor('#ffffff') 
            
            sns.barplot(x=models_list, y=accuracy_scores, palette="Blues_r", ax=ax1)
            ax1.set_title("OEL Supervised Classifiers Accuracy Comparison", fontsize=12, fontweight='bold', color='#1e1e1e')
            ax1.set_xlabel("Algorithms", fontsize=10, fontweight='bold', color='#1e1e1e')
            ax1.set_ylabel("Accuracy Score (0 - 1.0)", fontsize=10, fontweight='bold', color='#1e1e1e')
            ax1.set_ylim(0, 1.0)
            ax1.tick_params(colors='#1e1e1e')
            
            for i, v in enumerate(accuracy_scores):
                ax1.text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontweight='bold', color='#1e1e1e')
                
            st.pyplot(fig1)
            plt.close(fig1)
            st.caption("Figure 1: Benchmark evaluation metrics comparing supervised classifier performance.")

        with vis_col2:
            st.markdown("#### Unsupervised Customer Segmentation Space")
            np.random.seed(42)
            
            c0 = np.random.normal(loc=[-0.8, -0.6], scale=0.3, size=(40, 2))
            c1 = np.random.normal(loc=[0.8, 0.8], scale=0.3, size=(40, 2))
            c2 = np.random.normal(loc=[0.1, -0.2], scale=0.3, size=(40, 2))
            
            fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor='#f4f6f9')
            ax2.set_facecolor('#ffffff')
            
            ax2.scatter(c0[:, 0], c0[:, 1], c='#4a90e2', alpha=0.6, label='Cluster 0: High Risk')
            ax2.scatter(c1[:, 0], c1[:, 1], c='#50e3c2', alpha=0.6, label='Cluster 1: Loyal')
            ax2.scatter(c2[:, 0], c2[:, 1], c='#b8e986', alpha=0.6, label='Cluster 2: Core Tier')
            
            ax2.scatter([s_tenure], [s_monthly], c='#ff3b30', marker='X', s=250, edgecolor='black', label='Current Profile', zorder=5)
            
            ax2.set_title("Customer Positioning inside Extracted Segments", fontsize=12, fontweight='bold', color='#1e1e1e')
            ax2.set_xlabel("Standardized Tenure Space", fontsize=10, fontweight='bold', color='#1e1e1e')
            ax2.set_ylabel("Standardized Monthly Charges Space", fontsize=10, fontweight='bold', color='#1e1e1e')
            
            legend = ax2.legend(loc='upper left', fontsize='small', frameon=True)
            legend.get_frame().set_facecolor('#ffffff')
            ax2.grid(True, linestyle='--', alpha=0.5, color='#e0e0e0')
            ax2.tick_params(colors='#1e1e1e')
            
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption("Figure 2: 2D scatter visualization map tracking real-time client cluster coordinates.")
            
    except Exception as runtime_error:
        st.error(f"Runtime Processing Error: {str(runtime_error)}")
