# ============================================
# SMARTCHURN AI - STREAMLIT WEB APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="SmartChurn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD ML MODELS
# ============================================

@st.cache_resource
def load_models():
    model    = joblib.load('../models/xgboost_churn_model.pkl')
    scaler   = joblib.load('../models/scaler.pkl')
    features = joblib.load('../models/feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_models()

# ============================================
# SETUP GROQ AI
# ============================================

GROQ_API_KEY = "YOUR-API-KEY"
client = Groq(api_key=GROQ_API_KEY)

# ============================================
# AI ANALYST FUNCTION
# ============================================

def analyze_customer(customer_data, churn_probability):
    prompt = f"""
    You are an expert customer retention analyst for a telecom company.
    
    Customer Details:
    - Tenure: {customer_data['tenure']} months
    - Monthly Charges: ${customer_data['MonthlyCharges']}
    - Contract Type: {customer_data['Contract']}
    - Internet Service: {customer_data['InternetService']}
    - Tech Support: {customer_data['TechSupport']}
    - Payment Method: {customer_data['PaymentMethod']}
    - Churn Probability: {churn_probability:.1f}%
    
    Please provide:
    1. Top 3 reasons why this customer might churn
    2. Best retention offer for this customer
    3. Priority level (High/Medium/Low)
    
    Keep response short and practical.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ============================================
# MAIN HEADER
# ============================================

st.title("🤖 SmartChurn AI")
st.markdown("### Predict • Explain • Retain")
st.markdown("---")

# ============================================
# SIDEBAR - CUSTOMER INPUT
# ============================================

st.sidebar.title("👤 Customer Details")
st.sidebar.markdown("Enter customer information below:")

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=0,
    max_value=120,
    value=65
)

total_charges = monthly_charges * tenure

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)",
     "Credit card (automatic)"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Has Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Has Dependents",
    ["Yes", "No"]
)

# ============================================
# PREDICT BUTTON
# ============================================

st.sidebar.markdown("---")
predict_button = st.sidebar.button(
    "🔮 Predict Churn",
    use_container_width=True
)

# ============================================
# MAIN PAGE - PREDICTION RESULTS
# ============================================

if predict_button:

    # ---- Prepare Input Data ----
    input_data = {
        'gender'              : 0,
        'SeniorCitizen'       : 1 if senior_citizen == "Yes" else 0,
        'Partner'             : 1 if partner == "Yes" else 0,
        'Dependents'          : 1 if dependents == "Yes" else 0,
        'tenure'              : tenure,
        'PhoneService'        : 1,
        'MultipleLines'       : 0,
        'InternetService'     : ["DSL","Fiber optic","No"].index(internet_service),
        'OnlineSecurity'      : 0,
        'OnlineBackup'        : 0,
        'DeviceProtection'    : 0,
        'TechSupport'         : 1 if tech_support == "Yes" else 0,
        'StreamingTV'         : 0,
        'StreamingMovies'     : 0,
        'Contract'            : ["Month-to-month","One year","Two year"].index(contract),
        'PaperlessBilling'    : 1,
        'PaymentMethod'       : ["Electronic check","Mailed check",
                                  "Bank transfer (automatic)",
                                  "Credit card (automatic)"].index(payment_method),
        'MonthlyCharges'      : monthly_charges,
        'TotalCharges'        : total_charges,
        'tenure_group'        : 0,
        'avg_monthly_spend'   : monthly_charges / (tenure + 1),
        'has_premium_services': 0,
        'is_high_value'       : 1 if (tenure > 24 and monthly_charges > 50) else 0,
        'num_services'        : 3
    }

    # ---- Convert to DataFrame ----
    input_df = pd.DataFrame([input_data])

    # ---- Scale the Data ----
    input_scaled = scaler.transform(input_df[feature_names])

    # ---- Make Prediction ----
    churn_probability = model.predict_proba(input_scaled)[0][1] * 100
    churn_prediction  = "YES 🔴" if churn_probability > 50 else "NO 🟢"

    # ============================================
    # DISPLAY RESULTS
    # ============================================

    st.markdown("## 📊 Prediction Results")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🔮 Churn Prediction",
            value=churn_prediction
        )

    with col2:
        st.metric(
            label="📊 Churn Probability",
            value=f"{churn_probability:.1f}%"
        )

    with col3:
        risk_level = "🔴 HIGH"   if churn_probability > 70 else \
                     "🟡 MEDIUM" if churn_probability > 40 else \
                     "🟢 LOW"
        st.metric(
            label="⚠️ Risk Level",
            value=risk_level
        )

    st.markdown("---")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### 📋 Customer Summary")
        summary_data = {
            "Feature" : ["Tenure", "Monthly Charges",
                         "Contract", "Internet Service",
                         "Tech Support", "Payment Method"],
            "Value"   : [f"{tenure} months",
                         f"${monthly_charges}",
                         contract, internet_service,
                         tech_support, payment_method]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True,
                     use_container_width=True)

    with right_col:
        st.markdown("### 🤖 AI Retention Analysis")
        with st.spinner("🤖 AI is analyzing customer..."):
            try:
                customer_data = {
                    'tenure'         : tenure,
                    'MonthlyCharges' : monthly_charges,
                    'TotalCharges'   : total_charges,
                    'Contract'       : contract,
                    'InternetService': internet_service,
                    'TechSupport'    : tech_support,
                    'PaymentMethod'  : payment_method
                }
                ai_analysis = analyze_customer(
                    customer_data,
                    churn_probability
                )
                st.markdown(ai_analysis)
            except Exception as e:
                st.error("⚠️ AI analysis unavailable. Check API key.")

    # ============================================
    # SHAP EXPLANATION
    # ============================================

    st.markdown("---")
    st.markdown("## 🔍 Why This Prediction?")
    st.markdown("*(SHAP Explainability — What factors influenced this prediction)*")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(
            shap_values,
            input_df[feature_names],
            plot_type="bar",
            show=False
        )
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning("⚠️ SHAP chart unavailable.")

    # ============================================
    # FOOTER
    # ============================================

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            🤖 SmartChurn AI | Built by Arige Bharath Kumar | 
            XGBoost + SHAP + Groq LLM
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # ============================================
    # WELCOME SCREEN
    # ============================================

    st.markdown("## 👋 Welcome to SmartChurn AI!")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🔮 Predict\nEnter customer details in the sidebar and click Predict Churn")

    with col2:
        st.success("### 🔍 Explain\nSHAP analysis shows exactly why the model made its prediction")

    with col3:
        st.warning("### 💡 Retain\nGroq AI generates personalized retention offers instantly")

    st.markdown("---")
    st.markdown("### 🚀 How to Use:")
    st.markdown("""
    1. 👈 Fill in customer details in the **left sidebar**
    2. 🔮 Click **Predict Churn** button
    3. 📊 View prediction, probability and risk level
    4. 🤖 Read AI generated retention recommendation
    5. 🔍 Understand prediction with SHAP chart
    """)