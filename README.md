# 🤖 SmartChurn AI
### Predict • Explain • Retain

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq_LLM-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

> An end-to-end AI-powered customer churn prediction system for telecom companies — combining Machine Learning, Explainable AI, NLP Sentiment Analysis, and LLM-powered retention recommendations.

---

## 📌 Project Overview

**SmartChurn AI** is a complete data science project that predicts which customers are likely to churn, explains why, and suggests personalized retention strategies using Generative AI.

Built on the **Telco Customer Churn dataset** (7,043 customers), this project demonstrates a full ML pipeline from raw data to a deployed web application.

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| Model | XGBoost Classifier |
| Accuracy | 74.59% |
| **Recall** | **79.41%** |
| Precision | 51.38% |
| F1 Score | 62.39% |

> ⭐ **Recall of 79%** means the model successfully identifies 79 out of every 100 customers who will churn — enabling proactive retention before revenue is lost.

---

## 🏗️ Project Architecture

```
SmartChurn AI
├── 📊 Week 1 — EDA & Feature Engineering
├── 🤖 Week 2 — XGBoost Model + SHAP Explainability
├── 💬 Week 3 — NLP Sentiment Analysis
├── 🧠 Week 4 — LLM Integration (Groq AI)
└── 🌐 Week 5 — Streamlit Web App
```

---

## ✨ Features

- 🔮 **Churn Prediction** — XGBoost model predicts churn probability for any customer
- 🔍 **Explainable AI** — SHAP values explain exactly why a prediction was made
- 💬 **Sentiment Analysis** — TextBlob NLP analyzes customer review sentiment
- 🤖 **AI Retention Analyst** — Groq LLM (LLaMA 3.1) generates personalized retention offers
- 🌐 **Web App** — Interactive Streamlit dashboard for business users

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| ML Model | XGBoost, Scikit-learn |
| Explainability | SHAP |
| NLP | TextBlob |
| LLM | Groq API (LLaMA 3.1 8B) |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
smartchurn-ai/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   ├── churn_clean.csv                          # Cleaned + engineered features
│   └── customer_sentiment.csv                  # NLP sentiment results
│
├── notebook/
│   ├── 1_EDA.ipynb                             # Exploratory Data Analysis
│   ├── 2_ML Models.ipynb                       # XGBoost + SHAP
│   ├── 3_NLP.ipynb                             # Sentiment Analysis
│   └── 4_LLM.ipynb                             # Groq LLM Integration
│
├── models/
│   ├── xgboost_churn_model.pkl                 # Trained XGBoost model
│   ├── scaler.pkl                              # StandardScaler
│   ├── feature_names.pkl                       # Feature names list
│   └── config.json                             # App configuration
│
├── app/
│   └── app.py                                  # Streamlit web app
│
├── dashboard/                                  # Power BI dashboard (coming soon)
└── sql/                                        # SQL analysis (coming soon)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/bharathkumararige/SMART_CHURN_AI.git
cd SMART_CHURN_AI
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit textblob groq joblib
```

### 3. Add Your Groq API Key
Open `app/app.py` and replace:
```python
GROQ_API_KEY = "your-groq-api-key-here"
```
Get your free API key at: https://console.groq.com

### 4. Run the Web App
```bash
cd app
streamlit run app.py
```

### 5. Open in Browser
```
http://localhost:8501
```

---

## 📊 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No)
- **Class Distribution:** 73.5% stayed, 26.5% churned

---

## 🔬 Feature Engineering

5 new features created from raw data:

| Feature | Description |
|---|---|
| `tenure_group` | Customer lifecycle stage (New/Developing/Mature/Loyal) |
| `avg_monthly_spend` | Average spend per month |
| `has_premium_services` | Binary flag for premium service usage |
| `is_high_value` | High tenure + high charges customer |
| `num_services` | Total number of active services |

---

## 🧠 How the App Works

1. 👈 Enter customer details in the sidebar
2. 🔮 Click **Predict Churn** button
3. 📊 View churn probability and risk level
4. 🤖 Read AI-generated retention recommendation
5. 🔍 Understand prediction with SHAP chart

---

## 💡 Business Impact

> *"By identifying 79% of churning customers before they leave, a telecom company with 1 million customers and $50 average monthly revenue could potentially save **$39.5 million per year** in prevented churn."*

---

## 📸 App Screenshot

```
🤖 SmartChurn AI — Predict • Explain • Retain

┌─────────────────┬──────────────────┬─────────────────┐
│ 🔮 Prediction   │ 📊 Probability   │ ⚠️ Risk Level   │
│ YES 🔴          │ 86.1%            │ 🔴 HIGH         │
└─────────────────┴──────────────────┴─────────────────┘

📋 Customer Summary    │  🤖 AI Retention Analysis
───────────────────────│──────────────────────────────
Tenure    │ 2 months   │  1. Short tenure risk
Contract  │ M-t-M      │  2. High monthly charges
Internet  │ Fiber       │  Offer: 20% discount +
Tech Sup  │ No         │  3 months free tech support
```

---

## 👨‍💻 Author

**Arige Bharath Kumar**
- 🎓 B.Tech CSE (Data Science) — Graduating July 2026
- 📧 [arigebharathkumar@gmail.com](mailto:arigebharathkumar@gmail.com)
- 🔗 [LinkedIn](https://linkedin.com/in/arigebharath)
- 🐙 [GitHub](https://github.com/bharathkumararige)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">⭐ <b>If you found this project helpful, please give it a star!</b> ⭐</p>
