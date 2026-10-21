# 🏦 Loan Default Risk Predictor

An end-to-end machine learning system that predicts the probability of a loan applicant 
defaulting, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset.

🚀 **Live Demo:** [huggingface.co/spaces/amirrezaww/loan-default-risk-predictor](https://huggingface.co/spaces/amirrezaww/loan-default-risk-predictor)

---

## 🎯 Project Overview

Credit default risk is one of the most critical problems in financial services. This project 
builds a full production-style ML pipeline — from raw data to a deployed, interactive web 
application — that assesses the likelihood of a loan applicant defaulting based on their 
personal, financial, and employment profile.

---

## 🏗️ Architecture

```
Raw Data (Kaggle)
      ↓
EDA & Feature Engineering (Jupyter)
      ↓
SMOTENC Oversampling + XGBoost Model
      ↓
FastAPI Backend (/predict endpoint)
      ↓
Streamlit Frontend (Interactive UI)
      ↓
Deployed on Hugging Face Spaces
```

---

## 📊 Key Results

| Metric | Logistic Regression | XGBoost (Baseline) | XGBoost (Tuned) |
|---|---|---|---|
| AUC-ROC | 0.60 | 0.74 | **0.75** |
| Default Recall | 0.50 | 0.44 | 0.40 |
| Default Precision | 0.11 | 0.21 | **0.24** |
| Default F1 | 0.17 | 0.28 | **0.30** |

> Hyperparameter tuning via RandomizedSearchCV with 3-fold Stratified CV,  
> tested across 15 parameter combinations. SMOTENC applied inside a Pipeline to prevent 
> data leakage across folds.

---

## 🔍 Key Findings

- **EXT_SOURCE_2 & EXT_SOURCE_3** (external credit scores) were the strongest predictors 
  of default — low scores strongly indicated default risk
- **DAYS_EMPLOYED anomaly** — 55,374 applicants (18%) had a placeholder value of 365,243 
  days, indicating unemployed/pensioner status. Engineering a flag for this improved model performance
- **Income type matters** — Maternity leave (40%) and Unemployed (36%) applicants had 
  dramatically higher default rates than Pensioners (5.4%) and State servants (5.8%)
- **Education inversely correlated with default** — Lower secondary education: 10.9% 
  default rate vs Academic degree: 1.8%
- **Gender gap** — Male applicants defaulted at 10.1% vs 7.0% for female applicants
- **Class imbalance handled with SMOTENC** — minority class oversampled from 19,860 to 
  226,148 synthetic samples on training set only to prevent data leakage

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data & EDA | Python, Pandas, Seaborn, Matplotlib |
| Modelling | XGBoost, Scikit-learn, Imbalanced-learn |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Deployment | Hugging Face Spaces |

---

## 📁 Project Structure

```
Loan_default_risk_predictor/
├── data/
│   ├── raw/                  # Raw Kaggle data (not tracked)
│   └── processed/            # Cleaned data & plots
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory data analysis
│   ├── 02_modelling.ipynb                # Model training & evaluation
│   └── 03_hyperparameter_tuning.ipynb    # CV + RandomizedSearchCV
├── src/                      # Utility modules
├── api/
│   └── main.py               # FastAPI application
├── app/
│   └── streamlit_app.py      # Streamlit frontend
├── models/
│   ├── xgb_model.json        # Trained XGBoost model (not tracked)
│   └── artifacts.pkl         # Encoder, threshold, feature cols (not tracked)
├── space/                    # Hugging Face Space source (tuned model + Streamlit app)
├── .github/workflows/
│   └── deploy-space.yml      # Deploys space/ to Hugging Face on push to main
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/amirrezaww/Loan_default_risk_predictor.git
cd Loan_default_risk_predictor
```

**2. Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download data**

Download `application_train.csv` from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) 
and place it in `data/raw/`.

**4. Run notebooks**

Run `notebooks/01_eda.ipynb` then `notebooks/02_modelling.ipynb` to generate the model.

**5. Start the API**
```bash
uvicorn api.main:app --reload
```

**6. Start the UI**
```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Deployment

The live demo is deployed from the `space/` folder, which holds the Streamlit app,
the tuned model and the Space metadata. A GitHub Action (`.github/workflows/deploy-space.yml`)
uploads that folder to the Hugging Face Space whenever `space/` changes on `main`.

To enable it, add a Hugging Face access token with write permission as a repository secret
named `HF_TOKEN` (GitHub: Settings, Secrets and variables, Actions). To refresh the model,
copy the new files from `models/` into `space/` and push.

---

## 📦 Requirements

See `requirements.txt` for full dependencies.

---

## 👤 Author

Amirreza — [GitHub](https://github.com/amirrezaww)