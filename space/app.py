import streamlit as st
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load model & artifacts ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(BASE_DIR / "xgb_model_tuned.json"))
    with open(BASE_DIR / "artifacts_tuned.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts

model, artifacts = load_model()
encoder = artifacts['encoder']
cat_cols = artifacts['cat_cols']
feature_cols = artifacts['feature_cols']
threshold = artifacts['optimal_threshold']

# ── Feature engineering ─────────────────────────────────────────────
def engineer_features(df):
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365
    df['YEARS_EMPLOYED'] = np.abs(df['DAYS_EMPLOYED']) / 365
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
    return df

COLUMN_DEFAULTS = {
    'FLAG_MOBIL': 1, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1,
    'FLAG_EMAIL': 0, 'HOUR_APPR_PROCESS_START': 12,
    'REG_REGION_NOT_LIVE_REGION': 0, 'REG_REGION_NOT_WORK_REGION': 0,
    'LIVE_REGION_NOT_WORK_REGION': 0,
    'FLAG_DOCUMENT_2': 0, 'FLAG_DOCUMENT_4': 0, 'FLAG_DOCUMENT_5': 0,
    'FLAG_DOCUMENT_6': 0, 'FLAG_DOCUMENT_7': 0, 'FLAG_DOCUMENT_8': 0,
    'FLAG_DOCUMENT_9': 0, 'FLAG_DOCUMENT_10': 0, 'FLAG_DOCUMENT_11': 0,
    'FLAG_DOCUMENT_12': 0, 'FLAG_DOCUMENT_13': 0, 'FLAG_DOCUMENT_14': 0,
    'FLAG_DOCUMENT_15': 0, 'FLAG_DOCUMENT_16': 0, 'FLAG_DOCUMENT_17': 0,
    'FLAG_DOCUMENT_18': 0, 'FLAG_DOCUMENT_19': 0, 'FLAG_DOCUMENT_20': 0,
    'FLAG_DOCUMENT_21': 0,
    'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0, 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0,
    'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0, 'AMT_REQ_CREDIT_BUREAU_MON': 0.0,
    'HAS_CAR': 0
}

def predict(payload):
    df = pd.DataFrame([payload])
    df = engineer_features(df)
    for col, val in COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
    df[cat_cols] = encoder.transform(df[cat_cols])
    df = df[feature_cols]
    prob = model.predict_proba(df)[:, 1][0]
    label = int(prob >= threshold)
    return float(prob), "HIGH RISK" if label == 1 else "LOW RISK"

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Risk Predictor")
st.markdown("Enter the applicant's details below to assess their loan default risk.")
st.divider()

# ── Input form ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", ["M", "F"])
    age_years = st.slider("Age (years)", 20, 70, 35)
    education = st.selectbox("Education Type", [
        "Secondary / secondary special", "Higher education",
        "Incomplete higher", "Lower secondary", "Academic degree"
    ])
    family_status = st.selectbox("Family Status", [
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
    ])
    cnt_children = st.number_input("Number of Children", 0, 10, 0)
    cnt_fam_members = st.number_input("Family Members", 1, 10, 2)

with col2:
    st.subheader("Employment & Income")
    income_type = st.selectbox("Income Type", [
        "Working", "Commercial associate", "Pensioner",
        "State servant", "Unemployed"
    ])
    income = st.number_input("Annual Income (£)", 25000, 1000000, 135000, step=5000)
    years_employed = st.slider("Years Employed", 0, 40, 5)
    occupation = st.selectbox("Occupation Type", [
        "Laborers", "Core staff", "Managers", "Drivers",
        "High skill tech staff", "Accountants", "Medicine staff",
        "Security staff", "Cooking staff", "Cleaning staff",
        "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
        "Secretaries", "Sales staff", "Realty agents", "HR staff", "IT staff"
    ])
    organization = st.selectbox("Organization Type", [
        "Business Entity Type 3", "School", "Government", "Religion",
        "Other", "Medicine", "Business Entity Type 2", "Self-employed",
        "Transport: type 2", "Construction", "Housing", "Kindergarten",
        "Military", "Services", "Security Ministries", "Police",
        "Business Entity Type 1", "Postal", "Agriculture", "Restaurant",
        "Culture", "Hotel", "Bank", "Insurance", "Mobile", "Telecom",
        "Legal Services", "Advertising", "Cleaning", "Realtor", "XNA"
    ])

with col3:
    st.subheader("Loan Details")
    contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    own_car = st.selectbox("Owns a Car?", ["N", "Y"])
    own_realty = st.selectbox("Owns Realty?", ["Y", "N"])
    credit = st.number_input("Loan Amount (£)", 50000, 4000000, 500000, step=10000)
    annuity = st.number_input("Monthly Annuity (£)", 5000, 250000, 25000, step=1000)
    goods_price = st.number_input("Goods Price (£)", 50000, 4000000, 450000, step=10000)
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5, step=0.01)
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5, step=0.01)
    region_rating = st.selectbox("Region Risk Rating", [1, 2, 3])

st.divider()

# ── Predict ──────────────────────────────────────────────────────────
if st.button("🔍 Assess Default Risk", use_container_width=True, type="primary"):
    payload = {
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": float(income),
        "AMT_CREDIT": float(credit),
        "AMT_ANNUITY": float(annuity),
        "AMT_GOODS_PRICE": float(goods_price),
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -(age_years * 365),
        "DAYS_EMPLOYED": -(years_employed * 365),
        "DAYS_REGISTRATION": -5000.0,
        "DAYS_ID_PUBLISH": -3000,
        "FLAG_EMP_PHONE": 1,
        "FLAG_PHONE": 0,
        "FLAG_DOCUMENT_3": 1,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "REGION_RATING_CLIENT": region_rating,
        "REGION_RATING_CLIENT_W_CITY": region_rating,
        "REGION_POPULATION_RELATIVE": 0.02,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "OCCUPATION_TYPE": occupation,
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "ORGANIZATION_TYPE": organization,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DAYS_LAST_PHONE_CHANGE": -1000.0,
        "CNT_FAM_MEMBERS": float(cnt_fam_members),
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0
    }

    with st.spinner("Assessing risk..."):
        prob, label = predict(payload)

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Default Probability", f"{prob:.1%}")
        with r2:
            st.metric("Risk Assessment", label)
        with r3:
            st.metric("Threshold Used", f"{threshold:.2f}")

        if prob < threshold:
            st.success(f"✅ LOW RISK — This applicant has a {prob:.1%} probability of defaulting.")
        elif prob < 0.40:
            st.warning(f"⚠️ MEDIUM RISK — This applicant has a {prob:.1%} probability of defaulting.")
        else:
            st.error(f"🚨 HIGH RISK — This applicant has a {prob:.1%} probability of defaulting.")

        st.markdown("### Risk Score")
        st.progress(float(prob))