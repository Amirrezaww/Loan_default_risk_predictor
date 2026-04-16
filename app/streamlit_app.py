import streamlit as st
import requests

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
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree"
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
        "Business Entity Type 3", "School", "Government",
        "Religion", "Other", "Medicine", "Business Entity Type 2",
        "Self-employed", "Transport: type 2", "Construction",
        "Housing", "Kindergarten", "Trade: type 7", "Industry: type 11",
        "Military", "Services", "Security Ministries", "Transport: type 4",
        "Industry: type 1", "Emergency", "Security", "Trade: type 2",
        "University", "Transport: type 3", "Police", "Business Entity Type 1",
        "Postal", "Industry: type 4", "Agriculture", "Restaurant",
        "Culture", "Hotel", "Industry: type 7", "Trade: type 3",
        "Industry: type 3", "Bank", "Industry: type 9", "Insurance",
        "Trade: type 6", "Industry: type 2", "Transport: type 1",
        "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5",
        "Industry: type 10", "Legal Services", "Advertising",
        "Trade: type 5", "Cleaning", "Industry: type 13",
        "Trade: type 4", "Telecom", "Industry: type 8",
        "Realtor", "Industry: type 6", "XNA"
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

# ── Predict button ───────────────────────────────────────────────────
if st.button("🔍 Assess Default Risk", use_container_width=True, type="primary"):
    # Build payload
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
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()

            prob = result['default_probability']
            label = result['risk_label']

            # ── Results display ──────────────────────────────────────
            st.divider()
            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Default Probability", f"{prob:.1%}")

            with r2:
                st.metric("Risk Assessment", label)

            with r3:
                st.metric("Threshold Used", f"{result['threshold_used']:.2f}")

            # ── Risk gauge ───────────────────────────────────────────
            if prob < 0.17:
                st.success(f"✅ LOW RISK — This applicant has a {prob:.1%} probability of defaulting.")
            elif prob < 0.40:
                st.warning(f"⚠️ MEDIUM RISK — This applicant has a {prob:.1%} probability of defaulting.")
            else:
                st.error(f"🚨 HIGH RISK — This applicant has a {prob:.1%} probability of defaulting.")

            # ── Risk bar ─────────────────────────────────────────────
            st.markdown("### Risk Score")
            st.progress(float(prob))

        except Exception as e:
            st.error(f"API error: {e}. Make sure the FastAPI server is running.")