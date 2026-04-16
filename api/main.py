from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.json"
ARTIFACTS_PATH = BASE_DIR / "models" / "artifacts.pkl"

# ── Load model & artifacts ──────────────────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL_PATH))

with open(ARTIFACTS_PATH, 'rb') as f:
    artifacts = pickle.load(f)

encoder = artifacts['encoder']
cat_cols = artifacts['cat_cols']
feature_cols = artifacts['feature_cols']
threshold = artifacts['optimal_threshold']

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Risk Predictor",
    description="Predicts probability of loan default using XGBoost",
    version="1.0.0"
)

# ── Input schema — only key user-facing fields ───────────────────────
class LoanApplication(BaseModel):
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "M"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    CNT_CHILDREN: int = 0
    AMT_INCOME_TOTAL: float = 135000.0
    AMT_CREDIT: float = 500000.0
    AMT_ANNUITY: float = 25000.0
    AMT_GOODS_PRICE: float = 450000.0
    NAME_TYPE_SUITE: str = "Unaccompanied"
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    DAYS_BIRTH: int = -15000
    DAYS_EMPLOYED: int = -3000
    DAYS_REGISTRATION: float = -5000.0
    DAYS_ID_PUBLISH: int = -3000
    FLAG_EMP_PHONE: int = 1
    FLAG_PHONE: int = 0
    FLAG_DOCUMENT_3: int = 1
    AMT_REQ_CREDIT_BUREAU_YEAR: float = 1.0
    AMT_REQ_CREDIT_BUREAU_QRT: float = 0.0
    REGION_RATING_CLIENT: int = 2
    REGION_RATING_CLIENT_W_CITY: int = 2
    REGION_POPULATION_RELATIVE: float = 0.02
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    OCCUPATION_TYPE: str = "Laborers"
    WEEKDAY_APPR_PROCESS_START: str = "MONDAY"
    ORGANIZATION_TYPE: str = "Business Entity Type 3"
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    OBS_30_CNT_SOCIAL_CIRCLE: float = 0.0
    OBS_60_CNT_SOCIAL_CIRCLE: float = 0.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    DAYS_LAST_PHONE_CHANGE: float = -1000.0
    CNT_FAM_MEMBERS: float = 2.0
    LIVE_CITY_NOT_WORK_CITY: int = 0
    REG_CITY_NOT_LIVE_CITY: int = 0
    REG_CITY_NOT_WORK_CITY: int = 0

# ── Default values for columns not in input schema ───────────────────
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

# ── Helper ───────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365
    df['YEARS_EMPLOYED'] = np.abs(df['DAYS_EMPLOYED']) / 365
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
    return df

# ── Routes ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Loan Default Risk Predictor API", "status": "running"}

@app.post("/predict")
def predict(application: LoanApplication):
    # Convert to dataframe
    df = pd.DataFrame([application.model_dump()])

    # Engineer features
    df = engineer_features(df)

    # Fill missing columns with defaults
    for col, val in COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val

    # Encode categoricals
    df[cat_cols] = encoder.transform(df[cat_cols])

    # Align to training feature order
    df = df[feature_cols]

    # Predict
    prob = xgb_model.predict_proba(df)[:, 1][0]
    label = int(prob >= threshold)

    return {
        "default_probability": round(float(prob), 4),
        "risk_label": "HIGH RISK" if label == 1 else "LOW RISK",
        "threshold_used": threshold
    }