# predict.py

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from .feature_extractor import build_feature_dataframe
from .nlp_engine import prepare_ml_text, process_job_description, calculate_final_score

# CONFIGURATION
MODELS_DIR = Path(__file__).parent / "models"

ENSEMBLE_WEIGHTS = {
    "xgboost":          0.50,
    "random_forest":    0.35,
    "isolation_forest": 0.15,
}

# DATA STRUCTURE
@dataclass
class MLPrediction:
    """Result from ML prediction with explanations."""
    xgboost_score:          float = 0.0
    random_forest_score:    float = 0.0
    isolation_forest_score: float = 0.0

    ensemble_score: float = 0.0
    risk_level:     str   = "Safe"
    is_scam:        bool  = False
    confidence:     float = 0.0

    top_risk_features: list = field(default_factory=list)
    top_safe_features: list = field(default_factory=list)

    # Explainable AI parameters
    matched_keywords:     list = field(default_factory=list)
    financial_risk_score: float = 0.0
    contact_risk_score:   float = 0.0
    urgency_score:        float = 0.0
    domain_risk_score:    float = 0.0
    reasons:              list = field(default_factory=list)

# MODEL LOADING (Singleton - load once, reuse)
_models_cache = {
    "loaded": False,
    "xgboost": None,
    "random_forest": None,
    "isolation_forest": None,
    "scaler": None,
    "feature_columns": None,
    "tfidf": None,
}

def load_models() -> dict:
    """Load all trained models (cached after first call)."""
    if _models_cache["loaded"]:
        return _models_cache

    print("Loading trained models...")
    try:
        _models_cache["xgboost"]          = joblib.load(MODELS_DIR / "xgboost.pkl")
        _models_cache["random_forest"]    = joblib.load(MODELS_DIR / "random_forest.pkl")
        _models_cache["isolation_forest"] = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        _models_cache["scaler"]           = joblib.load(MODELS_DIR / "scaler.pkl")
        _models_cache["tfidf"]            = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

        with open(MODELS_DIR / "feature_columns.json") as f:
            _models_cache["feature_columns"] = json.load(f)

        _models_cache["loaded"] = True
        print("   All models loaded successfully")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Models not found: {e}\n"
            f"Run: python -m backend.ml.train_models"
        )

    return _models_cache

# PREDICTION FUNCTIONS

def predict_job(job: dict, verbose: bool = False) -> MLPrediction:
    """
    Predict if a single job is a scam.
    """
    models = load_models()

    import copy
    job_copy = copy.deepcopy(job)
    clean_text = prepare_ml_text(job_copy.get("job_description", "") or "")
    job_copy["job_description"] = clean_text

    # Step 1: Extract features
    df = build_feature_dataframe([job_copy], fit_tfidf=False)

    # Step 2: Align columns with training data
    expected_cols = models["feature_columns"]
    missing_cols = {col: 0 for col in expected_cols if col not in df.columns}
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(missing_cols, index=df.index)], axis=1)
    df = df[expected_cols]

    # Step 3: Get predictions from each model
    prediction = MLPrediction()

    xgb_proba = models["xgboost"].predict_proba(df)[0]
    prediction.xgboost_score = float(xgb_proba[1] * 100)

    rf_proba = models["random_forest"].predict_proba(df)[0]
    prediction.random_forest_score = float(rf_proba[1] * 100)

    iso_score_raw = models["isolation_forest"].score_samples(df)[0]
    iso_normalized = max(0, min(100, (-iso_score_raw + 0.5) * 100))
    prediction.isolation_forest_score = float(iso_normalized)

    # Step 4: ML Ensemble weighted average
    ml_ensemble_score = (
        prediction.xgboost_score          * ENSEMBLE_WEIGHTS["xgboost"]          +
        prediction.random_forest_score    * ENSEMBLE_WEIGHTS["random_forest"]    +
        prediction.isolation_forest_score * ENSEMBLE_WEIGHTS["isolation_forest"]
    )

    # Connect NLP Engine Heuristics
    nlp_analysis = process_job_description(job.get("job_description", "") or "")
    keyword_score = nlp_analysis.get("keyword_score", 0.0)

    # Compute Calibrated File Score: (0.6 * NLP) + (0.4 * ML)
    hybrid_results = calculate_final_score(keyword_score=keyword_score, ml_score=ml_ensemble_score)
    prediction.ensemble_score = hybrid_results["final_score"]
    
    # Step 5: Risk level mapping based on hybrid score
    if prediction.ensemble_score <= 20:
        prediction.risk_level = "Safe"
    elif prediction.ensemble_score <= 40:
        prediction.risk_level = "Low Risk"
    elif prediction.ensemble_score <= 60:
        prediction.risk_level = "Medium Risk"
    elif prediction.ensemble_score <= 80:
        prediction.risk_level = "High Risk"
    else:
        prediction.risk_level = "Scam Likely"

    prediction.is_scam = prediction.ensemble_score > 50

    scores = [prediction.xgboost_score, prediction.random_forest_score, prediction.isolation_forest_score]
    score_std = float(np.std(scores))
    prediction.confidence = round(max(0, 100 - score_std), 2)

    # Populate Explainable AI Parameters
    prediction.financial_risk_score = float(df.get("financial_risk_score", [0.0]).iloc[0])
    prediction.contact_risk_score   = float(df.get("contact_risk_score", [0.0]).iloc[0])
    prediction.urgency_score        = float(df.get("urgency_score", [0.0]).iloc[0])
    prediction.domain_risk_score    = float(df.get("domain_risk_score", [0.0]).iloc[0])
    prediction.matched_keywords     = nlp_analysis.get("matched_keywords", [])

    # Generate reasons based on active features
    reasons = []
    if df.get("has_registration_fee", [0]).iloc[0] > 0:
        reasons.append("Registration fee detected")
    if df.get("has_training_fee", [0]).iloc[0] > 0:
        reasons.append("Training fee detected")
    if df.get("has_deposit", [0]).iloc[0] > 0:
        reasons.append("Security deposit detected")
    if df.get("has_telegram", [0]).iloc[0] > 0:
        reasons.append("Telegram contact detected")
    if df.get("has_whatsapp", [0]).iloc[0] > 0:
        reasons.append("WhatsApp contact detected")
    if df.get("has_personal_email", [0]).iloc[0] > 0:
        reasons.append("Personal email address used")
    if df.get("has_disposable_email", [0]).iloc[0] > 0:
        reasons.append("Disposable email address used")
    if df.get("has_corporate_email", [0]).iloc[0] == 0:
        reasons.append("No corporate email")
    if df.get("domain_age", [0]).iloc[0] < 365 and df.get("domain_age", [0]).iloc[0] > 0:
        reasons.append("Suspicious domain age")
    elif df.get("whois_available", [0]).iloc[0] == 0:
        reasons.append("WHOIS records not available")
    if df.get("ssl_valid", [0]).iloc[0] == 0:
        reasons.append("No valid SSL certificate")
    if df.get("suspicious_tld", [0]).iloc[0] > 0:
        reasons.append("Suspicious domain extension")
    prediction.reasons = reasons

    prediction.top_risk_features = _get_top_risk_features(df, models)
    prediction.top_safe_features = _get_top_safe_features(df, models)

    if verbose:
        print_prediction(prediction, job)
        print("   [HYBRID PIPELINE BRIDGE]")
        print(f"   |-- Raw NLP Keyword Score: {keyword_score}/100")
        print(f"   |-- Raw ML Ensemble Score: {ml_ensemble_score:.2f}/100")
        print(f"   +-- Calculated File Score: {prediction.ensemble_score}/100")

    return prediction

def predict_batch(jobs: list, verbose: bool = False) -> list:
    if verbose:
        print(f"\nPredicting on {len(jobs)} jobs...")

    results = []
    for i, job in enumerate(jobs, 1):
        result = predict_job(job, verbose=False)
        results.append(result)

        if verbose and i % 10 == 0:
            print(f"   Processed {i}/{len(jobs)}")

    return results

# FEATURE INTERPRETATION

def _get_top_risk_features(df: pd.DataFrame, models: dict, top_n: int = 5) -> list:
    rf_model = models["random_forest"]
    feature_importance = rf_model.feature_importances_
    feature_values = df.iloc[0].values
    risk_scores = feature_importance * feature_values

    feature_names = list(df.columns)
    risk_features = []

    sorted_indices = np.argsort(risk_scores)[::-1]
    for idx in sorted_indices[:top_n * 3]:
        name = feature_names[idx]
        value = feature_values[idx]

        if name.startswith("tfidf_") or name.startswith("emb_") or value == 0:
            continue

        if risk_scores[idx] > 0.001:
            risk_features.append({
                "feature": _humanize_feature_name(name),
                "value": float(value),
                "impact": round(float(risk_scores[idx]), 4)
            })

        if len(risk_features) >= top_n:
            break

    return risk_features

def _get_top_safe_features(df: pd.DataFrame, models: dict, top_n: int = 3) -> list:
    safe_indicators = []
    job_row = df.iloc[0]

    safe_checks = [
        ("has_corporate_email", "Corporate email used"),
        ("legit_keyword_count", "Professional language detected"),
        ("has_skills",          "Skills clearly listed"),
        ("has_salary",          "Salary disclosed"),
        ("skill_count",         "Multiple skills required"),
        ("is_internshala",      "From verified platform (Internshala)"),
        ("is_linkedin",         "From verified platform (LinkedIn)"),
        ("is_ncs",              "From verified platform (NCS - Govt)"),
    ]

    for feature, description in safe_checks:
        if feature in job_row.index and job_row[feature] > 0:
            safe_indicators.append({
                "feature": description,
                "value": float(job_row[feature]),
            })
            if len(safe_indicators) >= top_n:
                break

    return safe_indicators

def _humanize_feature_name(name: str) -> str:
    name_map = {
        "fraud_keyword_count":   "Contains suspicious keywords",
        "has_registration_fee":  "Mentions registration fee",
        "has_training_fee":      "Mentions training fee",
        "has_whatsapp":          "WhatsApp contact in description",
        "has_telegram":          "Telegram contact in description",
        "has_personal_email":    "Uses personal email (Gmail/Yahoo)",
        "has_disposable_email":  "Uses disposable email",
        "has_urgent":            "Contains urgent language",
        "has_guarantee":         "Makes guarantees",
        "unrealistic_salary":    "Unrealistic salary claims",
        "has_deposit":           "Asks for deposit",
        "has_fee":               "Mentions fees",
        "has_instagram":         "Uses Instagram for hiring",
    }
    return name_map.get(name, name.replace("_", " ").title())

# DISPLAY HELPERS

def print_prediction(prediction: MLPrediction, job: dict = None):
    print("\n" + "=" * 70)
    if job:
        title = job.get("job_title", "Unknown")[:60]
        company = job.get("company_name", "Unknown")[:40]
        print(f"JOB:     {title}")
        print(f"COMPANY: {company}")
    print("=" * 70)

    print(f"\nVERDICT:    {prediction.risk_level} ({prediction.ensemble_score}/100)")
    print(f"CONFIDENCE: {prediction.confidence}%")
    print(f"IS SCAM:    {'YES' if prediction.is_scam else 'NO'}")

    print(f"\nMODEL BREAKDOWN:")
    print(f"   XGBoost:          {prediction.xgboost_score:5.1f}/100")
    print(f"   Random Forest:    {prediction.random_forest_score:5.1f}/100")
    print(f"   Isolation Forest: {prediction.isolation_forest_score:5.1f}/100")
    print(f"   {'-' * 30}")
    print(f"   ENSEMBLE:         {prediction.ensemble_score:5.1f}/100")

    if prediction.top_risk_features:
        print(f"\nTOP RISK SIGNALS:")
        for f in prediction.top_risk_features:
            print(f"   - {f['feature']}")

    if prediction.top_safe_features:
        print(f"\nSAFETY SIGNALS:")
        for f in prediction.top_safe_features:
            print(f"   - {f['feature']}")

    print("\n" + "=" * 70)

# SELF-TEST
def main():
    print("=" * 70)
    print("ML PREDICTION ENGINE - SELF-TEST")
    print("=" * 70)

    test_jobs = [
        {
            "job_title": "Earn 50k Daily From Home!",
            "job_description": (
                "URGENT! Earn Rs 50,000 daily working from home! "
                "No experience needed. Pay just Rs 500 registration fee. "
                "WhatsApp us at +91-9876543210. Apply on Telegram t.me/jobs123. "
                "Limited slots! Hurry up!"
            ),
            "skills_required": [],
            "skill_categories": {},
            "salary_min": 0,
            "salary_max": 50000000,
            "salary_raw": "Unlimited earnings",
            "city": "Remote",
            "mode": "Remote",
            "platform_name": "Unknown",
            "company_name": "ABC IT Solutions Pvt Ltd",
        }
    ]

    for job in test_jobs:
        predict_job(job, verbose=True)

if __name__ == "__main__":
    main()
