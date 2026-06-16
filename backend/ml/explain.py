"""
explain.py
SHAP-based explanations for ML predictions.

Tells users WHY the model predicted scam/safe by showing each feature's
contribution to the final score.

Usage:
    python -m backend.ml.explain

    # Programmatically:
    from backend.ml.explain import explain_prediction
    explanation = explain_prediction(job_dict)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from .feature_extractor import build_feature_dataframe
from .predict import load_models


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent / "models"

FEATURE_DESCRIPTIONS = {
    "fraud_keyword_count":   "Number of suspicious keywords",
    "legit_keyword_count":   "Number of professional keywords",
    "has_registration_fee":  "Mentions registration fee",
    "has_training_fee":      "Mentions training fee",
    "has_deposit":           "Asks for deposit",
    "has_fee":               "Mentions any kind of fee",
    "has_whatsapp":          "WhatsApp contact in description",
    "has_telegram":          "Telegram contact in description",
    "has_instagram":         "Instagram contact in description",
    "has_email":             "Email contact provided",
    "has_phone":             "Phone number in description",
    "has_personal_email":    "Uses personal email (Gmail/Yahoo)",
    "has_disposable_email":  "Uses disposable/temp email",
    "has_corporate_email":   "Uses corporate email",
    "has_urgent":            "Uses urgent language",
    "has_guarantee":         "Makes guarantees",
    "unrealistic_salary":    "Unrealistic salary claim",
    "salary_min":            "Minimum salary",
    "salary_max":            "Maximum salary",
    "salary_range":          "Salary range",
    "has_salary":            "Salary disclosed",
    "description_length":    "Description length",
    "word_count":            "Word count",
    "has_description":       "Has detailed description",
    "skill_count":           "Number of required skills",
    "has_skills":            "Skills listed",
    "programming_count":     "Programming skills",
    "framework_count":       "Framework skills",
    "database_count":        "Database skills",
    "cloud_count":           "Cloud/DevOps skills",
    "ml_count":              "ML/AI skills",
    "business_count":        "Business skills",
    "is_remote":             "Remote position",
    "has_location":          "Location specified",
    "company_trust_score":   "Company trust score",
    "recruiter_verif_score": "Recruiter verification score",
    "is_internshala":        "From Internshala",
    "is_linkedin":           "From LinkedIn",
    "is_naukri":             "From Naukri",
    "is_ncs":                "From NCS (Government)",
    "is_shine":              "From Shine",
    "mode_remote":           "Remote work",
    "mode_hybrid":           "Hybrid work",
    "mode_onsite":           "On-site work",
}


# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class SHAPExplanation:
    """SHAP-based explanation for a single prediction."""
    base_value:  float = 0.0
    final_value: float = 0.0

    risk_factors: list = field(default_factory=list)
    safe_factors: list = field(default_factory=list)
    all_contributions: list = field(default_factory=list)


# ============================================================================
# SHAP EXPLAINER (Cached singleton)
# ============================================================================

_explainer_cache = {
    "loaded": False,
    "xgboost_explainer": None,
    "rf_explainer": None,
}


def get_shap_explainer():
    """Load SHAP explainer (cached after first call)."""
    if _explainer_cache["loaded"]:
        return _explainer_cache

    print("Initializing SHAP explainers...")

    models = load_models()

    _explainer_cache["xgboost_explainer"] = shap.TreeExplainer(models["xgboost"])
    _explainer_cache["rf_explainer"]      = shap.TreeExplainer(models["random_forest"])

    _explainer_cache["loaded"] = True
    print("   SHAP explainers ready")

    return _explainer_cache


# ============================================================================
# MAIN EXPLANATION FUNCTION
# ============================================================================

def explain_prediction(
    job: dict,
    model_name: str = "xgboost",
    top_n: int = 5,
) -> SHAPExplanation:
    """
    Explain why the model gave this prediction using SHAP values.

    Args:
        job: Job dict (same format as predict_job)
        model_name: "xgboost" or "random_forest"
        top_n: Number of top features to return per direction

    Returns:
        SHAPExplanation with detailed breakdown
    """
    models = load_models()
    explainers = get_shap_explainer()

    # Step 1: Extract features
    df = build_feature_dataframe([job], fit_tfidf=False)

    # Step 2: Align columns
    expected_cols = models["feature_columns"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Step 3: Get SHAP values
    if model_name == "xgboost":
        explainer = explainers["xgboost_explainer"]
    else:
        explainer = explainers["rf_explainer"]

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values

    if len(shap_matrix.shape) == 2:
        shap_array = shap_matrix[0]
    else:
        shap_array = shap_matrix

    # Step 4: Get base value safely
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
        else:
            base_value = float(explainer.expected_value)
    else:
        base_value = 0.0

    # Step 5: Build contribution list
    feature_names = list(df.columns)
    feature_values = df.iloc[0].values

    contributions = []
    for name, value, shap_val in zip(feature_names, feature_values, shap_array):
        if name.startswith("tfidf_"):
            continue
        if abs(shap_val) < 0.001:
            continue

        contributions.append({
            "feature":      name,
            "description":  _humanize_feature(name),
            "value":        float(value),
            "contribution": float(shap_val),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Step 6: Separate risk vs safe factors
    risk_factors = [c for c in contributions if c["contribution"] > 0][:top_n]
    safe_factors = [c for c in contributions if c["contribution"] < 0][:top_n]

    final_value = base_value + sum(c["contribution"] for c in contributions)

    return SHAPExplanation(
        base_value=base_value,
        final_value=final_value,
        risk_factors=risk_factors,
        safe_factors=safe_factors,
        all_contributions=contributions[:20],
    )


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def _humanize_feature(name: str) -> str:
    """Convert feature name to readable description."""
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ").title())


def print_explanation(explanation: SHAPExplanation, job: dict = None):
    """Print a formatted SHAP explanation."""
    print("\n" + "=" * 70)
    if job:
        print(f"JOB:     {job.get('job_title', 'Unknown')[:60]}")
        print(f"COMPANY: {job.get('company_name', 'Unknown')[:50]}")
        print("=" * 70)

    print(f"\nSHAP EXPLANATION (Why this score?)")
    print(f"{'-' * 70}")
    print(f"   Base value (avg prediction): {explanation.base_value:.4f}")
    print(f"   Final prediction:            {explanation.final_value:.4f}")
    print(f"   Total contribution:          {explanation.final_value - explanation.base_value:+.4f}")

    if explanation.risk_factors:
        print(f"\nTOP REASONS THIS LOOKS LIKE A SCAM:")
        print(f"   {'-' * 67}")
        for i, factor in enumerate(explanation.risk_factors, 1):
            arrow = "+" * min(5, int(abs(factor['contribution']) * 10 + 1))
            print(f"   {i}. {factor['description']:<40s} "
                  f"[{factor['value']:.0f}] {arrow} +{factor['contribution']:.3f}")

    if explanation.safe_factors:
        print(f"\nTOP REASONS THIS LOOKS LEGITIMATE:")
        print(f"   {'-' * 67}")
        for i, factor in enumerate(explanation.safe_factors, 1):
            arrow = "-" * min(5, int(abs(factor['contribution']) * 10 + 1))
            print(f"   {i}. {factor['description']:<40s} "
                  f"[{factor['value']:.0f}] {arrow} {factor['contribution']:.3f}")

    if not explanation.risk_factors and not explanation.safe_factors:
        print(f"\n   No strong contributing features (uncertain prediction)")

    print("\n" + "=" * 70)


def generate_user_friendly_explanation(explanation: SHAPExplanation) -> dict:
    """
    Generate user-friendly explanation text (for frontend display).

    Returns:
        Dict with 'summary', 'why_risky', 'why_safe', 'verdict'
    """
    result = {
        "summary":   "",
        "why_risky": [],
        "why_safe":  [],
        "verdict":   "uncertain",
    }

    total_risk = sum(c["contribution"] for c in explanation.risk_factors)
    total_safe = abs(sum(c["contribution"] for c in explanation.safe_factors))

    if total_risk > total_safe * 1.5:
        result["verdict"] = "scam"
        result["summary"] = "This job has multiple scam indicators."
    elif total_safe > total_risk * 1.5:
        result["verdict"] = "safe"
        result["summary"] = "This job appears legitimate."
    else:
        result["verdict"] = "uncertain"
        result["summary"] = "This job has mixed signals. Be cautious."

    for factor in explanation.risk_factors[:5]:
        if factor["value"] > 0:
            result["why_risky"].append({
                "reason":   factor["description"],
                "severity": "high" if factor["contribution"] > 0.5 else "medium",
            })

    for factor in explanation.safe_factors[:3]:
        result["why_safe"].append({
            "reason":   factor["description"],
            "strength": "strong" if abs(factor["contribution"]) > 0.3 else "moderate",
        })

    return result


# ============================================================================
# SELF-TEST
# ============================================================================

def main():
    print("=" * 70)
    print("SHAP EXPLANATION ENGINE - SELF-TEST")
    print("=" * 70)

    test_jobs = [
        # Obvious scam
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
        },

        # Legitimate
        {
            "job_title": "Python Developer Intern",
            "job_description": (
                "We are looking for a passionate Python developer for our team. "
                "Mentorship provided. Health insurance and ESOP available. "
                "Apply via our careers page."
            ),
            "skills_required": ["Python", "Django", "PostgreSQL", "AWS"],
            "skill_categories": {
                "programming": ["Python"],
                "frameworks":  ["Django"],
                "databases":   ["PostgreSQL"],
                "cloud_devops":["AWS"],
            },
            "salary_min": 25000,
            "salary_max": 25000,
            "salary_raw": "Rs 25,000/month",
            "city": "Bengaluru",
            "mode": "Hybrid",
            "platform_name": "Internshala",
            "company_name": "Brightline Software",
        },
    ]

    for job in test_jobs:
        explanation = explain_prediction(job, model_name="xgboost")
        print_explanation(explanation, job)

        friendly = generate_user_friendly_explanation(explanation)
        print(f"\nUSER-FRIENDLY EXPLANATION:")
        print(f"   Verdict: {friendly['verdict'].upper()}")
        print(f"   Summary: {friendly['summary']}")

        if friendly["why_risky"]:
            print(f"\n   Why it might be risky:")
            for r in friendly["why_risky"]:
                marker = "[HIGH]  " if r["severity"] == "high" else "[MED]   "
                print(f"   {marker}{r['reason']}")

        if friendly["why_safe"]:
            print(f"\n   Why it might be safe:")
            for r in friendly["why_safe"]:
                marker = "[STRONG]" if r["strength"] == "strong" else "[MOD]   "
                print(f"   {marker}{r['reason']}")

    # Test on real Supabase job
    print("\n" + "=" * 70)
    print("TESTING ON REAL JOB FROM SUPABASE")
    print("=" * 70)

    try:
        from ..scraper.storage.supabase_client import get_client
        sb = get_client()

        response = sb.table("jobs").select(
            "*, companies(name, company_trust_score)"
        ).limit(1).execute()

        if response.data:
            job = response.data[0]
            if job.get("companies"):
                job["company_name"] = job["companies"].get("name", "")
                job["company_trust_score"] = job["companies"].get("company_trust_score", 50)

            explanation = explain_prediction(job)
            print_explanation(explanation, job)

    except Exception as e:
        print(f"Could not test on Supabase: {e}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()