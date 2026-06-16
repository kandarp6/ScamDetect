"""
feature_extractor.py
Convert scraped jobs into ML-ready numerical features.

Outputs ~146 features per job:
    - Numeric metrics (description length, salary, skill count, etc.)
    - Boolean flags (has_email, has_whatsapp, has_registration_fee, etc.)
    - Skill category counts
    - TF-IDF text features (top 100 keywords)

Usage:
    python -m backend.ml.feature_extractor
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

from .nlp_engine import prepare_ml_text


# ============================================================================
# CONSTANTS
# ============================================================================

FRAUD_KEYWORDS = [
    "registration fee", "training fee", "joining fee", "deposit",
    "earn daily", "earn weekly", "unlimited earning", "guaranteed",
    "no experience", "anyone can", "urgent hiring", "limited seats",
    "whatsapp", "telegram", "bitcoin", "crypto", "mlm", "pyramid",
]

LEGIT_KEYWORDS = [
    "salary", "benefits", "insurance", "provident fund", "esop",
    "mentorship", "training program", "professional", "growth",
    "interview", "experience required", "qualified",
]

PERSONAL_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
DISPOSABLE_DOMAINS = {"mailinator.com", "tempmail.com", "yopmail.com"}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Set to True for noisy debug output, False for batch/production use
VERBOSE = False


# ============================================================================
# NUMERIC FEATURE EXTRACTION
# ============================================================================

def extract_numeric_features(job: dict) -> dict:
    """Extract numeric and boolean features from a single job."""
    features = {}

    desc = job.get("job_description", "") or ""
    title = job.get("job_title", "") or ""

    # Text metrics
    features["description_length"] = len(desc)
    features["word_count"]         = len(desc.split())
    features["title_length"]       = len(title)
    features["has_description"]    = 1 if len(desc) > 100 else 0

    desc_lower = desc.lower()

    # Keyword counts
    features["fraud_keyword_count"] = sum(1 for kw in FRAUD_KEYWORDS if kw in desc_lower)
    features["legit_keyword_count"] = sum(1 for kw in LEGIT_KEYWORDS if kw in desc_lower)

    # Contact method flags
    features["has_email"]     = 1 if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', desc) else 0
    features["has_phone"]     = 1 if re.search(r'\+?91[\s\-]?[6-9]\d{9}', desc) else 0
    features["has_whatsapp"]  = 1 if re.search(r'wa\.me/|whatsapp|whats\s*app', desc_lower) else 0
    features["has_telegram"]  = 1 if re.search(r't\.me/|telegram', desc_lower) else 0
    features["has_instagram"] = 1 if re.search(r'instagram|insta\b', desc_lower) else 0

    # Payment fraud flags
    features["has_registration_fee"] = 1 if "registration fee" in desc_lower else 0
    features["has_training_fee"]     = 1 if "training fee" in desc_lower else 0
    features["has_deposit"]          = 1 if "deposit" in desc_lower else 0
    features["has_fee"]              = 1 if " fee" in desc_lower else 0

    # Urgent / guarantee language
    features["has_urgent"] = 1 if any(
        word in desc_lower for word in ["urgent", "immediate", "limited", "hurry"]
    ) else 0
    features["has_guarantee"] = 1 if "guarantee" in desc_lower or "guaranteed" in desc_lower else 0

    # Email domain analysis
    email_match = re.search(r'@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', desc)
    email_domain = email_match.group(1).lower() if email_match else ""

    features["has_personal_email"]   = 1 if email_domain in PERSONAL_EMAIL_DOMAINS else 0
    features["has_disposable_email"] = 1 if email_domain in DISPOSABLE_DOMAINS else 0
    features["has_corporate_email"]  = 1 if (
        email_domain
        and email_domain not in PERSONAL_EMAIL_DOMAINS
        and email_domain not in DISPOSABLE_DOMAINS
    ) else 0

    # Salary features
    features["salary_min"]         = float(job.get("salary_min", 0) or 0)
    features["salary_max"]         = float(job.get("salary_max", 0) or 0)
    features["salary_range"]       = features["salary_max"] - features["salary_min"]
    features["has_salary"]         = 1 if features["salary_min"] > 0 else 0
    features["unrealistic_salary"] = 1 if features["salary_max"] > 50_000_000 else 0

    # Skills
    skills = job.get("skills_required", []) or []
    features["skill_count"] = len(skills)
    features["has_skills"]  = 1 if len(skills) > 0 else 0

    # Skill categories
    categories = job.get("skill_categories", {}) or {}
    features["programming_count"] = len(categories.get("programming", []))
    features["framework_count"]   = len(categories.get("frameworks", []))
    features["database_count"]    = len(categories.get("databases", []))
    features["cloud_count"]       = len(categories.get("cloud_devops", []))
    features["ml_count"]          = len(categories.get("data_ml", []))
    features["business_count"]    = len(categories.get("business", []))
    features["mobile_count"]      = len(categories.get("mobile", []))

    # Location
    features["is_remote"] = 1 if (
        (job.get("mode", "") or "").lower() == "remote"
        or (job.get("city", "") or "").lower() == "remote"
    ) else 0
    features["has_location"] = 1 if job.get("city") else 0

    # Trust scores
    features["company_trust_score"]   = float(job.get("company_trust_score", 50) or 50)
    features["recruiter_verif_score"] = float(job.get("recruiter_verification_score", 30) or 30)

    # Platform one-hot
    platform = (job.get("platform_name", "") or "").lower()
    features["is_internshala"] = 1 if "internshala" in platform else 0
    features["is_linkedin"]    = 1 if "linkedin" in platform else 0
    features["is_naukri"]      = 1 if "naukri" in platform else 0
    features["is_ncs"]         = 1 if "ncs" in platform else 0
    features["is_shine"]       = 1 if "shine" in platform else 0

    # Mode one-hot
    mode = (job.get("mode", "") or "").lower()
    features["mode_remote"] = 1 if "remote" in mode else 0
    features["mode_hybrid"] = 1 if "hybrid" in mode else 0
    features["mode_onsite"] = 1 if "on-site" in mode or "onsite" in mode else 0

    return features


# ============================================================================
# TF-IDF TEXT FEATURES
# ============================================================================

def fit_tfidf_vectorizer(descriptions: list, max_features: int = 100) -> TfidfVectorizer:
    """
    Train a TF-IDF vectorizer on all job descriptions.
    Adapts settings automatically for small datasets.
    """
    descriptions = list(descriptions)
    n_docs = len(descriptions)

    if n_docs < 5:
        min_df, max_df = 1, 1.0
    else:
        min_df, max_df = 2, 0.95

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        strip_accents='unicode',
    )

    vectorizer.fit(descriptions)
    joblib.dump(vectorizer, TFIDF_PATH)

    if VERBOSE:
        print(f"TF-IDF vectorizer saved to {TFIDF_PATH}")

    return vectorizer


def get_tfidf_features(descriptions: list, vectorizer: TfidfVectorizer = None) -> pd.DataFrame:
    """Convert descriptions to TF-IDF feature matrix."""
    if vectorizer is None:
        vectorizer = joblib.load(TFIDF_PATH)

    tfidf_matrix = vectorizer.transform(descriptions)
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]

    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_feature_dataframe(jobs: list, fit_tfidf: bool = True) -> pd.DataFrame:
    """
    Convert a list of jobs into a complete feature DataFrame.

    Args:
        jobs: List of job dicts
        fit_tfidf: True = train new TF-IDF (training only)
                   False = load existing (production / prediction)
    """
    if not jobs:
        return pd.DataFrame()

    if VERBOSE:
        print(f"\nExtracting features from {len(jobs)} jobs...")

    # Numeric features
    numeric_features = [extract_numeric_features(job) for job in jobs]
    df_numeric = pd.DataFrame(numeric_features)

    if VERBOSE:
        print(f"   Numeric features: {df_numeric.shape[1]} columns")

    # Overwrite job["job_description"] with clean_text before TF-IDF calculation
    for job in jobs:
        desc = job.get("job_description", "") or ""
        job["job_description"] = prepare_ml_text(desc)

    # TF-IDF text features
    descriptions = [
        (job.get("job_description", "") or "") + " " + (job.get("job_title", "") or "")
        for job in jobs
    ]

    if fit_tfidf:
        vectorizer = fit_tfidf_vectorizer(descriptions, max_features=100)
    else:
        vectorizer = joblib.load(TFIDF_PATH)

    df_tfidf = get_tfidf_features(descriptions, vectorizer)

    if VERBOSE:
        print(f"   TF-IDF features:  {df_tfidf.shape[1]} columns")

    # Combine
    df_combined = pd.concat(
        [df_numeric.reset_index(drop=True), df_tfidf.reset_index(drop=True)],
        axis=1
    )

    if VERBOSE:
        print(f"   Final matrix:     {df_combined.shape[0]} rows x {df_combined.shape[1]} columns")

    return df_combined


# ============================================================================
# LABEL EXTRACTION
# ============================================================================

def extract_labels(jobs: list) -> pd.DataFrame:
    """
    Extract target labels from jobs for ML training.

    Returns DataFrame with:
        - scam_score (0-100) — for regression
        - risk_level (categorical) — for classification
        - is_scam (0/1) — binary classification
    """
    labels = []
    for job in jobs:
        score = float(job.get("scam_score", 0) or 0)
        risk = job.get("scam_risk_level", "Safe") or "Safe"
        is_scam = 1 if risk in ("High Risk", "Scam Likely") else 0

        labels.append({
            "scam_score": score,
            "risk_level": risk,
            "is_scam":    is_scam,
        })

    return pd.DataFrame(labels)


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test():
    """Run feature extraction on real Supabase data (or demo data)."""
    global VERBOSE
    VERBOSE = True  # enable detailed prints for the self-test

    print("=" * 70)
    print("FEATURE EXTRACTOR - SELF-TEST")
    print("=" * 70)

    print("\nLoading jobs from Supabase...")

    try:
        from ..scraper.storage.supabase_client import get_client
        sb = get_client()

        response = sb.table("jobs").select(
            "*, companies(company_trust_score), recruiters(recruiter_verification_score)"
        ).execute()

        jobs = response.data
        print(f"   Loaded {len(jobs)} jobs")

        if not jobs:
            print("\nNo jobs in database. Run scraper first:")
            print("    python -m backend.scraper.main")
            return

        for job in jobs:
            if job.get("companies"):
                job["company_trust_score"] = job["companies"].get("company_trust_score", 50)
            if job.get("recruiters"):
                job["recruiter_verification_score"] = job["recruiters"].get("recruiter_verification_score", 30)

    except Exception as e:
        print(f"Failed to load from Supabase: {e}")
        print("\nFalling back to demo data...")

        jobs = [
            {
                "job_title": "Python Developer Intern",
                "job_description": "Looking for Python developer with Django, AWS skills. Mentorship provided. Health insurance.",
                "skills_required": ["Python", "Django", "AWS"],
                "skill_categories": {
                    "programming": ["Python"],
                    "frameworks":  ["Django"],
                    "cloud_devops":["AWS"],
                },
                "salary_min": 300000, "salary_max": 500000,
                "city": "Bengaluru", "mode": "Remote",
                "platform_name": "Internshala",
                "scam_score": 15, "scam_risk_level": "Safe",
            },
            {
                "job_title": "Earn 50k Daily",
                "job_description": "Earn 50k daily on WhatsApp! Pay 500 registration fee. Apply on Telegram now!",
                "skills_required": [],
                "skill_categories": {},
                "salary_min": 0, "salary_max": 0,
                "city": "", "mode": "Remote",
                "platform_name": "Unknown",
                "scam_score": 89, "scam_risk_level": "Scam Likely",
            },
        ]

    df_features = build_feature_dataframe(jobs, fit_tfidf=True)
    df_labels = extract_labels(jobs)

    print("\nSAMPLE FEATURES (first 3 rows, first 15 columns):")
    print(df_features.iloc[:3, :15].to_string())

    print("\nLABELS:")
    print(df_labels.head().to_string())

    print("\nSTATISTICS:")
    print(f"   Total jobs:     {len(df_features)}")
    print(f"   Total features: {df_features.shape[1]}")
    print(f"   Scam jobs:      {df_labels['is_scam'].sum()}")
    print(f"   Safe jobs:      {(df_labels['is_scam'] == 0).sum()}")

    print("\nRISK DISTRIBUTION:")
    print(df_labels['risk_level'].value_counts().to_string())

    print("\nSaving feature data...")
    features_path = MODELS_DIR / "features.csv"
    labels_path   = MODELS_DIR / "labels.csv"
    df_features.to_csv(features_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    print(f"   Saved: {features_path}")
    print(f"   Saved: {labels_path}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()