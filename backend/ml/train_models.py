"""
train_models.py
Train multiple ML models for job fraud detection.

Pipeline:
    1. Load real jobs from Supabase + generate synthetic data
    2. Split into train/test (RAW jobs, no leakage)
    3. Fit TF-IDF on training data only
    4. Cross-validate models on train set
    5. Train RF + XGBoost + Isolation Forest
    6. Save models + metadata

Usage:
    python -m backend.ml.train_models
"""

import json
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from .feature_extractor import (
    build_feature_dataframe,
    extract_labels,
)
from .nlp_engine import prepare_ml_text

# Suppress sklearn / xgboost noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# SYNTHETIC SCAM DATA GENERATION
# ============================================================================

SCAM_TEMPLATES = [
    """
    URGENT HIRING! {role} position available!
    Earn Rs {amount} {period} working from home!
    No experience needed. Anyone can apply!
    Pay just Rs {fee} registration fee to confirm your slot.
    Limited seats available! Hurry up!
    Contact us on WhatsApp: +91-{phone}
    Or join our Telegram channel: t.me/jobchannel{num}
    Send your resume to: hr{num}@gmail.com
    """,
    """
    Looking for {role}! Network marketing opportunity!
    Unlimited earning potential. Guaranteed salary Rs {amount}/month.
    Direct selling experience preferred but not required.
    Pay refundable security deposit of Rs {fee}.
    100% job guarantee after training.
    DM us on Instagram or WhatsApp +91-{phone}
    """,
    """
    {role} needed for bitcoin trading firm!
    Earn lakhs per week! Forex trading expertise.
    No experience required. We provide free training (Rs {fee} fee applies).
    Apply on Telegram: t.me/cryptojobs{num}
    Email: trading{num}@yahoo.com
    """,
    """
    Urgent {role} required for MLM business!
    Pyramid structure with unlimited income.
    Earn daily by recruiting others.
    Limited slots! Last chance to join!
    Initial investment: Rs {fee} (refundable)
    Guaranteed monthly income: Rs {amount}
    Contact HR on WhatsApp: +91-{phone}
    """,
    """
    {role} position! Immediate joining!
    Pay Rs {fee} training fee to get certified.
    100% job placement guarantee after training.
    Stipend during training: Rs {amount}
    Apply now or never! Limited seats!
    WhatsApp: +91-{phone}
    Telegram: t.me/training{num}
    Email: training{num}@gmail.com
    """,
]

SCAM_ROLES = [
    "Data Entry Operator", "Customer Support", "Marketing Executive",
    "Business Development", "Telecaller", "Receptionist",
    "Computer Operator", "Office Assistant", "Sales Representative",
]


def generate_synthetic_scam(num: int) -> dict:
    template = random.choice(SCAM_TEMPLATES)
    role = random.choice(SCAM_ROLES)

    description = template.format(
        role=role,
        amount=random.choice(["50,000", "1,00,000", "75,000", "2,00,000"]),
        period=random.choice(["daily", "weekly", "per day"]),
        fee=random.choice(["500", "999", "1500", "2500", "5000"]),
        phone=random.randint(9000000000, 9999999999),
        num=num,
    )

    # Mix lengths: append professional filler text to 30% of scams to make them long
    if random.random() < 0.3:
        filler = "\nResponsibilities include designing marketing materials, communicating with potential clients, and reporting to the director. Requirements: basic computer knowledge, internet access, and ability to follow instructions."
        description += filler

    # Add random skills to some scams
    skills = []
    if random.random() < 0.4:
        skills = random.sample(["Data Entry", "Excel", "Typing", "Communication"], k=random.randint(1, 3))

    # Mix trust scores to include realistic default scores (50 and 30)
    company_trust = random.choice([random.uniform(5, 20), 50.0, random.uniform(20, 60)])
    recruiter_verif = random.choice([random.uniform(5, 20), 30.0, random.uniform(20, 50)])

    # Mix titles to vary title lengths (overlap with legit titles)
    title_style = random.random()
    if title_style < 0.5:
        title = f"{role} - Urgent Hiring!!!"
    elif title_style < 0.8:
        title = role
    else:
        title = f"{role} (Work From Home)"

    # Mix salaries
    salary_style = random.random()
    if salary_style < 0.4:
        salary_min = 0.0
        salary_max = 0.0
    elif salary_style < 0.7:
        salary_min = float(random.choice([10000, 15000, 20000]))
        salary_max = float(salary_min + random.choice([5000, 10000, 15000]))
    else:
        # Unrealistic high salaries
        salary_min = 0.0
        salary_max = random.choice([50000000.0, 100000000.0])

    return {
        "job_title": title,
        "job_description": description.strip(),
        "skills_required": skills,
        "skill_categories": {},
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_raw": random.choice([
            "Earn 50k daily", "Unlimited", "Best in industry",
            "Rs 1 lakh per week", "Negotiable"
        ]),
        "city": random.choice(["Remote", "Anywhere", "Pan India", ""]),
        "state": "",
        "country": "India",
        "mode": "Remote",
        "platform_name": random.choice([
            "Unknown", "Random Site", "FakeSite.com", ""
        ]),
        "company_name": random.choice([
            "ABC IT Solutions Pvt Ltd",
            "XYZ Consultancy Services",
            "Tech Solutions Group",
            "ABC Software Solutions",
        ]),
        "scam_score": random.uniform(80, 99),
        "scam_risk_level": "Scam Likely",
        "company_trust_score": company_trust,
        "recruiter_verification_score": recruiter_verif,
    }


def generate_synthetic_dataset(num_scams: int = 50) -> list:
    return [generate_synthetic_scam(i) for i in range(num_scams)]


# ============================================================================
# SYNTHETIC LEGIT JOB GENERATION
# ============================================================================

LEGIT_TEMPLATES = [
    """
    {role} position open at {company}.
    Responsibilities:
    - Design, develop and maintain scalable systems
    - Collaborate with cross-functional teams
    - Mentor junior engineers and conduct code reviews
    Requirements:
    - {years}+ years of professional experience
    - Strong knowledge of Python, Java, or Go
    - Excellent problem-solving skills
    - Bachelor's or Master's degree in CS or related field
    Benefits:
    - Competitive salary and performance bonus
    - Health insurance for self and family
    - Provident fund and gratuity
    - ESOPs and stock options
    - Professional development budget and mentorship program
    Apply via our careers portal. Qualified candidates will be contacted for interview.
    """,
    """
    We are hiring an experienced {role} at {company}.
    What you will do:
    - Build production-grade software for millions of users
    - Drive technical decisions and architecture
    - Work with modern cloud infrastructure (AWS / GCP / Azure)
    What we look for:
    - {years}+ years experience in software development
    - Hands-on with distributed systems and microservices
    - Strong fundamentals in data structures and algorithms
    What we offer:
    - Industry-leading salary package
    - Comprehensive health insurance and wellness programs
    - Provident fund, ESOP, performance bonus
    - Hybrid work model and flexible hours
    - Continuous learning budget and training programs
    Interview process: 1 screening + 2 technical + 1 system design + 1 HR round.
    """,
    """
    {company} is hiring {role}s for our growing team.
    Job description:
    - Develop and ship features end-to-end
    - Participate in design and code reviews
    - Improve performance, reliability and quality of our products
    Required qualifications:
    - {years}+ years of relevant professional experience
    - Solid CS fundamentals and clean coding practices
    - Experience with SQL/NoSQL databases and REST APIs
    Compensation and benefits:
    - Best-in-industry salary
    - Comprehensive medical insurance, life and accident cover
    - Provident fund and gratuity
    - ESOPs / RSUs
    - Learning and development support
    Submit your application via our official careers page.
    """,
]

LEGIT_ROLES = [
    "Software Engineer", "Senior Software Engineer", "Backend Developer",
    "Frontend Developer", "Full Stack Developer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "SRE",
    "Product Manager", "Engineering Manager", "QA Engineer",
    "Mobile Developer", "Cloud Engineer", "Data Engineer",
]

LEGIT_COMPANIES = [
    ("Microsoft India", "microsoft.com"),
    ("Google India", "google.com"),
    ("Amazon Development Center", "amazon.com"),
    ("Flipkart Internet Pvt Ltd", "flipkart.com"),
    ("Razorpay", "razorpay.com"),
    ("Swiggy", "swiggy.in"),
    ("Zomato", "zomato.com"),
    ("PhonePe", "phonepe.com"),
    ("Freshworks", "freshworks.com"),
    ("Tata Consultancy Services", "tcs.com"),
    ("Infosys", "infosys.com"),
    ("Wipro", "wipro.com"),
]


def generate_synthetic_legit(num: int) -> dict:
    template = random.choice(LEGIT_TEMPLATES)
    role = random.choice(LEGIT_ROLES)
    company, domain = random.choice(LEGIT_COMPANIES)
    years = random.choice([2, 3, 4, 5, 6, 7, 8])

    description = template.format(role=role, company=company, years=years)

    # Mix lengths: 30% of legit jobs have short descriptions
    if random.random() < 0.3:
        description = f"Looking for a passionate {role} at {company}. Must have at least {years} years of professional experience in development. Apply via our portal."

    # Mix titles to vary title lengths (overlap with scam titles)
    title_style = random.random()
    if title_style < 0.4:
        title = role
    elif title_style < 0.7:
        title = f"{role} - {company}"
    else:
        title = f"{role} ({random.choice(['Remote', 'Hybrid', 'Full-time', 'Internship'])})"

    # Mix salaries: some are entry-level/internships (stipends), some are high LPA, some are 0/not disclosed
    salary_style = random.random()
    if salary_style < 0.4:
        # High LPA job
        salary_min = float(random.choice([600000, 1000000, 1500000, 2000000]))
        salary_max = float(salary_min + random.choice([200000, 400000, 600000]))
    elif salary_style < 0.8:
        # Internships / Entry level stipends (which is realistic for scraped postings)
        salary_min = float(random.choice([8000, 12000, 15000, 20000, 25000]))
        salary_max = float(salary_min + random.choice([0, 5000, 10000]))
    else:
        # Not disclosed
        salary_min = 0.0
        salary_max = 0.0

    # Mix trust scores to include realistic default scores (50 and 30)
    company_trust = random.choice([random.uniform(70, 95), 50.0, random.uniform(40, 75)])
    recruiter_verif = random.choice([random.uniform(65, 95), 30.0, random.uniform(35, 70)])

    return {
        "job_title": title,
        "job_description": description.strip(),
        "skills_required": random.sample(
            ["python", "java", "sql", "aws", "docker", "kubernetes",
             "react", "node.js", "spring", "postgresql", "redis", "kafka"],
            k=random.randint(4, 8)
        ) if random.random() > 0.1 else [],
        "skill_categories": {},
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_raw": f"{salary_min // 100000 if salary_min > 50000 else salary_min}-{salary_max // 100000 if salary_max > 50000 else salary_max}",
        "city": random.choice(["Bengaluru", "Hyderabad", "Pune", "Gurgaon", "Mumbai", "Chennai", "Noida"]),
        "state": random.choice(["Karnataka", "Telangana", "Maharashtra", "Haryana", "Tamil Nadu", "Uttar Pradesh"]),
        "country": "India",
        "mode": random.choice(["On-site", "Hybrid", "Remote"]),
        "platform_name": random.choice(["LinkedIn", "Naukri", "Indeed", "Shine"]),
        "company_name": company,
        "recruiter_email": f"careers@{domain}",
        "scam_score": random.uniform(0, 15),
        "scam_risk_level": "Safe",
        "company_trust_score": company_trust,
        "recruiter_verification_score": recruiter_verif,
    }


def generate_synthetic_legit_dataset(num_legit: int = 50) -> list:
    return [generate_synthetic_legit(i) for i in range(num_legit)]


# ============================================================================
# CROSS VALIDATION
# ============================================================================

def run_cross_validation(X, y) -> dict:
    """5-Fold Stratified Cross Validation."""
    print("\n" + "=" * 70)
    print("CROSS VALIDATION (5-Fold Stratified)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    rf_scores  = cross_val_score(rf_model,  X, y, cv=cv, scoring="f1_weighted")
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1_weighted")

    print(f"Random Forest CV F1: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    print(f"XGBoost       CV F1: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

    return {
        "rf_cv_f1_mean":  float(rf_scores.mean()),
        "rf_cv_f1_std":   float(rf_scores.std()),
        "xgb_cv_f1_mean": float(xgb_scores.mean()),
        "xgb_cv_f1_std":  float(xgb_scores.std()),
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    print("\nTOP 15 IMPORTANT FEATURES:")
    print(feature_importance.to_string(index=False))

    return model, accuracy, f1


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    n_safe = (y_train == 0).sum()
    n_scam = (y_train == 1).sum()
    scale_pos_weight = n_safe / max(1, n_scam)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC:  {auc:.4f}")
    except ValueError:
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, accuracy, f1


def train_isolation_forest(X_train) -> object:
    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST (Anomaly Detection)")
    print("=" * 70)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        max_samples='auto',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train)

    predictions = model.predict(X_train)
    n_anomalies = (predictions == -1).sum()
    print(f"\nDetected {n_anomalies}/{len(predictions)} anomalies "
          f"({n_anomalies/len(predictions):.1%})")

    return model


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("GRAPHURA ML TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load real jobs from Supabase
    print("\nStep 1: Loading real jobs from Supabase...")

    try:
        from ..scraper.storage.supabase_client import get_client
        sb = get_client()

        response = sb.table("jobs").select(
            "*, companies(company_trust_score), recruiters(recruiter_verification_score)"
        ).execute()

        real_jobs = response.data

        for job in real_jobs:
            if job.get("companies"):
                job["company_trust_score"] = job["companies"].get("company_trust_score", 50)
            if job.get("recruiters"):
                job["recruiter_verification_score"] = job["recruiters"].get("recruiter_verification_score", 30)

        print(f"   Loaded {len(real_jobs)} real jobs from Supabase")

    except Exception as e:
        print(f"   Could not load from Supabase: {e}")
        print(f"   Continuing with synthetic data only...")
        real_jobs = []

    # Step 2: Generate synthetic scam jobs
    print("\nStep 2: Generating synthetic scam jobs...")
    num_scams = max(500, len(real_jobs) * 2)
    synthetic_scams = generate_synthetic_dataset(num_scams)
    print(f"   Generated {len(synthetic_scams)} synthetic scam jobs")

    # Step 2b: Generate synthetic legit jobs
    print("\nStep 2b: Generating synthetic legit (safe) jobs...")
    num_legit_needed = max(500, num_scams - len(real_jobs))
    synthetic_legit = generate_synthetic_legit_dataset(num_legit_needed)
    print(f"   Generated {len(synthetic_legit)} synthetic legit jobs")

    # Step 3: Combine datasets
    all_jobs = real_jobs + synthetic_scams + synthetic_legit
    print(f"\nStep 3: Combined dataset")
    print(f"   Total jobs:      {len(all_jobs)}")
    print(f"   Real:            {len(real_jobs)}")
    print(f"   Synthetic scams: {len(synthetic_scams)}")
    print(f"   Synthetic legit: {len(synthetic_legit)}")

    # Step 4: Extract labels
    print("\nStep 4: Creating labels...")
    y_df = extract_labels(all_jobs)
    y = y_df["is_scam"].values
    print(f"   Labels: {(y == 0).sum()} safe, {(y == 1).sum()} scam")

    if len(np.unique(y)) < 2:
        print("\nERROR: Only one class found. Cannot train classifiers.")
        return

    # Step 5: Split RAW jobs (prevents data leakage)
    print("\nStep 5: Splitting raw jobs (80/20)...")
    jobs_train, jobs_test, y_train, y_test = train_test_split(
        all_jobs, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"   Train jobs: {len(jobs_train)}")
    print(f"   Test jobs:  {len(jobs_test)}")

    # Step 6: TF-IDF fit ONLY on training data
    print("\nStep 6: Extracting features (TF-IDF fit on train only)...")
    jobs_train_cleaned = [dict(job) for job in jobs_train]
    jobs_test_cleaned = [dict(job) for job in jobs_test]
    for job in jobs_train_cleaned:
        job["job_description"] = prepare_ml_text(job.get("job_description", "") or "")
    for job in jobs_test_cleaned:
        job["job_description"] = prepare_ml_text(job.get("job_description", "") or "")

    X_train = build_feature_dataframe(jobs_train_cleaned, fit_tfidf=True)
    X_test  = build_feature_dataframe(jobs_test_cleaned,  fit_tfidf=False)

    # Align test columns to match train columns
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]

    print(f"   Train Features: {X_train.shape}")
    print(f"   Test Features:  {X_test.shape}")

    # Step 7: Cross-validation on training data only
    cv_results = run_cross_validation(X_train, y_train)

    # Step 8: Fit scaler (saved for downstream use even if not applied here)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Step 9: Train all three models
    rf_model,  rf_acc,  rf_f1  = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(X_train, y_train, X_test, y_test)
    iso_model                  = train_isolation_forest(X_train)

    # Step 10: Save models
    print("\nStep 10: Saving trained models...")
    joblib.dump(rf_model,  MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(iso_model, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler,    MODELS_DIR / "scaler.pkl")

    feature_columns = list(X_train.columns)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    metadata = {
        "models": {
            "random_forest": {
                "accuracy": float(rf_acc),
                "f1_score": float(rf_f1),
                "path": str(MODELS_DIR / "random_forest.pkl")
            },
            "xgboost": {
                "accuracy": float(xgb_acc),
                "f1_score": float(xgb_f1),
                "path": str(MODELS_DIR / "xgboost.pkl")
            },
            "isolation_forest": {
                "path": str(MODELS_DIR / "isolation_forest.pkl")
            }
        },
        "cross_validation": cv_results,
        "training_data": {
            "total_jobs":      len(all_jobs),
            "real_jobs":       len(real_jobs),
            "synthetic_scams": len(synthetic_scams),
            "synthetic_legit": len(synthetic_legit),
            "features":        len(feature_columns)
        },
        "trained_at": pd.Timestamp.now().isoformat()
    }

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved: random_forest.pkl, xgboost.pkl, isolation_forest.pkl")
    print(f"   Saved: scaler.pkl, feature_columns.json, metadata.json")




if __name__ == "__main__":
    main()