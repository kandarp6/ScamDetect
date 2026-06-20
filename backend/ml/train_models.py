import sys

# DEPENDENCY VALIDATION
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "nltk": "nltk",
    "joblib": "joblib",
    "spacy": "spacy",
    "textstat": "textstat",
    "sentence_transformers": "sentence-transformers",
    "shap": "shap"
}

missing_deps = []
for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_deps.append((module_name, pip_name))

if missing_deps:
    for module_name, pip_name in missing_deps:
        print(f"Missing dependency: {pip_name}")
        print(f"Run:")
        print(f"pip install {pip_name}\n")
    sys.exit(1)

# NLTK AUTO DOWNLOAD logic
try:
    import nltk
    for resource in ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            res_name = resource.split("/")[-1]
            print(f"Downloading missing NLTK resource: {res_name}...")
            nltk.download(res_name, quiet=True)
except Exception as e:
    print(f"Warning: NLTK initialization failed: {e}")

import json
import random
import re
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
    load_dataset_from_csv,
)
from .nlp_engine import prepare_ml_text

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATA_CSV_PATH = Path(__file__).parent / "data" / "jobs_with_extracted_skills.csv"

LABEL_NOISE_RATE = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_salary_string(raw) -> tuple:
    if not isinstance(raw, str):
        return 0.0, 0.0

    text = raw.strip()
    if text == "" or text.lower() in ("not mentioned", "unpaid", "n/a", "na"):
        return 0.0, 0.0

    is_lacs = bool(re.search(r"lac|lakh", text, re.IGNORECASE))

    numbers = re.findall(r"\d[\d,]*\.?\d*", text)
    values = []
    for n in numbers:
        try:
            values.append(float(n.replace(",", "")))
        except ValueError:
            continue

    if not values:
        return 0.0, 0.0

    if is_lacs:
        values = [v * 100000 for v in values]

    if len(values) == 1:
        return values[0], values[0]

    lo, hi = values[0], values[1]
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def derive_employment_mode(job_type, description) -> str:
    text = f"{job_type or ''} {description or ''}".lower()
    if "work from home" in text or "remote" in text:
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    return "On-site"


def map_recruiter_verification(label) -> tuple:
    if pd.isna(label):
        return 50.0, 30.0

    label = str(label).strip()
    if label == "Verified":
        return random.uniform(70, 90), random.uniform(75, 95)
    if label == "Unverified":
        return random.uniform(35, 55), random.uniform(20, 40)
    if label == "Suspicious":
        return random.uniform(10, 25), random.uniform(5, 20)
    return 50.0, 30.0


def load_real_jobs_from_csv(csv_path: Path) -> list:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"   Real data CSV not found at {csv_path}, skipping.")
        return []

    df = pd.read_csv(csv_path)
    jobs = []

    for _, row in df.iterrows():
        title = str(row.get("title") or "").strip()
        description = str(row.get("description") or "").strip()
        if not title and not description:
            continue

        salary_min, salary_max = parse_salary_string(row.get("salary"))

        skills_raw = row.get("Skills")
        if isinstance(skills_raw, str) and skills_raw.strip():
            skills = sorted(set(s.strip().lower() for s in skills_raw.split(",") if s.strip()))
        else:
            skills = []

        location = str(row.get("location") or "").strip()
        loc_parts = [p.strip() for p in location.split(",") if p.strip()]
        city = loc_parts[0] if loc_parts else ""
        state = loc_parts[1] if len(loc_parts) > 1 else ""

        is_flagged = str(row.get("is_flagged")).strip().lower() == "true"
        scam_score_val = row.get("scam_score")
        scam_score_val = float(scam_score_val) if pd.notna(scam_score_val) else 0.0
        is_scam_real = is_flagged or scam_score_val >= 50

        company_trust, recruiter_verif = map_recruiter_verification(row.get("recruiter_verified"))

        jobs.append({
            "job_title": title or "Untitled Position",
            "job_description": description,
            "skills_required": skills,
            "skill_categories": {},
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_raw": str(row.get("salary") or ""),
            "city": city,
            "state": state,
            "country": "India",
            "mode": derive_employment_mode(row.get("job_type"), description),
            "platform_name": str(row.get("posting_platform") or row.get("source") or "Unknown"),
            "company_name": str(row.get("company") or "Unknown"),
            "scam_score": scam_score_val if is_scam_real else min(scam_score_val, 15.0),
            "scam_risk_level": "Scam Likely" if is_scam_real else "Safe",
            "company_trust_score": company_trust,
            "recruiter_verification_score": recruiter_verif,
        })

    n_scam = sum(1 for j in jobs if j["scam_risk_level"] == "Scam Likely")
    print(f"   Parsed {len(jobs)} real jobs from CSV ({n_scam} flagged scam, {len(jobs) - n_scam} legit)")
    return jobs


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

SCAM_TEMPLATES_SUBTLE = [
    """
    {role} - {company_like} is expanding its team and hiring immediately.
    We are looking for motivated candidates who can start right away.
    Compensation: Rs {amount} per {period}, based on performance.
    A one-time refundable registration fee of Rs {fee} applies to confirm onboarding.
    Interested candidates can share their details by email.
    """,
    """
    Hiring {role} for a fast-growing business. Flexible hours, work from anywhere.
    Stipend of Rs {amount} during the initial training period.
    Please complete the verification process by paying Rs {fee} (refundable on joining).
    No prior experience required, full training provided.
    """,
    """
    {company_like} is looking for a {role} to join our growing team.
    Day-to-day responsibilities include client communication and basic reporting.
    Selected candidates will be asked to pay a nominal Rs {fee} for ID card and kit processing.
    Salary: Rs {amount} per {period}. Apply by sending your details by email.
    """,
    """
    Immediate opening for {role} at {company_like}.
    Great opportunity for freshers. No interview required for early applicants.
    A refundable security amount of Rs {fee} secures your training slot.
    Expected earnings: Rs {amount} per {period} after the training window.
    """,
]

SCAM_TEMPLATES_STEALTH = [
    """
    {role} opening at {company_like}. We are streamlining our hiring process
    to get candidates started quickly. Once shortlisted, a one-time processing
    charge of Rs {fee} is collected to cover background verification and is
    adjusted against your first paycheck. Share your CV by email and our team
    will reach out within a few days.
    """,
    """
    {company_like} is onboarding new {role}s this month. Selected applicants
    will need to complete a paid orientation session (Rs {fee}) before
    receiving their offer letter; this amount is later included as part of
    the first month's compensation. Apply by replying with your resume.
    """,
    """
    We have an opening for {role} at {company_like}. Expected take-home:
    Rs {amount} per {period}. To finalize your placement, candidates cover
    a small administrative charge of Rs {fee} for documentation, refunded
    after three months of employment. Send your resume to proceed.
    """,
    """
    {role} position at {company_like}, fully remote, flexible hours.
    Take-home pay of Rs {amount} per {period} regardless of prior background.
    Apply by sending your resume and basic details over email; our team will
    follow up with next steps.
    """,
]

GENERIC_SCAM_COMPANY_NAMES = [
    "ABC IT Solutions Pvt Ltd", "XYZ Consultancy Services", "Tech Solutions Group",
    "ABC Software Solutions", "Bright Path Careers", "NextGen Talent Solutions",
    "Horizon Business Services", "Skyline Consulting Group", "Prime Career Network",
    "Elevate HR Solutions",
]

SCAM_ROLES = [
    "Data Entry Operator", "Customer Support", "Marketing Executive",
    "Business Development", "Telecaller", "Receptionist",
    "Computer Operator", "Office Assistant", "Sales Representative",
]


def soften_scam_text(description: str, strip_probability: float = 0.35) -> str:
    obvious_markers = ("whatsapp", "telegram", "urgent", "limited seats", "hurry", "immediate joining")
    kept_lines = []
    for line in description.split("\n"):
        if any(m in line.lower() for m in obvious_markers) and random.random() < strip_probability:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def generate_synthetic_scam(num: int) -> dict:
    pool_roll = random.random()
    if pool_roll < 0.30:
        pool, template = "overt", random.choice(SCAM_TEMPLATES)
    elif pool_roll < 0.65:
        pool, template = "subtle", random.choice(SCAM_TEMPLATES_SUBTLE)
    else:
        pool, template = "stealth", random.choice(SCAM_TEMPLATES_STEALTH)

    role = random.choice(SCAM_ROLES)
    company_like = random.choice(GENERIC_SCAM_COMPANY_NAMES)

    description = template.format(
        role=role,
        company_like=company_like,
        amount=random.choice([
            "15,000", "20,000", "25,000",
            "50,000", "1,00,000", "75,000", "2,00,000",
        ]),
        period=random.choice(["daily", "weekly", "per day", "monthly", "per month"]),
        fee=random.choice(["500", "999", "1500", "2500", "5000"]),
        phone=random.randint(9000000000, 9999999999),
        num=num,
    )

    if pool == "overt":
        description = soften_scam_text(description)

    if random.random() < 0.3:
        filler = "\nResponsibilities include designing marketing materials, communicating with potential clients, and reporting to the director. Requirements: basic computer knowledge, internet access, and ability to follow instructions."
        description += filler

    skills = []
    if random.random() < 0.4:
        skills = random.sample(["Data Entry", "Excel", "Typing", "Communication"], k=random.randint(1, 3))

    company_trust = random.choice([random.uniform(5, 30), 50.0, random.uniform(20, 65)])
    recruiter_verif = random.choice([random.uniform(5, 30), 30.0, random.uniform(20, 55)])

    title_style = random.random()
    if title_style < 0.5:
        title = f"{role} - Urgent Hiring!!!"
    elif title_style < 0.8:
        title = role
    else:
        title = f"{role} (Work From Home)"

    salary_style = random.random()
    if salary_style < 0.4:
        salary_min = 0.0
        salary_max = 0.0
    elif salary_style < 0.7:
        salary_min = float(random.choice([10000, 15000, 20000]))
        salary_max = float(salary_min + random.choice([5000, 10000, 15000]))
    else:
        salary_min = 0.0
        salary_max = random.choice([50000000.0, 100000000.0])

    mode = random.choices(["Remote", "On-site", "Hybrid"], weights=[0.55, 0.30, 0.15])[0]
    platform_name = random.choices(
        ["Unknown", "Random Site", "FakeSite.com", "", "Naukri", "Internshala", "Indeed", "Apna"],
        weights=[0.30, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05],
    )[0]
    city = random.choice(["Remote", "Anywhere", "Pan India", "", "Mumbai", "Bengaluru", "Delhi", "Kolkata"])

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
        "city": city,
        "state": "",
        "country": "India",
        "mode": mode,
        "platform_name": platform_name,
        "company_name": company_like,
        "scam_score": random.uniform(45, 95),
        "scam_risk_level": "Scam Likely",
        "company_trust_score": company_trust,
        "recruiter_verification_score": recruiter_verif,
    }


def generate_synthetic_dataset(num_scams: int = 50) -> list:
    return [generate_synthetic_scam(i) for i in range(num_scams)]


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

LEGIT_FALSE_POSITIVE_SNIPPETS = [
    "Limited seats available for this batch - apply early to secure your spot in the program.",
    "Every new hire is guaranteed a dedicated mentor and a structured onboarding process.",
    "No experience necessary for this role - we welcome freshers and career switchers, with full training provided.",
    "A refundable equipment deposit of Rs 2,000 is required for company laptop issuance and is returned in full when the laptop is returned.",
    "Top performers can earn weekly recognition bonuses on top of their fixed salary.",
    "Urgent hiring for our growing support team ahead of the holiday season rush.",
    "Our recruiter will follow up over WhatsApp to schedule your interview after the initial screening call.",
]


def generate_synthetic_legit(num: int) -> dict:
    template = random.choice(LEGIT_TEMPLATES)
    role = random.choice(LEGIT_ROLES)
    company, domain = random.choice(LEGIT_COMPANIES)
    years = random.choice([2, 3, 4, 5, 6, 7, 8])

    description = template.format(role=role, company=company, years=years)

    if random.random() < 0.3:
        description = f"Looking for a passionate {role} at {company}. Must have at least {years} years of professional experience in development. Apply via our portal."

    title_style = random.random()
    if title_style < 0.4:
        title = role
    elif title_style < 0.7:
        title = f"{role} - {company}"
    else:
        title = f"{role} ({random.choice(['Remote', 'Hybrid', 'Full-time', 'Internship'])})"

    salary_style = random.random()
    if salary_style < 0.4:
        salary_min = float(random.choice([600000, 1000000, 1500000, 2000000]))
        salary_max = float(salary_min + random.choice([200000, 400000, 600000]))
    elif salary_style < 0.8:
        salary_min = float(random.choice([8000, 12000, 15000, 20000, 25000]))
        salary_max = float(salary_min + random.choice([0, 5000, 10000]))
    else:
        salary_min = 0.0
        salary_max = 0.0

    company_trust = random.choice([random.uniform(55, 95), 50.0, random.uniform(35, 75)])
    recruiter_verif = random.choice([random.uniform(60, 95), 30.0, random.uniform(35, 70)])

    if random.random() < 0.18:
        description = description.strip() + "\n" + random.choice(LEGIT_FALSE_POSITIVE_SNIPPETS)

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
        "scam_score": random.uniform(0, 50),
        "scam_risk_level": "Safe",
        "company_trust_score": company_trust,
        "recruiter_verification_score": recruiter_verif,
    }


def generate_synthetic_legit_dataset(num_legit: int = 50) -> list:
    return [generate_synthetic_legit(i) for i in range(num_legit)]


def apply_label_noise(y: np.ndarray, noise_rate: float = LABEL_NOISE_RATE, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y_noisy) * noise_rate)
    if n_flip == 0:
        return y_noisy
    flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def run_cross_validation(X, y) -> dict:
    print("\n" + "=" * 70)
    print("CROSS VALIDATION (Dynamic-Fold Stratified)")
    print("=" * 70)

    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts) if len(class_counts) > 0 else 0
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        print(f"Skipping cross-validation: too few samples per class (min class count = {min_class_count})")
        return {
            "rf_cv_f1_mean": 0.0,
            "rf_cv_f1_std": 0.0,
            "xgb_cv_f1_mean": 0.0,
            "xgb_cv_f1_std": 0.0,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    rf_model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring="f1_weighted")
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1_weighted")

    print(f"Random Forest CV F1: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    print(f"XGBoost       CV F1: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

    return {
        "rf_cv_f1_mean": float(rf_scores.mean()),
        "rf_cv_f1_std": float(rf_scores.std()),
        "xgb_cv_f1_mean": float(xgb_scores.mean()),
        "xgb_cv_f1_std": float(xgb_scores.std()),
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

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
        n_estimators=20,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

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
        max_samples="auto",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train)

    predictions = model.predict(X_train)
    n_anomalies = (predictions == -1).sum()
    print(f"\nDetected {n_anomalies}/{len(predictions)} anomalies "
          f"({n_anomalies/len(predictions):.1%})")

    return model


def main():
    print("=" * 70)
    print("GRAPHURA ML TRAINING PIPELINE")
    print("=" * 70)

    print("\nStep 1: Loading jobs from local CSV dataset...")
    import sys
    csv_path = Path(__file__).parent.parent.parent / "data" / "training" / "processed_cleaned_data.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "data" / "processed_cleaned_data.csv"
        
    if not csv_path.exists():
        print(f"ERROR: Dataset file is missing. Please place the dataset at: {csv_path}")
        sys.exit(1)
        
    try:
        all_jobs = load_dataset_from_csv(str(csv_path))
        print(f"   Loaded {len(all_jobs)} jobs from {csv_path.name}")
    except ValueError as val_err:
        print(f"VALIDATION ERROR: {val_err}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)
        
    if not all_jobs:
        print("ERROR: The loaded dataset is empty.")
        sys.exit(1)

    print("\nCreating labels...")
    y_df = extract_labels(all_jobs)
    y = y_df["is_scam"].values
    print(f"   Labels: {(y == 0).sum()} safe, {(y == 1).sum()} scam")

    if len(np.unique(y)) < 2:
        print("\nERROR: Only one class found. Cannot train classifiers.")
        return

    print("\nSplitting raw jobs (80/20)...")
    jobs_train, jobs_test, y_train, y_test = train_test_split(
        all_jobs, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"   Train jobs: {len(jobs_train)}")
    print(f"   Test jobs:  {len(jobs_test)}")

    # Keep a copy of original labels for saving features.csv and labels.csv later
    y_train_orig = y_train.copy()
    y_test_orig = y_test.copy()

    n_flipped_train = int(len(y_train) * LABEL_NOISE_RATE)
    y_train = apply_label_noise(y_train, noise_rate=LABEL_NOISE_RATE, seed=RANDOM_SEED)
    y_test = apply_label_noise(y_test, noise_rate=LABEL_NOISE_RATE, seed=RANDOM_SEED + 1)
    print(f"   Applied {LABEL_NOISE_RATE:.0%} label noise to training labels "
          f"({n_flipped_train} flipped) and testing labels to simulate real-world noise")

    print("\nExtracting features (TF-IDF fit on train only)...")
    jobs_train_cleaned = [dict(job) for job in jobs_train]
    jobs_test_cleaned = [dict(job) for job in jobs_test]
    for job in jobs_train_cleaned:
        job["job_description"] = prepare_ml_text(job.get("job_description", "") or "")
    for job in jobs_test_cleaned:
        job["job_description"] = prepare_ml_text(job.get("job_description", "") or "")

    X_train = build_feature_dataframe(jobs_train_cleaned, fit_tfidf=True)
    X_test = build_feature_dataframe(jobs_test_cleaned, fit_tfidf=False)

    missing_test_cols = {col: 0 for col in X_train.columns if col not in X_test.columns}
    if missing_test_cols:
        X_test = pd.concat([X_test, pd.DataFrame(missing_test_cols, index=X_test.index)], axis=1)
    X_test = X_test[X_train.columns]

    print(f"   Train Features: {X_train.shape}")
    print(f"   Test Features:  {X_test.shape}")

    cv_results = run_cross_validation(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    rf_model, rf_acc, rf_f1 = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(X_train, y_train, X_test, y_test)
    iso_model = train_isolation_forest(X_train)

    print("\nSaving trained models...")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(iso_model, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Download/save SentenceTransformer embedding model locally if not already present
    embedding_model_path = MODELS_DIR / "embedding_model"
    if not embedding_model_path.exists():
        print("Saving SentenceTransformer embedding model locally...")
        try:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_model.save(str(embedding_model_path))
        except Exception as e:
            print(f"Warning: Could not save SentenceTransformer to {embedding_model_path}: {e}")
    else:
        print("SentenceTransformer embedding model is already saved locally.")

    # Save features.csv and labels.csv for the training dataset
    print("Saving features.csv and labels.csv...")
    df_features_all = pd.concat([X_train, X_test], ignore_index=True)
    df_labels_all = pd.DataFrame({"is_scam": np.concatenate([y_train_orig, y_test_orig])})
    
    df_features_all.to_csv(MODELS_DIR / "features.csv", index=False)
    df_labels_all.to_csv(MODELS_DIR / "labels.csv", index=False)

    # Save matched_keywords.csv
    print("Saving matched_keywords.csv...")
    from .nlp_engine import get_matched_keywords
    matched_keywords_records = []
    for job in all_jobs:
        keywords = get_matched_keywords(job.get("job_description", "") or "")
        matched_keywords_records.append({
            "job_title": job.get("job_title", "Unknown"),
            "matched_keywords": ", ".join(keywords),
            "keyword_count": len(keywords)
        })
    pd.DataFrame(matched_keywords_records).to_csv(MODELS_DIR / "matched_keywords.csv", index=False)

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
            "total_jobs": len(all_jobs),
            "features": len(feature_columns)
        },
        "trained_at": pd.Timestamp.now().isoformat()
    }

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved: random_forest.pkl, xgboost.pkl, isolation_forest.pkl")
    print(f"   Saved: scaler.pkl, feature_columns.json, metadata.json")
    print(f"   Saved: features.csv, labels.csv, matched_keywords.csv, domain_analysis.csv")


if __name__ == "__main__":
    main()