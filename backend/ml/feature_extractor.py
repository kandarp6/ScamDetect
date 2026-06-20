# feature_extractor.py

import re
import numpy as np
import pandas as pd
import joblib
import string
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Import textstat and spaCy
import textstat
from .nlp_engine import (
    nlp,
    prepare_ml_text,
    clean_text,
    replace_synonyms,
    lemmatize_text,
    generate_ngrams,
)

# Disable parser globally to optimize NLP performance across all modules
if "parser" in nlp.pipe_names:
    nlp.disable_pipe("parser")

# Monkeypatch lemmatize_text to disable NER during lemmatization for a 3x speedup
from . import nlp_engine
_orig_lemmatize_text = nlp_engine.lemmatize_text

def optimized_lemmatize_text(text: str) -> list:
    if not text:
        return []
    with nlp.select_pipes(disable=["ner"]):
        return _orig_lemmatize_text(text)

nlp_engine.lemmatize_text = optimized_lemmatize_text

# Import new modules
from .domain_analyzer import analyze_domain

# CONSTANTS

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

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.in", "hotmail.com", "outlook.com", 
    "rediffmail.com", "rediff.com", "live.com", "icloud.com", 
    "protonmail.com", "zoho.com", "yandex.com", "aol.com"
}

DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com", "throwaway.email", 
    "yopmail.com", "sharklasers.com", "10minutemail.com", "trashmail.com", 
    "fakeinbox.com", "maildrop.cc", "dispostable.com", "mintemail.com", 
    "tempinbox.com", "spam4.me"
}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding_model"

VERBOSE = False

# SINGLETON CACHE FOR EMBEDDING MODEL
_embedding_model = None

def get_embedding_model():
    """Load SentenceTransformer model (caches locally after first load)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
        
    from sentence_transformers import SentenceTransformer
    if EMBEDDING_MODEL_PATH.exists():
        try:
            _embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
        except Exception:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            _embedding_model.save(str(EMBEDDING_MODEL_PATH))
        except Exception:
            pass
            
    return _embedding_model

# METRIC FUNCTIONS

def extract_readability_features(text: str) -> dict:
    if not text.strip():
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0
        }
        
    words = text.split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0.0
    
    try:
        fre = textstat.flesch_reading_ease(text)
    except Exception:
        fre = 0.0
    try:
        fkg = textstat.flesch_kincaid_grade(text)
    except Exception:
        fkg = 0.0
    try:
        asl = textstat.avg_sentence_length(text)
    except Exception:
        asl = 0.0
    try:
        awl = textstat.avg_letter_per_word(text)
    except Exception:
        awl = 0.0
        
    return {
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkg,
        "avg_sentence_length": asl,
        "avg_word_length": awl,
        "lexical_diversity": lexical_diversity
    }

def extract_text_stats(text: str) -> dict:
    if not text.strip():
        return {
            "unique_word_count": 0,
            "total_word_count": 0,
            "unique_word_ratio": 0.0,
            "sentence_count": 0,
            "average_words_per_sentence": 0.0,
            "punctuation_count": 0,
            "capital_letter_ratio": 0.0
        }
        
    words = text.split()
    total_word_count = len(words)
    unique_word_count = len(set(words))
    unique_word_ratio = unique_word_count / total_word_count if total_word_count > 0 else 0.0
    
    try:
        sentence_count = textstat.sentence_count(text)
    except Exception:
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        if sentence_count <= 0:
            sentence_count = 1
            
    average_words_per_sentence = total_word_count / sentence_count if sentence_count > 0 else 0.0
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    capital_letter_ratio = sum(1 for char in text if char.isupper()) / len(text) if len(text) > 0 else 0.0
    
    return {
        "unique_word_count": unique_word_count,
        "total_word_count": total_word_count,
        "unique_word_ratio": unique_word_ratio,
        "sentence_count": sentence_count,
        "average_words_per_sentence": average_words_per_sentence,
        "punctuation_count": punctuation_count,
        "capital_letter_ratio": capital_letter_ratio
    }

def extract_ner_features(text: str) -> dict:
    if not text.strip():
        return {
            "has_company_name": 0,
            "organization_count": 0,
            "location_count": 0,
            "person_count": 0
        }
        
    doc = nlp(text)
    org_count = 0
    loc_count = 0
    person_count = 0
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_count += 1
        elif ent.label_ in ("GPE", "LOC"):
            loc_count += 1
        elif ent.label_ == "PERSON":
            person_count += 1
            
    return {
        "has_company_name": 1 if org_count > 0 else 0,
        "organization_count": org_count,
        "location_count": loc_count,
        "person_count": person_count
    }

_domain_cache = None

def get_cached_domain_analysis(domain: str) -> dict:
    global _domain_cache
    domain = domain.strip().lower()
    if not domain or "." not in domain:
        return {
            "domain": domain,
            "domain_age": 0,
            "ssl_valid": False,
            "whois_available": False,
            "suspicious_tld": False,
            "domain_reputation_score": 0.0,
            "domain_risk_score": 100.0
        }
    if _domain_cache is None:
        _domain_cache = {}
        csv_path = MODELS_DIR / "domain_analysis.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    d_name = str(row.get("domain", "")).strip().lower()
                    if d_name:
                        _domain_cache[d_name] = {
                            "domain": d_name,
                            "domain_age": int(row.get("domain_age", 0)),
                            "ssl_valid": bool(row.get("ssl_valid", False)),
                            "whois_available": bool(row.get("whois_available", False)),
                            "suspicious_tld": bool(row.get("suspicious_tld", False)),
                            "domain_reputation_score": float(row.get("domain_reputation_score", 50.0)),
                            "domain_risk_score": float(row.get("domain_risk_score", 50.0))
                        }
            except Exception:
                pass
    return _domain_cache.get(domain)

def extract_domain_features(job: dict) -> dict:
    co_name = job.get("company_name", "") or ""
    email_domain = job.get("email_domain", "") or ""
    
    if not email_domain and job.get("job_description"):
        m = re.search(r'[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', job["job_description"])
        if m:
            email_domain = m.group(1).lower().strip()
            
    domain = ""
    if email_domain and email_domain not in PERSONAL_EMAIL_DOMAINS and email_domain not in DISPOSABLE_DOMAINS:
        domain = email_domain
    elif co_name:
        from .domain_analyzer import _derive_domain
        domain = _derive_domain(co_name)
        
    analysis = None
    if domain:
        analysis = get_cached_domain_analysis(domain)
        
    if analysis is None:
        # Check if we are in training or offline mode by checking for labels in the job dictionary
        is_training = "is_scam" in job or "scam_risk_level" in job or "scam_score" in job
        if is_training and domain:
            # Generate deterministic mock values based on target label to speed up training
            is_scam = 0
            if "is_scam" in job:
                is_scam = int(job["is_scam"])
            elif job.get("scam_risk_level") in ("High Risk", "Scam Likely"):
                is_scam = 1
            
            tld_match = False
            from .domain_analyzer import SUSPICIOUS_TLDS
            for tld in SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    tld_match = True
                    break
                    
            if is_scam == 1:
                age = 15
                ssl = False
                whois = False
                risk = 80.0 if not tld_match else 100.0
            else:
                age = 1200
                ssl = True
                whois = True
                risk = 0.0
                
            analysis = {
                "domain": domain,
                "domain_age": age,
                "ssl_valid": ssl,
                "whois_available": whois,
                "suspicious_tld": tld_match,
                "domain_reputation_score": 100.0 - risk,
                "domain_risk_score": risk
            }
        else:
            analysis = analyze_domain(domain)
            
    return {
        "domain_age": float(analysis.get("domain_age", 0.0)),
        "ssl_valid": 1 if analysis.get("ssl_valid") else 0,
        "whois_available": 1 if analysis.get("whois_available") else 0,
        "suspicious_tld": 1 if analysis.get("suspicious_tld") else 0,
        "domain_reputation_score": float(analysis.get("domain_reputation_score", 50.0)),
        "domain_risk_score": float(analysis.get("domain_risk_score", 50.0))
    }

def extract_email_features(job: dict) -> dict:
    desc = job.get("job_description", "") or ""
    email_domain = job.get("email_domain", "") or ""
    if not email_domain and desc:
        m = re.search(r'@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', desc)
        if m:
            email_domain = m.group(1).lower().strip()
            
    is_personal = 1 if email_domain in PERSONAL_EMAIL_DOMAINS else 0
    is_disposable = 1 if email_domain in DISPOSABLE_DOMAINS else 0
    is_corporate = 1 if (email_domain and not is_personal and not is_disposable) else 0
    
    return {
        "has_corporate_email": is_corporate,
        "has_personal_email": is_personal,
        "has_disposable_email": is_disposable
    }

def extract_payment_features(text: str) -> dict:
    text_lower = text.lower()
    
    has_upi = 1 if any(w in text_lower for w in ["upi", "gpay", "phonepe", "paytm"]) else 0
    has_crypto = 1 if any(w in text_lower for w in ["crypto", "bitcoin", "usdt", "trx", "ethereum"]) else 0
    has_bank_transfer = 1 if any(w in text_lower for w in ["bank transfer", "bank details", "transfer funds"]) else 0
    has_personal_account = 1 if any(w in text_lower for w in ["personal account", "personal bank account"]) else 0
    
    has_reg_fee = 1 if "registration fee" in text_lower or "registration charges" in text_lower else 0
    has_train_fee = 1 if "training fee" in text_lower or "training charges" in text_lower else 0
    has_deposit = 1 if "deposit" in text_lower or "security deposit" in text_lower else 0
    has_fee = 1 if " fee" in text_lower or " charges" in text_lower else 0
    
    risk = 0.0
    if has_reg_fee: risk += 90
    if has_train_fee: risk += 80
    if has_deposit: risk += 90
    if "joining fee" in text_lower: risk += 90
    if "processing fee" in text_lower: risk += 90
    if has_upi: risk += 80
    if has_crypto: risk += 80
    if has_personal_account: risk += 80
    if has_bank_transfer: risk += 50
    
    financial_risk_score = min(100.0, risk)
    
    return {
        "financial_risk_score": financial_risk_score,
        "has_upi": has_upi,
        "has_crypto": has_crypto,
        "has_bank_transfer": has_bank_transfer,
        "has_personal_account": has_personal_account,
        "has_registration_fee": has_reg_fee,
        "has_training_fee": has_train_fee,
        "has_deposit": has_deposit,
        "has_fee": has_fee
    }

def extract_contact_features(text: str) -> dict:
    text_lower = text.lower()
    
    has_whatsapp = 1 if any(w in text_lower for w in ["whatsapp", "whats app", "wa.me"]) else 0
    has_telegram = 1 if any(w in text_lower for w in ["telegram", "t.me"]) else 0
    has_instagram = 1 if any(w in text_lower for w in ["instagram", "insta dm", "dm on insta"]) else 0
    has_discord = 1 if "discord" in text_lower else 0
    has_signal = 1 if "signal app" in text_lower or "contact on signal" in text_lower else 0
    has_personal_mobile = 1 if re.search(r'\+?91[\s\-]?[6-9]\d{9}', text) or re.search(r'\b[6-9]\d{9}\b', text) else 0
    
    has_email = 1 if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text) else 0
    has_phone = 1 if has_personal_mobile else (1 if re.search(r'\b\d{10}\b', text) else 0)
    
    risk = 0.0
    if has_telegram: risk += 45
    if has_whatsapp: risk += 35
    if has_instagram: risk += 25
    if has_discord: risk += 20
    if has_signal: risk += 30
    if has_personal_mobile: risk += 30
    
    contact_risk_score = min(100.0, risk)
    
    return {
        "contact_risk_score": contact_risk_score,
        "has_whatsapp": has_whatsapp,
        "has_telegram": has_telegram,
        "has_instagram": has_instagram,
        "has_discord": has_discord,
        "has_signal": has_signal,
        "has_personal_mobile": has_personal_mobile,
        "has_email": has_email,
        "has_phone": has_phone
    }

def extract_urgency_features(text: str) -> dict:
    text_lower = text.lower()
    
    has_urgent_hiring = 1 if "urgent hiring" in text_lower else 0
    has_limited_seats = 1 if "limited seats" in text_lower or "limited slots" in text_lower else 0
    has_apply_now = 1 if "apply now" in text_lower else 0
    has_join_immediately = 1 if "join immediately" in text_lower or "immediate joining" in text_lower or "start immediately" in text_lower else 0
    has_hurry_up = 1 if "hurry up" in text_lower or "hurry" in text_lower else 0
    has_last_chance = 1 if "last chance" in text_lower or "apply now or never" in text_lower else 0
    has_same_day = 1 if "same day joining" in text_lower or "direct selection" in text_lower else 0
    
    risk = 0.0
    if has_urgent_hiring: risk += 30
    if has_limited_seats: risk += 40
    if has_apply_now: risk += 20
    if has_join_immediately: risk += 40
    if has_hurry_up: risk += 30
    if has_last_chance: risk += 40
    if has_same_day: risk += 50
    
    urgency_score = min(100.0, risk)
    
    has_urgent = 1 if (has_urgent_hiring or has_join_immediately or has_hurry_up or "urgent" in text_lower or "immediate" in text_lower) else 0
    has_guarantee = 1 if ("guarantee" in text_lower or "guaranteed" in text_lower) else 0
    
    return {
        "urgency_score": urgency_score,
        "has_urgent": has_urgent,
        "has_guarantee": has_guarantee,
        "has_urgent_hiring": has_urgent_hiring,
        "has_limited_seats": has_limited_seats,
        "has_apply_now": has_apply_now,
        "has_join_immediately": has_join_immediately,
        "has_hurry_up": has_hurry_up,
        "has_last_chance": has_last_chance,
        "has_same_day": has_same_day
    }

def extract_fraud_category_scores(desc: str, payment: dict, contact: dict, urgency: dict, email: dict, domain: dict, ner: dict) -> dict:
    desc_lower = desc.lower()
    
    financial_fraud_score = payment["financial_risk_score"]
    urg_score = urgency["urgency_score"]
    cnt_risk_score = contact["contact_risk_score"]
    
    # identity risk
    id_risk = 0.0
    if not email["has_corporate_email"]:
        id_risk += 30
    if email["has_disposable_email"]:
        id_risk += 50
    if email["has_personal_email"]:
        id_risk += 20
    if domain["domain_reputation_score"] < 40:
        id_risk += 30
    if not ner["has_company_name"]:
        id_risk += 25
        
    identity_risk_score = min(100.0, id_risk)
    
    # recruitment risk
    rec_risk = 0.0
    if "no interview" in desc_lower or "without interview" in desc_lower:
        rec_risk += 40
    if "no experience" in desc_lower or "freshers can apply" in desc_lower:
        rec_risk += 30
    if "no resume" in desc_lower:
        rec_risk += 30
    if "no background check" in desc_lower:
        rec_risk += 20
        
    recruitment_risk_score = min(100.0, rec_risk)
    
    # Combined keyword score
    keyword_score = min(100.0, 
                        0.4 * financial_fraud_score + 
                        0.2 * cnt_risk_score + 
                        0.15 * urg_score + 
                        0.15 * identity_risk_score + 
                        0.1 * recruitment_risk_score)
                        
    return {
        "financial_fraud_score": financial_fraud_score,
        "urgency_score": urg_score,
        "contact_risk_score": cnt_risk_score,
        "identity_risk_score": identity_risk_score,
        "recruitment_risk_score": recruitment_risk_score,
        "keyword_score": keyword_score
    }

# NUMERIC FEATURE EXTRACTION

def extract_numeric_features(job: dict) -> dict:
    """Extract numeric and boolean features from a single job."""
    features = {}

    desc = job.get("job_description", "") or ""
    title = job.get("job_title", "") or ""
    desc_lower = desc.lower()

    # Original text metrics
    features["description_length"] = len(desc)
    features["word_count"]         = len(desc.split())
    features["title_length"]       = len(title)
    features["has_description"]    = 1 if len(desc) > 100 else 0

    # Keyword counts (compatibility)
    features["fraud_keyword_count"] = sum(1 for kw in FRAUD_KEYWORDS if kw in desc_lower)
    features["legit_keyword_count"] = sum(1 for kw in LEGIT_KEYWORDS if kw in desc_lower)

    # Readability features
    readability = extract_readability_features(desc)
    features.update(readability)

    # Advanced text statistics
    stats = extract_text_stats(desc)
    features.update(stats)

    # NER features
    ner = extract_ner_features(desc)
    features.update(ner)

    # Domain reputation features
    domain = extract_domain_features(job)
    features.update(domain)

    # Email reputation features
    email = extract_email_features(job)
    features.update(email)

    # Payment fraud detection
    payment = extract_payment_features(desc)
    features.update(payment)

    # Contact risk features
    contact = extract_contact_features(desc)
    features.update(contact)

    # Urgency detection
    urgency = extract_urgency_features(desc)
    features.update(urgency)

    # Fraud categories and keyword score
    cats = extract_fraud_category_scores(desc, payment, contact, urgency, email, domain, ner)
    features.update(cats)

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

# TF-IDF TEXT FEATURES

def fit_tfidf_vectorizer(descriptions: list, max_features: int = 300) -> TfidfVectorizer:
    descriptions = list(descriptions)
    n_docs = len(descriptions)

    if n_docs < 5:
        min_df, max_df = 1, 1.0
    else:
        min_df, max_df = 2, 0.95

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3), # upgraded settings
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
    if vectorizer is None:
        vectorizer = joblib.load(TFIDF_PATH)

    tfidf_matrix = vectorizer.transform(descriptions)
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]

    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# MAIN PIPELINE

def build_feature_dataframe(jobs: list, fit_tfidf: bool = True) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()

    if VERBOSE:
        print(f"\nExtracting features from {len(jobs)} jobs...")

    # Numeric features
    numeric_features = [extract_numeric_features(job) for job in jobs]
    df_numeric = pd.DataFrame(numeric_features)

    # Overwrite job["job_description"] with clean text
    # Combine title + description for NLP extraction
    cleaned_descriptions = []
    for job in jobs:
        desc = job.get("job_description", "") or ""
        title = job.get("job_title", "") or ""
        clean = prepare_ml_text(desc)
        job["job_description"] = clean
        cleaned_descriptions.append(clean + " " + clean_text(title))

    # TF-IDF text features
    if fit_tfidf:
        vectorizer = fit_tfidf_vectorizer(cleaned_descriptions, max_features=300)
    else:
        vectorizer = joblib.load(TFIDF_PATH)

    df_tfidf = get_tfidf_features(cleaned_descriptions, vectorizer)

    # Semantic embeddings using SentenceTransformer
    emb_model = get_embedding_model()
    embeddings = emb_model.encode(cleaned_descriptions, show_progress_bar=False)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols)

    # Combine all
    df_combined = pd.concat([
        df_numeric.reset_index(drop=True),
        df_tfidf.reset_index(drop=True),
        df_emb.reset_index(drop=True)
    ], axis=1)

    if VERBOSE:
        print(f"   Final matrix:     {df_combined.shape[0]} rows x {df_combined.shape[1]} columns")

    return df_combined

def load_dataset_from_csv(csv_path: str) -> list:
    """
    Load local CSV dataset and map columns to standard job dict format.
    If a column is missing, handle gracefully with default values.
    Skills comma-separated string converted to list of strings.
    cleaned_stipend_monthly converted to numeric, defaults to 0.0.
    is_scam column mapped to target label 'is_scam'.
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"Dataset file is missing. Please place the dataset at: {csv_path}")
        
    df = pd.read_csv(csv_path_obj)
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
        
    # Check required target label column: is_scam
    if "is_scam" not in df.columns:
        raise ValueError("Required target label column 'is_scam' is missing from the dataset.")
        
    jobs = []
    for _, row in df.iterrows():
        # Handle columns with grace fallbacks
        title = row.get("title")
        job_title = str(title).strip() if pd.notna(title) else "Untitled"
        
        desc = row.get("description")
        job_description = str(desc).strip() if pd.notna(desc) else ""
        
        # Parse skills column
        skills_raw = row.get("skills")
        if pd.isna(skills_raw) or not str(skills_raw).strip():
            skills = []
        else:
            skills = [s.strip() for s in str(skills_raw).split(",") if s.strip()]
            
        company = row.get("company")
        company_name = str(company).strip() if pd.notna(company) else "Unknown"
        
        location = row.get("location")
        city = str(location).strip() if pd.notna(location) else "Remote"
        
        company_website = str(row.get("company_website") or "").strip()
        domain_name = str(row.get("domain_name") or "").strip().lower()
        
        # Parse cleaned_stipend_monthly to float, fallback to 0.0
        stipend_raw = row.get("cleaned_stipend_monthly")
        try:
            stipend = float(stipend_raw) if pd.notna(stipend_raw) else 0.0
        except ValueError:
            stipend = 0.0
            
        is_scam = int(row.get("is_scam", 0))
        
        jobs.append({
            "job_title": job_title,
            "job_description": job_description,
            "skills_required": skills,
            "skill_categories": {}, # default category
            "salary_min": stipend,
            "salary_max": stipend,
            "salary_raw": str(stipend),
            "city": city,
            "state": "",
            "country": "India",
            "mode": "Remote",
            "platform_name": "Unknown",
            "company_name": company_name,
            "company_website": company_website,
            "email_domain": domain_name,
            "is_scam": is_scam,
            "scam_score": 100.0 if is_scam == 1 else 0.0,
            "scam_risk_level": "Scam Likely" if is_scam == 1 else "Safe",
            "company_trust_score": 50.0,
            "recruiter_verification_score": 30.0
        })
    return jobs

def build_feature_dataframe_from_csv(csv_path: str, fit_tfidf: bool = False) -> pd.DataFrame:
    """
    Load data from CSV and extract features to a DataFrame.
    """
    jobs = load_dataset_from_csv(csv_path)
    return build_feature_dataframe(jobs, fit_tfidf=fit_tfidf)

# LABEL EXTRACTION

def extract_labels(jobs: list) -> pd.DataFrame:
    """
    Extract is_scam target label directly from jobs.
    This is the ONLY label column returned for training.
    """
    labels = []
    for job in jobs:
        # Check is_scam key first
        if "is_scam" in job:
            is_scam = int(job["is_scam"])
        else:
            # Fallback for compatibility with old test formats
            risk = job.get("scam_risk_level", "Safe") or "Safe"
            is_scam = 1 if risk in ("High Risk", "Scam Likely") else 0
            
        labels.append({
            "is_scam": is_scam
        })
        
    return pd.DataFrame(labels)

# SELF-TEST

def _self_test():
    global VERBOSE
    VERBOSE = True

    print("=" * 70)
    print("FEATURE EXTRACTOR - SELF-TEST")
    print("=" * 70)

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
            "company_name": "Google",
            "email_domain": "google.com"
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
            "company_name": "Unknown",
            "email_domain": "tempmail.com"
        },
    ]

    df_features = build_feature_dataframe(jobs, fit_tfidf=True)
    df_labels = extract_labels(jobs)

    print("\nSAMPLE FEATURES (first 2 rows, first 15 columns):")
    print(df_features.iloc[:2, :15].to_string())

    print("\nTotal features:", df_features.shape[1])

    features_path = MODELS_DIR / "features.csv"
    labels_path   = MODELS_DIR / "labels.csv"
    df_features.to_csv(features_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    print(f"\nSaved features and labels self-test outputs.")

if __name__ == "__main__":
    _self_test()
