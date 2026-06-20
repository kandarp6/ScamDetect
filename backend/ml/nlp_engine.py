"""
Fake Internship & Job Scam Detection System
NLP Engine - Upgraded with spaCy en_core_web_sm Lemmatization and Normalization
"""

import re
import spacy
from typing import List, Dict

# INITIALIZE SPACY MODEL WITH AUTO-DOWNLOAD FALLBACK
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# SCAM KEYWORD DATABASE
# Weight : 0 - 100
# Weights are set up for cumulative scoring (summed and capped at 100)
SCAM_KEYWORDS = {
    # High Severity (Direct indicators of financial or identity scams)
    "registration fee": 90,
    "processing fee": 90,
    "joining fee": 90,
    "security deposit": 90,
    "pay before joining": 95,
    "pay and join": 95,
    "investment required": 95,
    "transfer funds": 95,
    "personal account": 90,
    "instant payout": 80,
    
    # Medium Severity (Highly suspicious terms correlating with fraud)
    "easy money": 70,
    "earn daily": 70,
    "earn money": 50,
    "quick earning": 60,
    "make money": 50,
    "comfort of bed": 60,
    "whatsapp hr": 65,
    "telegram": 45,
    "100 guaranteed": 55,
    "salary guaranteed": 55,
    "work from home without skills": 65,
    "accept automatically": 60,

    # Low Severity (Often found in legitimate entry-level/intern postings)
    "limited seats": 20,
    "urgent hiring": 15,
    "no interview": 25,
    "no experience": 15,
    "no resume": 20,
    "no background check": 20,
    "start immediately": 15,
    "dm now": 15,
}

# SYNONYM DATABASE
# Used before keyword matching
SCAM_SYNONYMS = {
    "registration fee": [
        "joining fee", "processing fee", "training fee",
        "security deposit", "refundable deposit", "enrolment fee",
        "reg fee", "registration charge", "onboarding fee",
        "zero cost onboarding", "complimentary training",
        "admission fee", "application fee",
        "joining charges", "joining charge", "joining fees",
        "registration charges", "registration charge", "registration fees",
        "processing charges", "processing charge", "processing fees",
        "training charges", "training charge", "training fees",
        "onboarding charges", "onboarding charge", "onboarding fees",
        "joining cost", "registration cost", "onboarding cost"
    ],
    "whatsapp hr": [
        "contact on whatsapp", "whatsapp only", "ping on whatsapp",
        "message on wa", "whatsapp number", "wa number",
        "contact via whatsapp", "whatsapp me", "msg on whatsapp",
        "contact wa", "contact on wa", "contact via wa", "msg on wa", "ping on wa",
        "whatsapp group", "wa group"
    ],
    "instant joining": [
        "direct joining", "immediate joining", "join immediately",
        "join today", "start today", "joining today",
        "same day joining", "joining without interview",
        "urgent workers"
    ],
    "no interview": [
        "no selection process", "instant selection", "no interview needed",
        "no interview required", "direct selection", "without interview",
        "no screening", "no test required"
    ],
    "no experience": [
        "no experience required", "absolutely none required", "no experience needed",
        "experience not required", "freshers can apply", "no experience necessary",
        "fresher welcome"
    ],
    "no resume": [
        "no resume needed", "no resume required", "apply without resume"
    ],
    "no background check": [
        "no background check required", "no background verification", "everyone is accepted automatically"
    ],
    "accept automatically": [
        "accepted automatically"
    ],
    "earn daily": [
        "daily earnings", "earn per day", "daily income",
        "paid daily", "daily payment", "earn every day",
        "get paid daily", "daily payout", "daily payouts", "daily payout instantly",
        "daily payouts instantly", "daily instant payouts", "instant payouts"
    ],
    "limited seats": [
        "limited slots", "hurry up", "last few seats",
        "only few left", "seats filling fast", "apply now limited",
        "urgent requirement", "limited vacancy"
    ],
    "guaranteed job": [
        "100 percent placement", "job guaranteed", "assured placement",
        "placement guarantee", "guaranteed placement", "sure shot job",
        "confirmed placement", "100% job"
    ],
    "work from home earn": [
        "earn from home", "work at home", "home based job",
        "online earning", "earn sitting at home", "home job",
        "remote earning", "online income",
        "comfort of your bed", "comfort of bed"
    ],
    "pay to work": [
        "pay and work", "invest to earn", "deposit to join",
        "pay before joining", "fee before work", "charges to join"
    ],
    "transfer funds": [
        "transfer processing funds", "transfer processing fund", "transfer fund",
        "transfer money", "transfer payments"
    ],
    "personal account": [
        "use personal account", "use your personal account", "using your personal account"
    ]
}

# REUSABLE PIPELINE FUNCTIONS

def clean_text(text: str) -> str:
    """
    1. Lowercase text
    2. Remove HTML
    3. Remove URLs
    4. Remove emails
    5. Remove special characters (keep alphanumeric and spaces)
    """
    if text is None:
        return ""
    text = str(text).lower()
    # remove html
    text = re.sub(r"<.*?>", " ", text)
    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)
    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def replace_synonyms(text: str) -> str:
    """
    6. Synonym replacement using word boundaries.
    """
    if not text:
        return ""
    for canonical, synonyms in SCAM_SYNONYMS.items():
        canonical_clean = clean_text(canonical)
        for synonym in synonyms:
            syn_clean = clean_text(synonym)
            if syn_clean:
                # Use word boundaries to prevent partial substring matches
                text = re.sub(r"\b" + re.escape(syn_clean) + r"\b", canonical_clean, text)
    return text

def lemmatize_text(text: str) -> List[str]:
    """
    7. Tokenization
    8. Stopword removal
    9. Lemmatization
    """
    if not text:
        return []
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Stopword and punctuation/whitespace removal
        if not token.is_stop and not token.is_punct and not token.is_space:
            lemma = token.lemma_.strip().lower()
            if lemma == "-pron-":
                lemma = token.text.strip().lower()
            if lemma:
                tokens.append(lemma)
    return tokens

def generate_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    10. N-Gram generation helper.
    """
    return [
        " ".join(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

def prepare_ml_text(text: str) -> str:
    """
    Combines the pipeline steps to return space-separated lemmatized text.
    """
    cleaned = clean_text(text)
    replaced = replace_synonyms(cleaned)
    tokens = lemmatize_text(replaced)
    return " ".join(tokens)

# BACKWARD COMPATIBLE FALLBACKS

def clean_and_replace_synonyms(text: str) -> str:
    cleaned = clean_text(text)
    return replace_synonyms(cleaned)

def preprocess_text(text: str) -> str:
    return prepare_ml_text(text)

def keyword_analysis(text: str) -> dict:
    """
    Detect fraud keywords and calculate keyword score using the updated spaCy pipeline.
    """
    # Preprocess text through updated pipeline
    clean_str = prepare_ml_text(text)
    tokens = clean_str.split()

    # Generate n-grams up to 3
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)

    searchable_terms = tokens + bigrams + trigrams
    matched_keywords = []
    matched_scores = []

    # keyword matching with lemmatized dictionary keys
    for keyword, score in SCAM_KEYWORDS.items():
        kw_clean = clean_text(keyword)
        kw_lemmatized = prepare_ml_text(kw_clean)

        if kw_lemmatized in searchable_terms or any(term == kw_lemmatized for term in searchable_terms):
            matched_keywords.append(keyword)
            matched_scores.append(score)

    keyword_score = min(sum(matched_scores), 100) if matched_scores else 0.0

    return {
        "clean_text": clean_str,
        "tokens": tokens,
        "matched_keywords": matched_keywords,
        "keyword_score": round(keyword_score, 2)
    }

def get_keyword_score(text: str) -> float:
    return keyword_analysis(text)["keyword_score"]

def get_matched_keywords(text: str) -> List[str]:
    return keyword_analysis(text)["matched_keywords"]

def get_clean_text(text: str) -> str:
    return keyword_analysis(text)["clean_text"]

def process_job_description(text: str) -> dict:
    return keyword_analysis(text)

def calculate_final_score(keyword_score: float, ml_score: float) -> dict:
    final_score = round((keyword_score * 0.6) + (ml_score * 0.4), 2)
    return {"final_score": final_score}
