"""
==========================================================
Fake Internship & Job Scam Detection System
NLP Engine - Part 1
Author: Graphura India Pvt Ltd
==========================================================
"""

# IMPORTS

import re
import nltk
from typing import List, Dict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# DOWNLOAD NLTK DATA

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")




# INITIALIZE NLP OBJECTS

stemmer = PorterStemmer()

STOP_WORDS = set(stopwords.words("english"))


# SCAM KEYWORD DATABASE
# Weight : 0 - 100

SCAM_KEYWORDS = {

    "registration fee": 95,

    "processing fee": 90,

    "joining fee": 92,

    "security deposit": 88,

    "pay before joining": 100,

    "instant joining": 82,

    "earn daily": 78,

    "earn money": 75,

    "quick earning": 80,

    "easy money": 85,

    "limited seats": 65,

    "urgent hiring": 55,

    "no interview": 78,

    "whatsapp hr": 92,

    "telegram": 70,

    "investment required": 95,

    "100 guaranteed": 85,

    "work from home without skills": 90,

    "dm now": 55,

    "salary guaranteed": 82,

    "pay and join": 98,

    # New additions
    "make money": 75,
    "no experience": 70,
    "no resume": 80,
    "no background check": 85,
    "start immediately": 80,
    "instant payout": 90,
    "transfer funds": 95,
    "personal account": 90,
    "comfort of bed": 80,
    "accept automatically": 85,
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
        # Expanded variations
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
        # Expanded variations
        "contact wa", "contact on wa", "contact via wa", "msg on wa", "ping on wa",
        "whatsapp group", "wa group"
    ],
    "instant joining": [
        "direct joining", "immediate joining", "join immediately",
        "join today", "start today", "joining today",
        "same day joining", "joining without interview",
        # New additions
        "start immediately", "urgent workers"
    ],
    "no interview": [
        "no selection process", "instant selection", "no interview needed",
        "no interview required", "direct selection", "without interview",
        "no screening", "no test required",
        # New additions
        "no resume needed", "no resume required", "no background check", 
        "everyone is accepted automatically", "no experience required", 
        "absolutely none required", "accepted automatically"
    ],
    "earn daily": [
        "daily earnings", "earn per day", "daily income",
        "paid daily", "daily payment", "earn every day",
        "get paid daily", "daily payout", "daily payouts", "daily payout instantly",
        # New additions
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
        # New additions
        "comfort of your bed", "comfort of bed"
    ],
    "pay to work": [
        "pay and work", "invest to earn", "deposit to join",
        "pay before joining", "fee before work", "charges to join"
    ],
    
    # New synonym rules
    "transfer funds": [
        "transfer processing funds", "transfer processing fund", "transfer fund",
        "transfer money", "transfer payments"
    ],
    "personal account": [
        "use personal account", "use your personal account", "using your personal account"
    ]

}


# HELPER FOR RAW TEXT CLEANING & SYNONYM REPLACEMENT (BEFORE STEMMING)

def clean_and_replace_synonyms(text: str) -> str:
    """
    Lowercase and clean text, then replace unstemmed synonyms using word boundaries
    to prevent partial substring replacements (e.g. replacing 'wa' inside 'wallet').
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
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # replace synonyms using word boundaries
    for canonical, synonyms in SCAM_SYNONYMS.items():
        canonical_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", canonical).lower()
        canonical_clean = re.sub(r"\s+", " ", canonical_clean).strip()
        for synonym in synonyms:
            syn_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", synonym).lower()
            syn_clean = re.sub(r"\s+", " ", syn_clean).strip()
            if syn_clean:
                text = re.sub(r"\b" + re.escape(syn_clean) + r"\b", canonical_clean, text)
                
    return text


# TEXT PREPROCESSING

def preprocess_text(text: str) -> str:
    """
    Tokenize, remove stopwords, and stem clean text.
    """
    if text is None:
        return ""

    text = str(text)

    # If it hasn't been cleaned yet (fallback safety), apply a light cleaning
    if any(c in text for c in ["<", ">", "@", "http", "www"]) or not text.islower():
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    # tokenize
    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if token not in STOP_WORDS:

            token = stemmer.stem(token)

            cleaned_tokens.append(token)

    cleaned_text = " ".join(cleaned_tokens)

    return cleaned_text


# REPLACE SYNONYMS (Backward-compatible fallback)

def replace_synonyms(text: str) -> str:
    """
    Backward-compatible method to replace synonyms.
    """
    return clean_and_replace_synonyms(text)


# GENERATE N-GRAMS

def generate_ngrams(tokens, n=2):
    """
    Generate n-grams.

    Example

    ["instant","joining","registration","fee"]

    bigram

    instant joining
    joining registration
    registration fee
    """
    return [
        " ".join(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]


# KEYWORD ANALYSIS

def keyword_analysis(text: str):
    """
    Detect fraud keywords and calculate keyword score using correctly-ordered
    synonym matching and dynamically preprocessed keyword stemming.
    """
    # Step 1: Clean and replace synonyms on raw unstemmed text first
    cleaned_synonym_text = clean_and_replace_synonyms(text)

    # Step 2: Stem and remove stopwords
    clean_text = preprocess_text(cleaned_synonym_text)

    # Step 3: Tokenize
    tokens = clean_text.split()

    # bigrams
    bigrams = generate_ngrams(tokens, 2)

    # trigrams
    trigrams = generate_ngrams(tokens, 3)

    # combine
    searchable_terms = (
        tokens +
        bigrams +
        trigrams
    )

    matched_keywords = []

    matched_scores = []

    # keyword matching with dynamically preprocessed and stemmed dictionary keys
    for keyword, score in SCAM_KEYWORDS.items():
        kw_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", keyword).lower()
        kw_clean = re.sub(r"\s+", " ", kw_clean).strip()
        kw_stemmed = preprocess_text(kw_clean)

        if kw_stemmed in searchable_terms:
            matched_keywords.append(keyword)
            matched_scores.append(score)

    # keyword score
    if len(matched_scores) == 0:
        keyword_score = 0
    else:
        # Average and cap at 100
        keyword_score = min(
            sum(matched_scores) / len(matched_scores),
            100
        )

    return {
        "clean_text": clean_text,
        "tokens": tokens,
        "matched_keywords": matched_keywords,
        "keyword_score": round(keyword_score, 2)
    }


# SIMPLE HELPER FUNCTIONS

def get_keyword_score(text: str):
    """
    Returns only keyword score.
    """
    result = keyword_analysis(text)
    return result["keyword_score"]


def get_matched_keywords(text: str):
    """
    Returns matched keywords only.
    """
    result = keyword_analysis(text)
    return result["matched_keywords"]


def get_clean_text(text: str):
    """
    Returns cleaned text.
    """
    result = keyword_analysis(text)
    return result["clean_text"]


# NEW INTEGRATED HELPERS

def prepare_ml_text(text: str) -> str:
    """
    Preprocess and replace synonyms in correct sequence for ML pipeline.
    """
    clean = clean_and_replace_synonyms(text)
    clean = preprocess_text(clean)
    return clean


def process_job_description(text: str) -> dict:
    """
    Exposes keyword analysis to scoring orchestrator.
    """
    return keyword_analysis(text)


def calculate_final_score(keyword_score: float, ml_score: float) -> dict:
    final_score = round((keyword_score * 0.6) + (ml_score * 0.4), 2)
    return {"final_score": final_score}
