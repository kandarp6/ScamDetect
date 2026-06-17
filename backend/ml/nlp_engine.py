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
from typing import List

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

    "pay and join": 98
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
        "same day joining", "joining without interview"
    ],
    "no interview": [
        "no selection process", "instant selection", "no interview needed",
        "no interview required", "direct selection", "without interview",
        "no screening", "no test required"
    ],
    "earn daily": [
        "daily earnings", "earn per day", "daily income",
        "paid daily", "daily payment", "earn every day",
        "get paid daily", "daily payout"
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
        "remote earning", "online income"
    ],
    "pay to work": [
        "pay and work", "invest to earn", "deposit to join",
        "pay before joining", "fee before work", "charges to join"
    ]

}

# TEXT PREPROCESSING

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove Emails
    4. Remove HTML
    5. Remove punctuation
    6. Remove extra spaces
    7. Tokenize
    8. Remove stopwords
    9. Lemmatize

    Returns cleaned string.
    """

    if text is None:
        return ""

    text = str(text)

    # lowercase
    text = text.lower()

    # remove html
    text = re.sub(r"<.*?>", " ", text)

    # remove urls
    text = re.sub(r"http\\S+|www\\S+", " ", text)

    # remove emails
    text = re.sub(r"\\S+@\\S+", " ", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\\s+", " ", text).strip()

    # tokenize
    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if token not in STOP_WORDS:

            token = stemmer.stem(token)

            cleaned_tokens.append(token)

    cleaned_text = " ".join(cleaned_tokens)

    return cleaned_text
# REPLACE SYNONYMS

def replace_synonyms(text: str) -> str:
    """
    Replace scam synonyms with standard keywords.

    Example:
    registration charges -> registration fee
    processing charges -> processing fee
    immediate joining -> instant joining
    """

    if not text:
        return ""

    updated_text = text

    for canonical, synonyms in SCAM_SYNONYMS.items():
        for synonym in synonyms:
            updated_text = updated_text.replace(synonym, canonical)

    return updated_text


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
    Detect fraud keywords and calculate keyword score.
    """

    # preprocess

    clean_text = preprocess_text(text)

    # synonym replacement

    clean_text = replace_synonyms(clean_text)

    # tokenize

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

    # keyword matching

    for keyword, score in SCAM_KEYWORDS.items():

        if keyword in searchable_terms:

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
    Preprocess and replace synonyms.
    """
    clean = preprocess_text(text)
    clean = replace_synonyms(clean)
    return clean


def process_job_description(text: str) -> dict:
    """
    Exposes keyword analysis to scoring orchestrator.
    """
    return keyword_analysis(text)


def calculate_final_score(keyword_score: float, ml_score: float) -> dict:
    final_score = round((keyword_score * 0.6) + (ml_score * 0.4), 2)
    return {"final_score": final_score}
