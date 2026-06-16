"""
scoring_engine.py
Core fraud detection engine. Combines multiple signals to compute a 0-100
fraud score with human-readable explanations.

Categories and weights:
    Company trust       30%   - Is the company real?
    Recruiter verify    25%   - Is the recruiter real?
    Social media risk   20%   - WhatsApp/Telegram contacts?
    Keyword risk        15%   - "Registration fee", "earn 50k daily"
    Salary realism      10%   - Unrealistic earnings claims?

Bonuses / penalties:
    Government bonus           -20  (NCS / verified gov portals)
    Skill-title mismatch       +15
    Platform trust bonus       Per-platform table (see PLATFORM_TRUST_BONUS)

Risk levels:
    0-20    Safe
    21-40   Low Risk
    41-60   Medium Risk
    61-80   High Risk
    81-100  Scam Likely
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

WEIGHTS = {
    "company":   0.30,
    "recruiter": 0.25,
    "social":    0.20,
    "keyword":   0.15,
    "salary":    0.10,
}

GOVERNMENT_BONUS       = -20.0
SKILL_MISMATCH_PENALTY = 15.0

PLATFORM_TRUST_BONUS = {
    "Internshala":          -15.0,
    "LinkedIn":             -10.0,
    "Naukri":               -10.0,
    "Foundit":              -10.0,
    "Shine":                 -5.0,
    "NCS (Govt. of India)": -25.0,
}


# ============================================================================
# KNOWLEDGE BASES
# ============================================================================

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.in", "hotmail.com",
    "outlook.com", "rediffmail.com", "rediff.com",
    "live.com", "icloud.com", "protonmail.com", "zoho.com",
    "yandex.com", "aol.com",
}

DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com",
    "throwaway.email", "yopmail.com", "sharklasers.com",
    "10minutemail.com", "trashmail.com", "fakeinbox.com",
    "maildrop.cc", "dispostable.com", "mintemail.com",
    "tempinbox.com", "spam4.me",
}

KEYWORD_SIGNALS = {
    # Payment scams
    "registration fee":         85,
    "registration charges":     85,
    "pay to apply":             85,
    "application fee":          80,
    "processing fee":           75,
    "refundable deposit":       75,
    "security deposit":         75,
    "training fee":             70,
    "training charges":         70,
    "course fee":               65,
    "joining fee":              80,

    # Unrealistic earnings
    "earn daily":               60,
    "daily income":             55,
    "earn weekly":              55,
    "weekly payout":            50,
    "100% job guarantee":       60,
    "guaranteed placement":     55,
    "guaranteed salary":        50,
    "work from home earn":      45,
    "unlimited income":         60,
    "no investment":            40,

    # Urgent / pressure tactics
    "urgent hiring":            30,
    "immediate joining":        25,
    "limited seats":            30,
    "limited slots":            30,
    "hurry up":                 25,
    "apply now or never":       40,
    "last chance":              35,

    # Vague / suspicious
    "no experience needed":     20,
    "any qualification":        20,
    "anyone can apply":         15,
    "10th 12th pass":           15,
    "no degree required":       15,

    # Communication red flags
    "apply on whatsapp":        75,
    "whatsapp resume":          70,
    "contact hr on whatsapp":   80,
    "apply on telegram":        80,
    "telegram channel":         70,
    "join our telegram":        65,
    "dm us on instagram":       60,
    "message us on facebook":   50,

    # Crypto / MLM
    "bitcoin trading":          70,
    "crypto trading":           65,
    "forex trading":            60,
    "binary options":           75,
    "network marketing":        55,
    "mlm":                      60,
    "multi-level marketing":    65,
    "direct selling":           50,
    "pyramid":                  85,
}

POSITIVE_KEYWORDS = [
    "competitive salary",
    "health insurance",
    "provident fund",
    "annual bonus",
    "equity",
    "esop",
    "professional development",
    "career growth",
    "mentorship",
    "structured training",
]


# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class ScoreBreakdown:
    """Detailed breakdown for explainable AI."""
    company_risk_raw:   float = 0.0
    recruiter_risk_raw: float = 0.0
    social_risk_raw:    float = 0.0
    keyword_risk_raw:   float = 0.0
    salary_risk_raw:    float = 0.0

    company_risk_weighted:   float = 0.0
    recruiter_risk_weighted: float = 0.0
    social_risk_weighted:    float = 0.0
    keyword_risk_weighted:   float = 0.0
    salary_risk_weighted:    float = 0.0

    total_score: float = 0.0
    risk_level:  str   = "Safe"

    risk_factors:     list = field(default_factory=list)
    positive_factors: list = field(default_factory=list)
    matched_keywords: list = field(default_factory=list)

    government_bonus_applied: bool = False
    mismatch_penalty_applied: bool = False
    platform_bonus_applied:   bool = False


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def compute_fraud_score(
    job: dict,
    company_trust: float = 50.0,
    recruiter_verif: float = 30.0,
    is_government: bool = False,
    skill_mismatch: bool = False,
    platform_name: str = "",
) -> ScoreBreakdown:
    """
    Compute the fraud score for a job posting.

    Args:
        job: Dict with job_description, salary_raw, email_domain, etc.
        company_trust: 0-100 from company_trust.py
        recruiter_verif: 0-100 from recruiter_verifier.py
        is_government: True if from NCS or verified gov portal
        skill_mismatch: True if title mentions skill not in skills list
        platform_name: For platform-specific bonus
    """
    b = ScoreBreakdown()

    desc                  = (job.get("job_description") or "").lower()
    salary_raw            = (job.get("salary_raw") or "").lower()
    email_domain          = (job.get("email_domain") or "").lower()
    is_suspicious_salary  = job.get("is_suspicious_salary", False)

    # Category 1: Company risk
    b.company_risk_raw      = max(0, 100 - company_trust)
    b.company_risk_weighted = b.company_risk_raw * WEIGHTS["company"]

    if company_trust < 10:
        b.risk_factors.append("Company domain does not exist")
    elif company_trust < 30:
        b.risk_factors.append("Company has no verifiable presence online")
    elif company_trust < 50:
        b.risk_factors.append("Limited company verification data")
    elif company_trust >= 80:
        b.positive_factors.append("Highly verified company")

    # Category 2: Recruiter risk
    b.recruiter_risk_raw = max(0, 100 - recruiter_verif)

    if email_domain in DISPOSABLE_EMAIL_DOMAINS:
        b.recruiter_risk_raw = min(100, b.recruiter_risk_raw + 30)
        b.risk_factors.append(f"Disposable email address ({email_domain})")
    elif email_domain in PERSONAL_EMAIL_DOMAINS:
        b.recruiter_risk_raw = min(100, b.recruiter_risk_raw + 15)
        b.risk_factors.append(f"Personal email contact ({email_domain})")
    elif email_domain and "." in email_domain:
        b.positive_factors.append(f"Corporate email used ({email_domain})")

    b.recruiter_risk_weighted = b.recruiter_risk_raw * WEIGHTS["recruiter"]

    if recruiter_verif < 20:
        b.risk_factors.append("Recruiter identity cannot be verified")
    elif recruiter_verif >= 80:
        b.positive_factors.append("Recruiter verified (corporate email + LinkedIn)")

    # Category 3: Social media risk
    social_signals = []

    if re.search(r't\.me/|telegram\.me/|join.*telegram', desc):
        b.social_risk_raw = min(100, b.social_risk_raw + 80)
        social_signals.append("Telegram contact in job description")

    if re.search(r'wa\.me/|whatsapp|whats\s*app', desc):
        b.social_risk_raw = min(100, b.social_risk_raw + 75)
        social_signals.append("WhatsApp contact in job description")

    if re.search(r'instagram\.com/|dm.*insta|insta.*dm', desc):
        b.social_risk_raw = min(100, b.social_risk_raw + 50)
        social_signals.append("Instagram DM as application channel")

    if re.search(r'facebook\.com/|message.*facebook|fb\.com', desc):
        b.social_risk_raw = min(100, b.social_risk_raw + 40)
        social_signals.append("Facebook message as application channel")

    b.social_risk_weighted = b.social_risk_raw * WEIGHTS["social"]
    b.risk_factors.extend(social_signals)

    # Category 4: Keyword risk
    matched_keywords_with_score = []
    for phrase, severity in KEYWORD_SIGNALS.items():
        if phrase in desc:
            matched_keywords_with_score.append((phrase, severity))
            b.matched_keywords.append(phrase)

    if matched_keywords_with_score:
        max_severity = max(s for _, s in matched_keywords_with_score)
        bonus = min(20, len(matched_keywords_with_score) * 5)
        b.keyword_risk_raw = min(100, max_severity + bonus)

        top_3 = sorted(matched_keywords_with_score, key=lambda x: -x[1])[:3]
        for phrase, _ in top_3:
            b.risk_factors.append(f'Suspicious phrase: "{phrase}"')

    positive_matches = [kw for kw in POSITIVE_KEYWORDS if kw in desc]
    if positive_matches:
        b.keyword_risk_raw = max(0, b.keyword_risk_raw - len(positive_matches) * 5)
        b.positive_factors.append(
            f"{len(positive_matches)} professional benefits mentioned"
        )

    b.keyword_risk_weighted = b.keyword_risk_raw * WEIGHTS["keyword"]

    # Category 5: Salary risk
    if is_suspicious_salary:
        b.salary_risk_raw = 80
        b.risk_factors.append("Unrealistic salary pattern detected")

    if re.search(r'earn\s+[\d,]+\s*(daily|per\s*day)', desc):
        b.salary_risk_raw = max(b.salary_risk_raw, 70)
        if "Unrealistic earnings claim" not in str(b.risk_factors):
            b.risk_factors.append("Unrealistic earnings claim in description")

    if "unlimited" in salary_raw or "uncapped" in salary_raw:
        b.salary_risk_raw = max(b.salary_risk_raw, 50)

    b.salary_risk_weighted = b.salary_risk_raw * WEIGHTS["salary"]

    # Total
    total = (
        b.company_risk_weighted   +
        b.recruiter_risk_weighted +
        b.social_risk_weighted    +
        b.keyword_risk_weighted   +
        b.salary_risk_weighted
    )

    # Bonuses and penalties
    if is_government:
        total = max(0, total + GOVERNMENT_BONUS)
        b.government_bonus_applied = True
        b.positive_factors.append("Verified government portal (NCS)")

    if skill_mismatch:
        total = min(100, total + SKILL_MISMATCH_PENALTY)
        b.mismatch_penalty_applied = True
        b.risk_factors.append("Job title mentions skills not in description")

    if platform_name in PLATFORM_TRUST_BONUS:
        platform_bonus = PLATFORM_TRUST_BONUS[platform_name]
        total = max(0, total + platform_bonus)
        b.platform_bonus_applied = True
        if platform_bonus <= -10:
            b.positive_factors.append(f"Verified platform: {platform_name}")

    # Finalize
    b.total_score = round(min(100, max(0, total)), 2)
    b.risk_level  = compute_risk_level(b.total_score)

    # Deduplicate factors
    b.risk_factors     = list(dict.fromkeys(b.risk_factors))
    b.positive_factors = list(dict.fromkeys(b.positive_factors))

    return b


# ============================================================================
# RISK LEVEL CLASSIFIER
# ============================================================================

def compute_risk_level(score: float) -> str:
    """Convert numeric score (0-100) to a categorical risk level."""
    if score <= 20:
        return "Safe"
    elif score <= 40:
        return "Low Risk"
    elif score <= 60:
        return "Medium Risk"
    elif score <= 80:
        return "High Risk"
    else:
        return "Scam Likely"


# ============================================================================
# HUMAN-READABLE REPORT
# ============================================================================

def format_score_report(breakdown: ScoreBreakdown) -> str:
    """Format ScoreBreakdown as a readable text report."""
    risk_tag = {
        "Safe":        "[SAFE]  ",
        "Low Risk":    "[LOW]   ",
        "Medium Risk": "[MED]   ",
        "High Risk":   "[HIGH]  ",
        "Scam Likely": "[SCAM]  ",
    }.get(breakdown.risk_level, "[?]     ")

    report = []
    report.append(f"\n{risk_tag}FRAUD SCORE: {breakdown.total_score}/100 ({breakdown.risk_level})")
    report.append("=" * 60)

    report.append("\nCATEGORY BREAKDOWN:")
    report.append(f"  Company Risk    : {breakdown.company_risk_raw:5.1f} x 30% = {breakdown.company_risk_weighted:5.2f}")
    report.append(f"  Recruiter Risk  : {breakdown.recruiter_risk_raw:5.1f} x 25% = {breakdown.recruiter_risk_weighted:5.2f}")
    report.append(f"  Social Risk     : {breakdown.social_risk_raw:5.1f} x 20% = {breakdown.social_risk_weighted:5.2f}")
    report.append(f"  Keyword Risk    : {breakdown.keyword_risk_raw:5.1f} x 15% = {breakdown.keyword_risk_weighted:5.2f}")
    report.append(f"  Salary Risk     : {breakdown.salary_risk_raw:5.1f} x 10% = {breakdown.salary_risk_weighted:5.2f}")

    if breakdown.risk_factors:
        report.append("\nRISK FACTORS:")
        for factor in breakdown.risk_factors:
            report.append(f"  - {factor}")

    if breakdown.positive_factors:
        report.append("\nPOSITIVE FACTORS:")
        for factor in breakdown.positive_factors:
            report.append(f"  - {factor}")

    if breakdown.matched_keywords:
        report.append(f"\nMATCHED KEYWORDS: {', '.join(breakdown.matched_keywords[:5])}")

    if breakdown.government_bonus_applied:
        report.append(f"\nGovernment bonus applied: {GOVERNMENT_BONUS}")

    if breakdown.mismatch_penalty_applied:
        report.append(f"\nSkill mismatch penalty: +{SKILL_MISMATCH_PENALTY}")

    if breakdown.platform_bonus_applied:
        report.append("\nPlatform trust bonus applied")

    return "\n".join(report)


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test():
    print("=" * 70)
    print("SCORING ENGINE - SELF-TEST")
    print("=" * 70)

    test_cases = [
        {
            "name": "Real Google internship (no platform bonus)",
            "job": {
                "job_description": "Software Engineering Intern at Google. Competitive salary, health insurance, ESOP available.",
                "salary_raw":      "Rs 50,000 - Rs 80,000 per month",
                "email_domain":    "google.com",
            },
            "company_trust":   95,
            "recruiter_verif": 90,
        },
        {
            "name": "Internshala internship (with platform bonus)",
            "job": {
                "job_description": "Python developer intern needed. Work from home. Stipend Rs 15,000/month.",
                "salary_raw":      "Rs 15,000/month",
                "email_domain":    "",
            },
            "company_trust":   50,
            "recruiter_verif": 30,
            "platform_name":   "Internshala",
        },
        {
            "name": "Classic scam (no platform bonus)",
            "job": {
                "job_description":      "Earn Rs 50,000 daily! Pay Rs 500 registration fee. WhatsApp us!",
                "salary_raw":           "Unlimited earnings",
                "email_domain":         "gmail.com",
                "is_suspicious_salary": True,
            },
            "company_trust":   5,
            "recruiter_verif": 10,
        },
        {
            "name": "Government job (NCS)",
            "job": {
                "job_description": "Junior Engineer at Ministry of Railways.",
                "salary_raw":      "Rs 56,100 per month",
                "email_domain":    "gov.in",
            },
            "company_trust":   100,
            "recruiter_verif": 50,
            "is_government":   True,
            "platform_name":   "NCS (Govt. of India)",
        },
    ]

    for tc in test_cases:
        print(f"\n{'=' * 70}")
        print(f"TEST: {tc['name']}")
        print(f"{'-' * 70}")

        breakdown = compute_fraud_score(
            job=tc["job"],
            company_trust=tc.get("company_trust", 50),
            recruiter_verif=tc.get("recruiter_verif", 30),
            is_government=tc.get("is_government", False),
            platform_name=tc.get("platform_name", ""),
        )

        print(format_score_report(breakdown))

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()