#recruiter_verifier.py


import re


# CONSTANTS

DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com",
    "throwaway.email", "yopmail.com", "sharklasers.com",
    "10minutemail.com", "trashmail.com", "fakeinbox.com",
    "maildrop.cc", "dispostable.com", "mintemail.com",
}

PERSONAL_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.in", "hotmail.com",
    "outlook.com", "outlook.in", "rediffmail.com", "rediff.com",
    "live.com", "icloud.com", "protonmail.com", "zoho.com",
    "yandex.com", "aol.com", "msn.com",
}

RECRUITER_TITLES = {
    "hr", "human resource", "human resources",
    "talent", "talent acquisition", "talent partner",
    "recruiter", "recruiting", "recruitment",
    "hiring", "hiring manager",
    "people", "people operations", "people partner",
    "ta partner", "ta manager",
    "head of hr", "vp people", "chief people",
}

GENERIC_NAMES = {
    "hr", "hr team", "hr department", "hr manager",
    "admin", "administrator",
    "hiring team", "hiring", "recruiting team",
    "talent team", "people team",
    "career", "careers",
    "not listed", "not specified", "n/a", "na",
    "unknown", "anonymous",
}


# MAIN VERIFICATION FUNCTION

def verify_recruiter(
    name: str = "",
    title: str = "",
    email_domain: str = "",
    linkedin_url: str = "",
    company_domain: str = "",
) -> tuple:
    """
    Compute recruiter verification score.

    Args:
        name:           Recruiter's full name
        title:          Job title (e.g., "HR Manager")
        email_domain:   Domain part of email (e.g., "google.com")
        linkedin_url:   Recruiter's LinkedIn profile URL
        company_domain: Company's domain (for cross-checking email)

    Returns:
        (score, flags)
            score: 0-100 (higher = more trusted)
            flags: List of detected signals

    Scoring logic (base = 30):
        +35  Corporate email matches company domain
        +25  Has LinkedIn profile
        +10  Recruiter title detected
        +10  Proper full name (2+ words, capitalized)
        +10  Non-personal email domain
        -30  Disposable email
        -20  Personal email (gmail/yahoo)
        -15  Generic name ("HR Team", "Admin")
        -10  No LinkedIn profile
    """
    score = 30.0
    flags = []

    if not name and not email_domain:
        return 5.0, ["no_recruiter_info"]

    # Check 1: Name quality
    if name:
        name_clean = name.strip().lower()

        if name_clean in GENERIC_NAMES:
            score -= 15
            flags.append("generic_name")
        else:
            words = name.strip().split()
            if len(words) >= 2:
                capitalized = sum(1 for w in words if w and w[0].isupper())
                if capitalized >= 2:
                    score += 10
                    flags.append("proper_full_name")
                else:
                    score += 3
                    flags.append("has_full_name")
            elif len(words) == 1 and len(words[0]) > 2:
                score += 5
                flags.append("single_name")
    else:
        flags.append("no_name")

    # Check 2: Title
    if title:
        title_lower = title.lower()
        if any(t in title_lower for t in RECRUITER_TITLES):
            score += 10
            flags.append("recruiter_title")
        else:
            flags.append("non_recruiter_title")

    # Check 3: Email domain (most important)
    if email_domain:
        email_domain_clean = email_domain.lower().strip()

        if email_domain_clean in DISPOSABLE_DOMAINS:
            score -= 30
            flags.append("disposable_email")

        elif email_domain_clean in PERSONAL_DOMAINS:
            score -= 20
            flags.append("personal_email")

        elif company_domain and email_domain_clean == company_domain.lower():
            score += 35
            flags.append("corporate_email_match")

        elif company_domain and _domain_matches_company(
            email_domain_clean, company_domain.lower()
        ):
            score += 25
            flags.append("corporate_email_similar")

        else:
            score += 10
            flags.append("non_personal_email")
    else:
        flags.append("no_email")

    # Check 4: LinkedIn presence
    if linkedin_url and "linkedin.com/in/" in linkedin_url.lower():
        score += 25
        flags.append("linkedin_present")
    else:
        score -= 10
        flags.append("no_linkedin")

    score = round(min(100, max(0, score)), 2)
    return score, flags


# HELPER FUNCTIONS

def _domain_matches_company(email_domain: str, company_domain: str) -> bool:
    """
    Check if email domain is a variant of the company domain.

    Examples:
        ("google.co.in",        "google.com") -> True (same base "google")
        ("careers.google.com",  "google.com") -> True (subdomain)
        ("acme.com",            "google.com") -> False
    """
    if not email_domain or not company_domain:
        return False

    def base(d: str) -> str:
        parts = d.split('.')
        if parts and parts[0] == 'www' and len(parts) > 1:
            parts = parts[1:]
        return parts[0] if parts else ""

    return base(email_domain) == base(company_domain)


# SELF-TEST

def _self_test():
    print("=" * 70)
    print("RECRUITER VERIFIER - SELF-TEST")
    print("=" * 70)

    test_cases = [
        {
            "name": "Real Google recruiter",
            "data": {
                "name":           "Rahul Sharma",
                "title":          "Senior Talent Acquisition Manager",
                "email_domain":   "google.com",
                "linkedin_url":   "https://linkedin.com/in/rahul-sharma-google",
                "company_domain": "google.com",
            },
        },
        {
            "name": "Suspicious 'HR Team'",
            "data": {
                "name":           "HR Team",
                "title":          "",
                "email_domain":   "gmail.com",
                "linkedin_url":   "",
                "company_domain": "",
            },
        },
        {
            "name": "Disposable email scam",
            "data": {
                "name":           "Admin",
                "title":          "Recruiter",
                "email_domain":   "mailinator.com",
                "linkedin_url":   "",
                "company_domain": "",
            },
        },
        {
            "name": "Real recruiter, gmail (medium)",
            "data": {
                "name":           "Priya Patel",
                "title":          "HR Manager",
                "email_domain":   "gmail.com",
                "linkedin_url":   "https://linkedin.com/in/priya-patel",
                "company_domain": "startup.in",
            },
        },
        {
            "name": "No info at all",
            "data": {},
        },
        {
            "name": "Corporate domain variant",
            "data": {
                "name":           "Amit Kumar",
                "title":          "Talent Partner",
                "email_domain":   "google.co.in",
                "company_domain": "google.com",
                "linkedin_url":   "https://linkedin.com/in/amit",
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{'-' * 70}")
        print(f"Test: {tc['name']}")

        score, flags = verify_recruiter(**tc["data"])

        if score >= 70:
            label = "[HIGH]   "
        elif score >= 40:
            label = "[MEDIUM] "
        elif score >= 20:
            label = "[LOW]    "
        else:
            label = "[NONE]   "

        print(f"  Verification Score: {label}{score}/100")
        print(f"  Flags: {', '.join(flags)}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
