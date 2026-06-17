#company_trust.py


import re
import socket
from dataclasses import dataclass, field


# DATA STRUCTURE

@dataclass
class CompanyIntelligence:
    """Holds computed trust data for a company."""
    name:           str   = ""
    domain:         str   = ""
    website_active: bool  = False
    trust_score:    float = 50.0
    trust_factors:  dict  = field(default_factory=lambda: {
        "positive": [],
        "negative": [],
    })


# KNOWN INDIAN AND GLOBAL COMPANIES (Auto-trust)

KNOWN_DOMAINS = {
    # IT Giants
    "google":          ("google.com",          98),
    "microsoft":       ("microsoft.com",       98),
    "amazon":          ("amazon.in",           97),
    "apple":           ("apple.com",           98),
    "meta":            ("meta.com",            95),
    "facebook":        ("facebook.com",        95),
    "netflix":         ("netflix.com",         95),
    "uber":            ("uber.com",            93),
    "salesforce":      ("salesforce.com",      95),
    "adobe":           ("adobe.com",           95),
    "ibm":             ("ibm.com",             95),
    "oracle":          ("oracle.com",          93),
    "cisco":           ("cisco.com",           93),
    "intel":           ("intel.com",           93),
    "nvidia":          ("nvidia.com",          95),

    # Indian IT Majors
    "tcs":              ("tcs.com",            95),
    "tata consultancy": ("tcs.com",            95),
    "infosys":          ("infosys.com",        95),
    "wipro":            ("wipro.com",          95),
    "hcl":              ("hcl.com",            93),
    "hcltech":          ("hcltech.com",        93),
    "tech mahindra":    ("techmahindra.com",   93),
    "techmahindra":     ("techmahindra.com",   93),
    "cognizant":        ("cognizant.com",      93),
    "accenture":        ("accenture.com",      95),
    "capgemini":        ("capgemini.com",      93),
    "deloitte":         ("deloitte.com",       93),
    "kpmg":             ("kpmg.com",           93),
    "pwc":              ("pwc.com",            93),
    "ey":               ("ey.com",             93),
    "mindtree":         ("mindtree.com",       90),
    "ltimindtree":      ("ltimindtree.com",    90),
    "persistent":       ("persistent.com",     90),
    "mphasis":          ("mphasis.com",        88),
    "zensar":           ("zensar.com",         85),

    # Indian Unicorns / Startups
    "flipkart":      ("flipkart.com",     93),
    "zomato":        ("zomato.com",       92),
    "swiggy":        ("swiggy.in",        92),
    "ola":           ("olacabs.com",      88),
    "byju":          ("byjus.com",        80),
    "byjus":         ("byjus.com",        80),
    "unacademy":     ("unacademy.com",    88),
    "razorpay":      ("razorpay.com",     92),
    "paytm":         ("paytm.com",        88),
    "phonepe":       ("phonepe.com",      92),
    "cred":          ("cred.club",        90),
    "zerodha":       ("zerodha.com",      93),
    "groww":         ("groww.in",         90),
    "upstox":        ("upstox.com",       88),
    "meesho":        ("meesho.com",       88),
    "nykaa":         ("nykaa.com",        90),
    "myntra":        ("myntra.com",       90),
    "bigbasket":     ("bigbasket.com",    88),
    "policybazaar":  ("policybazaar.com", 88),
    "makemytrip":    ("makemytrip.com",   90),
    "ixigo":         ("ixigo.com",        85),
    "freshworks":    ("freshworks.com",   92),
    "zoho":          ("zoho.com",         93),
    "postman":       ("postman.com",      90),
    "browserstack":  ("browserstack.com", 90),

    # Indian Conglomerates
    "reliance":   ("ril.com",          95),
    "tata":       ("tata.com",         95),
    "adani":      ("adani.com",        93),
    "mahindra":   ("mahindra.com",     93),
    "birla":      ("adityabirla.com",  93),
    "godrej":     ("godrej.com",       93),

    # Indian Banks
    "hdfc":     ("hdfcbank.com",  95),
    "icici":    ("icicibank.com", 95),
    "sbi":      ("sbi.co.in",     95),
    "axis":     ("axisbank.com",  93),
    "kotak":    ("kotak.com",     93),
    "yes bank": ("yesbank.in",    88),

    # Education
    "internshala":    ("internshala.com",  90),
    "scaler":         ("scaler.com",       85),
    "coding ninjas":  ("codingninjas.com", 85),
    "newton school":  ("newtonschool.co",  80),

    # Government
    "ncs":        ("ncs.gov.in", 100),
    "ministry":   ("",           100),
    "government": ("",           100),
}


# GENERIC NAME PATTERNS (red flag - anyone can use these)

GENERIC_NAME_PATTERNS = [
    r'\bit\s+solutions?',
    r'\bsoftware\s+solutions?',
    r'\btechnologies?\s+solutions?',
    r'\bconsultanc(?:y|ies)',
    r'\bservices?\s+(?:pvt|ltd|private|limited)',
    r'\bgroup\s+(?:of\s+)?compan(?:y|ies)',
    r'\benterprises?\s+(?:pvt|ltd)',
    r'\bventures?\s+(?:pvt|ltd)',
    r'\bsolutions?\s+(?:pvt|ltd|private|limited)',
]

LEGAL_SUFFIXES = [
    'pvt ltd', 'pvtltd', 'private limited', 'privatelimited',
    'ltd', 'limited', 'inc', 'incorporated', 'corporation',
    'llp', 'llc', 'gmbh', 'co', 'company',
]


# MAIN COMPUTE FUNCTION

def compute_company_trust(
    company_name: str,
    employee_count: int = 0,
    has_linkedin: bool = False,
) -> CompanyIntelligence:
    """
    Compute trust score (0-100) for a company.

    Args:
        company_name:   Company name from scraper
        employee_count: Number of employees if known
        has_linkedin:   True if a LinkedIn page was found
    """
    intel = CompanyIntelligence(name=company_name or "")

    # Handle empty/unknown
    if not company_name or company_name.strip().lower() in ("unknown", "n/a", ""):
        intel.trust_score = 5.0
        intel.trust_factors["negative"].append("No company name provided")
        return intel

    name = company_name.strip()
    name_lower = name.lower()
    score = 50.0  # Neutral starting point

    # Step 1: Check known companies (fast path)
    for key, (domain, known_score) in KNOWN_DOMAINS.items():
        if key in name_lower:
            intel.domain = domain
            intel.trust_score = float(known_score)
            intel.website_active = True
            intel.trust_factors["positive"].append(
                f"Recognized company ({key.title()})"
            )
            return intel

    # Step 2: Derive domain from name
    intel.domain = _derive_domain(name_lower)

    # Step 3: Check if domain exists via DNS
    if intel.domain:
        intel.website_active = _check_domain_exists(intel.domain)
        if intel.website_active:
            score += 20
            intel.trust_factors["positive"].append(
                f"Domain reachable: {intel.domain}"
            )
        else:
            score -= 25
            intel.trust_factors["negative"].append(
                f"Could not verify domain ({intel.domain})"
            )
    else:
        score -= 20
        intel.trust_factors["negative"].append("Cannot derive company domain")

    # Step 4: Generic name detection
    if _is_generic_name(name_lower):
        score -= 15
        intel.trust_factors["negative"].append(
            "Generic company name pattern (could be anyone)"
        )

    # Step 5: Name quality checks
    if len(name) > 3 and name[0].isupper():
        score += 5
        intel.trust_factors["positive"].append("Proper company name format")

    if len(name) < 3:
        score -= 10
        intel.trust_factors["negative"].append(
            "Suspiciously short company name"
        )

    if re.search(r'\d{3,}', name):
        score -= 5
        intel.trust_factors["negative"].append(
            "Numbers in company name (unusual)"
        )

    # Step 6: External signals
    if employee_count > 0:
        if employee_count > 1000:
            score += 15
            intel.trust_factors["positive"].append(
                f"Large company ({employee_count}+ employees)"
            )
        elif employee_count > 100:
            score += 10
            intel.trust_factors["positive"].append(
                f"Established company ({employee_count}+ employees)"
            )
        else:
            score += 5
            intel.trust_factors["positive"].append(
                f"{employee_count} employees listed"
            )

    if has_linkedin:
        score += 15
        intel.trust_factors["positive"].append("Has LinkedIn company page")

    intel.trust_score = round(min(100, max(0, score)), 2)
    return intel


# HELPER FUNCTIONS

def _derive_domain(name_lower: str) -> str:
    """
    Guess company domain from name.

    Examples:
        "acme corp"             -> "acme.com"
        "xyz pvt ltd"           -> "xyz.com"
        "tech solutions india"  -> "techsolutionsindia.com"
    """
    cleaned = name_lower
    for suffix in LEGAL_SUFFIXES:
        pattern = r'\s*[,.]?\s*' + re.escape(suffix) + r'\s*$'
        cleaned = re.sub(pattern, '', cleaned)

    slug = re.sub(r'[^a-z0-9]', '', cleaned)
    if not slug:
        return ""

    return f"{slug}.com"


def _check_domain_exists(domain: str, timeout: int = 3) -> bool:
    """
    Check if a domain exists via DNS lookup.
    Fast (< 3s), free. Only confirms DNS resolves, not site responsiveness.
    """
    if not domain:
        return False

    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False


def _is_generic_name(name_lower: str) -> bool:
    """Check if name matches a generic pattern (red flag)."""
    return any(re.search(p, name_lower) for p in GENERIC_NAME_PATTERNS)


# SELF-TEST

def _self_test():
    """Run a quick test of the trust scoring. Requires internet for DNS."""
    print("=" * 70)
    print("COMPANY TRUST - SELF-TEST")
    print("=" * 70)

    test_companies = [
        # Known companies (high trust)
        "Google",
        "Tata Consultancy Services",
        "Infosys",
        "Flipkart Pvt Ltd",
        "Zomato",

        # Generic names (low trust)
        "IT Solutions Pvt Ltd",
        "ABC Software Solutions",
        "Tech Consultancy Services",

        # Real-looking unknown
        "Brightline Software",
        "NovaTech Innovations",

        # Edge cases
        "XYZ",
        "Company123456",
        "",
        "Unknown",
    ]

    for name in test_companies:
        print(f"\n{'-' * 70}")
        print(f"Company: {name!r}")

        intel = compute_company_trust(name)

        if intel.trust_score >= 80:
            label = "[HIGH]   "
        elif intel.trust_score >= 50:
            label = "[MEDIUM] "
        elif intel.trust_score >= 25:
            label = "[LOW]    "
        else:
            label = "[NONE]   "

        print(f"  Trust Score: {label}{intel.trust_score}/100")
        print(f"  Domain:      {intel.domain or 'N/A'}")
        print(f"  Active:      {intel.website_active}")

        if intel.trust_factors["positive"]:
            print(f"  Positive:")
            for f in intel.trust_factors["positive"]:
                print(f"     - {f}")

        if intel.trust_factors["negative"]:
            print(f"  Negative:")
            for f in intel.trust_factors["negative"]:
                print(f"     - {f}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
