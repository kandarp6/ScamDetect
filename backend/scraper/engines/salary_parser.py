#salary_parser.py


import re
from dataclasses import dataclass, field


# DATA STRUCTURE

@dataclass
class ParsedSalary:
    """Structured salary data extracted from raw text."""
    min_amount:    float = 0.0
    max_amount:    float = 0.0
    currency:      str   = "INR"
    period:        str   = "year"   # Always normalized to yearly
    is_unpaid:     bool  = False
    is_suspicious: bool  = False
    raw:           str   = ""
    flags:         list  = field(default_factory=list)


# CONSTANTS

# Suspicious salary patterns (FRAUD signals)
SUSPICIOUS_PATTERNS = [
    (r'earn\s+\d+[k,]*\s*(daily|per\s*day|/day)',    "Unrealistic daily earnings claim"),
    (r'earn\s+\d+[k,]*\s*(weekly|per\s*week|/week)', "Unrealistic weekly earnings claim"),
    (r'\d+[k,]*\s*lakh\s*per\s*(day|week)',          "Lakhs per day/week (impossible)"),
    (r'unlimited\s*(earning|income|salary)',          "Unlimited earnings (red flag)"),
    (r'uncapped\s*(earning|income|salary)',           "Uncapped earnings (vague)"),
    (r'guaranteed\s+\d+',                             "Guaranteed earnings claim"),
    (r'no\s+limit\s+(earn|income)',                   "No limit earnings (suspicious)"),
]

# Vague / unclear salary patterns (low quality, NOT scam)
VAGUE_PATTERNS = [
    "best in industry",
    "as per industry standard",
    "as per company norms",
    "as per experience",
    "negotiable",
    "competitive",
    "to be discussed",
    "tbd",
    "tba",
    "not disclosed",
    "not specified",
    "confidential",
]

UNPAID_KEYWORDS = [
    "unpaid",
    "no stipend",
    "without stipend",
    "no pay",
    "voluntary",
    "free internship",
]

CURRENCY_MAP = {
    "₹":   "INR",
    "rs":  "INR",
    "rs.": "INR",
    "inr": "INR",
    "$":   "USD",
    "usd": "USD",
    "€":   "EUR",
    "eur": "EUR",
    "£":   "GBP",
    "gbp": "GBP",
}

# Period normalization multipliers (period -> times-per-year)
PERIOD_MULTIPLIERS = {
    "year":    1,
    "annual":  1,
    "annum":   1,
    "yr":      1,
    "pa":      1,
    "p.a":     1,
    "p.a.":    1,
    "lpa":     1,         # Lakhs Per Annum
    "month":   12,
    "monthly": 12,
    "/month":  12,
    "pm":      12,
    "p.m":     12,
    "p.m.":    12,
    "mo":      12,
    "week":    52,
    "weekly":  52,
    "/week":   52,
    "day":     365,
    "daily":   365,
    "/day":    365,
    "hour":    2080,      # 40hrs/week x 52 weeks
    "hourly":  2080,
    "/hr":     2080,
    "/hour":   2080,
}


# MAIN PARSER

def parse_salary(raw: str) -> ParsedSalary:
    """
    Parse a raw salary string into a ParsedSalary object.

    All numeric amounts are normalized to ANNUAL INR.
    """
    result = ParsedSalary(raw=raw or "")

    if not raw or not raw.strip():
        result.flags.append("Empty salary field")
        return result

    raw_lower = raw.lower().strip()

    # Unpaid check
    if any(kw in raw_lower for kw in UNPAID_KEYWORDS):
        result.is_unpaid = True
        result.flags.append("Unpaid position")
        return result

    # Suspicious / fraud patterns
    for pattern, warning in SUSPICIOUS_PATTERNS:
        if re.search(pattern, raw_lower):
            result.is_suspicious = True
            result.flags.append(warning)
            # don't return - still try to extract numbers

    # Vague descriptions
    if any(vp in raw_lower for vp in VAGUE_PATTERNS):
        result.flags.append("Vague salary description")
        return result

    # Currency + period
    result.currency = _detect_currency(raw_lower)
    detected_period, multiplier = _detect_period(raw_lower)

    # Extract numbers
    is_lpa_context = "lpa" in raw_lower or "lakh" in raw_lower
    numbers = _extract_numbers(raw_lower, is_lpa=is_lpa_context)

    if not numbers:
        result.flags.append("Could not extract salary numbers")
        return result

    # Min and max
    if len(numbers) == 1:
        result.min_amount = numbers[0]
        result.max_amount = numbers[0]
    else:
        result.min_amount = min(numbers)
        result.max_amount = max(numbers)

    # Normalize to annual amount
    result.min_amount = round(result.min_amount * multiplier, 2)
    result.max_amount = round(result.max_amount * multiplier, 2)
    result.period = "year"

    # Sanity checks
    if result.max_amount > 500_000_000:
        result.is_suspicious = True
        result.flags.append("Salary exceeds Rs 50 crore (unrealistic)")

    if result.min_amount > 0 and result.min_amount < 6000:
        result.flags.append("Salary below minimum wage (possibly scam)")

    return result


# HELPER FUNCTIONS

def _detect_currency(text: str) -> str:
    """Return currency code from text. Default INR."""
    for symbol, code in CURRENCY_MAP.items():
        if symbol in text:
            return code
    return "INR"


def _detect_period(text: str) -> tuple[str, int]:
    """
    Detect time period and return its annual multiplier.

    Examples:
        "Rs 20k/month"  -> ("month", 12)
        "Rs 15 LPA"     -> ("year", 1)
        "Rs 500 daily"  -> ("day", 365)
    """
    # Longest match first so "monthly" beats "month"
    sorted_periods = sorted(PERIOD_MULTIPLIERS.items(), key=lambda x: len(x[0]), reverse=True)

    for period, mult in sorted_periods:
        if len(period) <= 3:
            if re.search(r'\b' + re.escape(period) + r'\b', text):
                return period, mult
        else:
            if period in text:
                return period, mult

    # Default for Indian internships
    return "month", 12


def _extract_numbers(text: str, is_lpa: bool = False) -> list:
    """
    Extract numeric values from salary text.

    Handles Indian conventions:
        - Commas: "1,00,000" -> 100000
        - 'k' suffix:    "20k" -> 20000
        - 'L' suffix:    "5L"  -> 500000
        - 'C' suffix:    "1C"  -> 10000000
        - In LPA context, bare numbers <1000 treated as lakhs
    """
    cleaned = re.sub(r'[₹$€£,]', '', text)

    pattern = r'(\d+(?:\.\d+)?)\s*([kKlLcC]?)'
    matches = re.findall(pattern, cleaned)

    numbers = []
    for value_str, suffix in matches:
        try:
            value = float(value_str)
            if value == 0:
                continue

            if suffix.lower() == 'k':
                value *= 1000
            elif suffix.lower() == 'l':
                value *= 100000
            elif suffix.lower() == 'c':
                value *= 10000000
            elif is_lpa and value < 1000:
                # In LPA context, "15" means 15 lakhs
                value *= 100000

            numbers.append(value)
        except ValueError:
            continue

    # Filter unrealistic numbers (likely page IDs, etc.)
    numbers = [n for n in numbers if 100 <= n <= 100_000_000]
    return numbers


# DISPLAY HELPERS

def format_salary_inr(amount: float) -> str:
    """
    Format a number as an Indian Rupee string.

    Examples:
        1500000  -> "Rs 15.00 LPA"
        50000    -> "Rs 50k"
        15000000 -> "Rs 1.50 Cr"
    """
    if amount <= 0:
        return "Not specified"

    if amount >= 10_000_000:
        return f"Rs {amount/10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"Rs {amount/100_000:.2f} LPA"
    elif amount >= 1000:
        return f"Rs {amount/1000:.0f}k"
    else:
        return f"Rs {amount:.0f}"


def salary_summary(parsed: ParsedSalary) -> str:
    """Human-readable summary of a parsed salary."""
    if parsed.is_unpaid:
        return "Unpaid internship"

    if parsed.is_suspicious:
        return f"SUSPICIOUS: {parsed.raw}"

    if parsed.min_amount == 0 and parsed.max_amount == 0:
        return "Salary not disclosed"

    if parsed.min_amount == parsed.max_amount:
        return format_salary_inr(parsed.min_amount)

    return f"{format_salary_inr(parsed.min_amount)} - {format_salary_inr(parsed.max_amount)}"


# SELF-TEST

def _self_test():
    print("=" * 70)
    print("SALARY PARSER - SELF-TEST")
    print("=" * 70)

    test_cases = [
        # LPA / annual lakh formats
        "Rs 15-25 LPA",
        "8-12 LPA",
        "5 LPA",
        "Rs 3-5 Lakh per annum",

        # Monthly
        "Rs 6,000 - 33,000 /month",
        "20k-30k/month",
        "Rs 50,000 per month",
        "Rs. 25,000 monthly",

        # Unpaid
        "Unpaid",
        "No stipend",
        "Voluntary",

        # Vague
        "Best in industry",
        "Negotiable",
        "As per company norms",
        "Not disclosed",

        # Suspicious fraud signals
        "Earn Rs 50,000 daily working from home!",
        "Unlimited earnings opportunity",
        "Guaranteed Rs 1 lakh per week",
        "Uncapped salary",

        # Government style
        "Level 7 - Rs 56,100 per month",

        # Edge cases
        "",
        "Salary",
        "Hire 12345",

        # USD / hourly / single value
        "$50,000 - $80,000 per year",
        "Rs 500 per hour",
        "Rs 40,000",
    ]

    print(f"\n{'Raw Input':<50s} {'Parsed Output':<50s}")
    print("=" * 100)

    for raw in test_cases:
        parsed = parse_salary(raw)
        summary = salary_summary(parsed)

        prefix = "[!]" if parsed.is_suspicious else "   "
        print(f"{prefix} {raw[:48]:<48s} -> {summary}")

        if parsed.flags:
            for flag in parsed.flags:
                print(f"      {' ' * 48} {flag}")

    print("\n" + "=" * 70)
    print("DETAILED EXAMPLE:")
    print("=" * 70)

    example = parse_salary("Rs 15-25 LPA")
    print(f"\nInput: 'Rs 15-25 LPA'")
    print(f"  min_amount: {example.min_amount}")
    print(f"  max_amount: {example.max_amount}")
    print(f"  currency:   {example.currency}")
    print(f"  period:     {example.period}")
    print(f"  formatted:  {salary_summary(example)}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
