#location_normalizer.py


from dataclasses import dataclass


# DATA STRUCTURE

@dataclass
class NormalizedLocation:
    """Structured, normalized location."""
    city:    str = ""
    state:   str = ""
    country: str = "India"
    raw:     str = ""


# CITY MAPPINGS (Knowledge Base)

CITY_ALIAS_MAP = {
    # Delhi NCR
    "delhi ncr":      ("Delhi",          "Delhi"),
    "new delhi":      ("Delhi",          "Delhi"),
    "delhi":          ("Delhi",          "Delhi"),
    "gurugram":       ("Gurugram",       "Haryana"),
    "gurgaon":        ("Gurugram",       "Haryana"),
    "greater noida":  ("Greater Noida",  "Uttar Pradesh"),
    "noida":          ("Noida",          "Uttar Pradesh"),
    "faridabad":      ("Faridabad",      "Haryana"),
    "ghaziabad":      ("Ghaziabad",      "Uttar Pradesh"),

    # Mumbai region
    "navi mumbai":    ("Navi Mumbai",    "Maharashtra"),
    "mumbai":         ("Mumbai",         "Maharashtra"),
    "bombay":         ("Mumbai",         "Maharashtra"),
    "thane":          ("Thane",          "Maharashtra"),

    # Pune
    "pune":           ("Pune",           "Maharashtra"),
    "pimpri":         ("Pimpri-Chinchwad", "Maharashtra"),

    # South India
    "bengaluru":      ("Bengaluru",      "Karnataka"),
    "bangalore":      ("Bengaluru",      "Karnataka"),
    "hyderabad":      ("Hyderabad",      "Telangana"),
    "secunderabad":   ("Hyderabad",      "Telangana"),
    "chennai":        ("Chennai",        "Tamil Nadu"),
    "madras":         ("Chennai",        "Tamil Nadu"),
    "coimbatore":     ("Coimbatore",     "Tamil Nadu"),
    "kochi":          ("Kochi",          "Kerala"),
    "cochin":         ("Kochi",          "Kerala"),
    "trivandrum":     ("Thiruvananthapuram", "Kerala"),
    "thiruvananthapuram": ("Thiruvananthapuram", "Kerala"),
    "mysore":         ("Mysuru",         "Karnataka"),
    "mysuru":         ("Mysuru",         "Karnataka"),
    "mangalore":      ("Mangaluru",      "Karnataka"),
    "vijayawada":     ("Vijayawada",     "Andhra Pradesh"),
    "visakhapatnam":  ("Visakhapatnam",  "Andhra Pradesh"),
    "vizag":          ("Visakhapatnam",  "Andhra Pradesh"),

    # East India
    "kolkata":        ("Kolkata",        "West Bengal"),
    "calcutta":       ("Kolkata",        "West Bengal"),
    "bhubaneswar":    ("Bhubaneswar",    "Odisha"),
    "guwahati":       ("Guwahati",       "Assam"),
    "patna":          ("Patna",          "Bihar"),
    "ranchi":         ("Ranchi",         "Jharkhand"),
    "jamshedpur":     ("Jamshedpur",     "Jharkhand"),

    # West India
    "ahmedabad":      ("Ahmedabad",      "Gujarat"),
    "surat":          ("Surat",          "Gujarat"),
    "vadodara":       ("Vadodara",       "Gujarat"),
    "baroda":         ("Vadodara",       "Gujarat"),
    "rajkot":         ("Rajkot",         "Gujarat"),
    "jaipur":         ("Jaipur",         "Rajasthan"),
    "udaipur":        ("Udaipur",        "Rajasthan"),
    "jodhpur":        ("Jodhpur",        "Rajasthan"),

    # Central India
    "bhopal":         ("Bhopal",         "Madhya Pradesh"),
    "indore":         ("Indore",         "Madhya Pradesh"),
    "nagpur":         ("Nagpur",         "Maharashtra"),
    "raipur":         ("Raipur",         "Chhattisgarh"),

    # North India
    "lucknow":        ("Lucknow",        "Uttar Pradesh"),
    "kanpur":         ("Kanpur",         "Uttar Pradesh"),
    "varanasi":       ("Varanasi",       "Uttar Pradesh"),
    "agra":           ("Agra",           "Uttar Pradesh"),
    "chandigarh":     ("Chandigarh",     "Chandigarh"),
    "mohali":         ("Mohali",         "Punjab"),
    "ludhiana":       ("Ludhiana",       "Punjab"),
    "amritsar":       ("Amritsar",       "Punjab"),
    "dehradun":       ("Dehradun",       "Uttarakhand"),
    "shimla":         ("Shimla",         "Himachal Pradesh"),

    # Special cases
    "work from home": ("Remote",         "Remote"),
    "wfh":            ("Remote",         "Remote"),
    "remote":         ("Remote",         "Remote"),
    "anywhere":       ("Remote",         "Remote"),
    "any location":   ("Remote",         "Remote"),
    "pan india":      ("Pan India",      "Pan India"),
    "across india":   ("Pan India",      "Pan India"),
    "multiple cities":("Multiple",       "Multiple"),
}


# PRE-COMPUTED SORTED ALIASES (longest first)

_SORTED_ALIASES = sorted(
    CITY_ALIAS_MAP.items(),
    key=lambda x: len(x[0]),
    reverse=True,
)


# MAIN FUNCTION

def normalize_location(raw: str) -> NormalizedLocation:
    """
    Convert a raw location string into a structured NormalizedLocation.
    """
    if not raw or not str(raw).strip():
        return NormalizedLocation(raw=raw or "")

    key = str(raw).strip().lower()

    for alias, (city, state) in _SORTED_ALIASES:
        if alias in key:
            return NormalizedLocation(
                city=city,
                state=state,
                country="India",
                raw=raw
            )

    return NormalizedLocation(
        city=str(raw).strip().title(),
        state="",
        country="India",
        raw=raw
    )


# UTILITY FUNCTION

def is_remote_location(raw: str) -> bool:
    """Return True if location string indicates remote work."""
    if not raw:
        return False

    remote_keywords = ["remote", "work from home", "wfh", "anywhere", "any location"]
    key = str(raw).lower()
    return any(kw in key for kw in remote_keywords)


# SELF-TEST

def _self_test():
    print("=" * 70)
    print("LOCATION NORMALIZER - SELF-TEST")
    print("=" * 70)

    test_cases = [
        "Bangalore",
        "Bengaluru",
        "bangalore, india",
        "Delhi NCR",
        "New Delhi",
        "Gurgaon",
        "Greater Noida",
        "Mumbai, Maharashtra",
        "Work From Home",
        "Remote",
        "Pan India",
        "Some Random City",
        "",
        None,
        "   ",
    ]

    for raw in test_cases:
        try:
            result = normalize_location(raw)
            print(f"\nInput:  {raw!r}")
            print(f"  City:    {result.city}")
            print(f"  State:   {result.state}")
            print(f"  Country: {result.country}")
        except Exception as e:
            print(f"\nInput {raw!r} caused error: {e}")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
