import os
import re
import pandas as pd
from typing import Dict, Any
from pathlib import Path

# Locate paths relative to models directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOMAIN_CSV_PATH = MODELS_DIR / "domain_analysis.csv"

# Suspicious TLDs list
SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".work", ".loan", ".win"}

def _derive_domain(name_lower: str) -> str:
    """
    Derive domain from company name (e.g. Acme Corp -> acme.com).
    """
    cleaned = name_lower.strip().lower()
    legal_suffixes = [
        'pvt ltd', 'pvtltd', 'private limited', 'privatelimited',
        'ltd', 'limited', 'inc', 'incorporated', 'corporation',
        'llp', 'llc', 'gmbh', 'co', 'company',
    ]
    for suffix in legal_suffixes:
        pattern = r'\s*[,.]?\s*' + re.escape(suffix) + r'\s*$'
        cleaned = re.sub(pattern, '', cleaned)

    slug = re.sub(r'[^a-z0-9]', '', cleaned)
    if not slug:
        return ""

    return f"{slug}.com"

def analyze_domain(domain: str) -> Dict[str, Any]:
    """
    Perform deep analysis on a domain using WHOIS and SSL data.
    Caches results in domain_analysis.csv to avoid external API spam.
    """
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
        
    # Check CSV Cache
    if DOMAIN_CSV_PATH.exists():
        try:
            df = pd.read_csv(DOMAIN_CSV_PATH)
            if "domain" in df.columns:
                match = df[df["domain"] == domain]
                if not match.empty:
                    rec = match.iloc[0].to_dict()
                    # Ensure correct typing
                    return {
                        "domain": str(rec.get("domain", "")),
                        "domain_age": int(rec.get("domain_age", 0)),
                        "ssl_valid": bool(rec.get("ssl_valid", False)),
                        "whois_available": bool(rec.get("whois_available", False)),
                        "suspicious_tld": bool(rec.get("suspicious_tld", False)),
                        "domain_reputation_score": float(rec.get("domain_reputation_score", 0.0)),
                        "domain_risk_score": float(rec.get("domain_risk_score", 100.0))
                    }
        except Exception:
            pass
            
    # Perform checks
    from .whois_checker import get_whois_info
    from .ssl_checker import check_ssl
    
    whois_info = get_whois_info(domain)
    ssl_info = check_ssl(domain)
    
    tld_match = False
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            tld_match = True
            break
            
    # Calculate scores
    risk_score = 0.0
    if tld_match:
        risk_score += 40
    if not ssl_info["ssl_valid"]:
        risk_score += 30
    if not whois_info["whois_available"]:
        risk_score += 20
        
    # Age-based risk
    age_days = whois_info["domain_age_days"]
    if age_days < 180: # less than 6 months
        risk_score += 20
    elif age_days < 365: # less than 1 year
        risk_score += 10
        
    risk_score = min(100.0, max(0.0, risk_score))
    reputation_score = 100.0 - risk_score
    
    result = {
        "domain": domain,
        "domain_age": age_days,
        "ssl_valid": ssl_info["ssl_valid"],
        "whois_available": whois_info["whois_available"],
        "suspicious_tld": tld_match,
        "domain_reputation_score": reputation_score,
        "domain_risk_score": risk_score
    }
    
    # Save to Cache
    try:
        if DOMAIN_CSV_PATH.exists():
            df = pd.read_csv(DOMAIN_CSV_PATH)
            df = df[df["domain"] != domain] # Remove old record
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        df.to_csv(DOMAIN_CSV_PATH, index=False)
    except Exception:
        pass
        
    return result
