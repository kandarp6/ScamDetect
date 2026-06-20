import socket
import re
from datetime import datetime
from typing import Dict, Any, Optional

def query_whois(domain: str) -> str:
    """
    Query WHOIS server for a domain name.
    Queries whois.iana.org to find the referral WHOIS server, then queries that server.
    """
    domain = domain.strip().lower()
    if not domain:
        return ""
    
    # Handle subdomains by extracting top-level and second-level domain
    parts = domain.split(".")
    if len(parts) > 2:
        # e.g. sub.example.com -> example.com
        domain = ".".join(parts[-2:])
        
    try:
        # Query IANA first
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(("whois.iana.org", 43))
        s.sendall((domain + "\r\n").encode("utf-8"))
        
        response = b""
        while True:
            data = s.recv(4096)
            if not data:
                break
            response += data
        s.close()
        
        res_text = response.decode("utf-8", errors="ignore")
        
        # Look for referral WHOIS server
        referral = None
        for line in res_text.splitlines():
            line = line.strip()
            if line.startswith("refer:") or line.startswith("whois:"):
                referral = line.split(":", 1)[1].strip()
                break
                
        # Connect to referral server or fall back
        whois_server = referral if referral else "whois.verisign-grs.com"
        if domain.endswith(".in"):
            whois_server = "whois.inregistry.net"
            
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((whois_server, 43))
        s.sendall((domain + "\r\n").encode("utf-8"))
        
        response = b""
        while True:
            data = s.recv(4096)
            if not data:
                break
            response += data
        s.close()
        
        return response.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_creation_date(whois_text: str) -> Optional[datetime]:
    """
    Extract the domain creation/registration date from WHOIS response text.
    """
    if not whois_text:
        return None
        
    # Match lines like "Creation Date: 2020-03-24T12:00:00Z" or "Created On: 24-Mar-2020"
    patterns = [
        r"(?:Creation Date|Created On|Registered on|Registration Time|Domain Name Commencement Date|Create Date|Created Date|Created|Registered)\s*:\s*([^\r\n]+)",
        r"(?:created|registered)\s*\.+\s*:\s*([^\r\n]+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, whois_text, re.IGNORECASE)
        for val in matches:
            val = val.strip()
            # Split time indicators
            date_clean = re.split(r'[T\s]', val)[0]
            # Try multiple parsing patterns
            for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y.%m.%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"):
                try:
                    return datetime.strptime(date_clean, fmt)
                except ValueError:
                    continue
                    
    return None

def get_whois_info(domain: str) -> Dict[str, Any]:
    """
    Public entry point for checking WHOIS details of a domain.
    """
    whois_text = query_whois(domain)
    whois_available = len(whois_text) > 0
    creation_date = extract_creation_date(whois_text)
    
    domain_age_days = 0
    if creation_date:
        domain_age_days = (datetime.now() - creation_date).days
        
    return {
        "whois_available": whois_available,
        "creation_date": creation_date.isoformat() if creation_date else None,
        "domain_age_days": max(0, domain_age_days),
        "raw_whois": whois_text[:1000]
    }
