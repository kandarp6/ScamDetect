import ssl
import socket
from datetime import datetime
from typing import Dict, Any

def check_ssl(domain: str, timeout: float = 3.0) -> Dict[str, Any]:
    """
    Check if a domain has a valid SSL certificate.
    """
    domain = domain.strip().lower()
    if not domain:
        return {
            "ssl_valid": False,
            "expiry_date": None,
            "days_to_expiry": 0,
            "issuer": {}
        }
        
    # Handle subdomains or custom port indicators
    parts = domain.split(".")
    if len(parts) > 2:
        domain = ".".join(parts[-2:])
        
    context = ssl.create_default_context()
    try:
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                # cert is a dictionary when handshake succeeded
                notAfter_str = cert.get('notAfter')
                if notAfter_str:
                    # Parse GMT date string (e.g. "May  4 23:59:59 2026 GMT")
                    expiry_date = datetime.strptime(notAfter_str, '%b %d %H:%M:%S %Y %Z')
                    ssl_valid = expiry_date > datetime.now()
                    days_to_expiry = (expiry_date - datetime.now()).days
                    return {
                        "ssl_valid": ssl_valid,
                        "expiry_date": expiry_date.isoformat(),
                        "days_to_expiry": max(0, days_to_expiry),
                        "issuer": dict(x[0] for x in cert.get('issuer', []))
                    }
    except Exception:
        pass
        
    return {
        "ssl_valid": False,
        "expiry_date": None,
        "days_to_expiry": 0,
        "issuer": {}
    }
