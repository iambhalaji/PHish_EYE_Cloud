import math
import ipaddress
from collections import Counter
from urllib.parse import urlparse
from typing import Dict, Any, List
import pandas as pd
FEATURE_NAMES: List[str] = [
    "url_length",            
    "hostname_length",       
    "count_dots",            
    "count_hyphens",         
    "count_at",              
    "count_qm",              
    "count_and",             
    "has_ip",                
    "entropy",               
    "count_slash",           
    "count_equal",           
    "count_percent",         
    "count_digits",          
    "count_subdomains",      
    "has_https",             
    "path_length",           
    "has_suspicious_words",  
]

SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "verification", "secure", "account",
    "update", "banking", "wallet", "confirm", "password", "authenticate",
    "credential", "recovery", "support", "service", "ebayisapi", "webscr",
)


def calculate_entropy(text: str) -> float:
    """Shannon entropy — higher = more random / DGA-like."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def is_ip_address(hostname: str) -> int:
    """Returns 1 if hostname is a raw IPv4 / IPv6 address, else 0."""
    if not hostname:
        return 0
    host_clean = hostname.split(":")[0].strip("[]")
    try:
        ipaddress.ip_address(host_clean)
        return 1
    except ValueError:
        return 0


def normalize_url(url: str) -> str:
    """Ensures scheme exists so urlparse works correctly."""
    url = url.strip()
    if not url.startswith(("http://", "https://", "ftp://")):
        url = "http://" + url
    return url


def extract_features(url: str) -> Dict[str, Any]:
    """Extracts all 17 lexical/structural features from a single URL string."""
    if not isinstance(url, str) or not url.strip():
        return {f: 0.0 if f == "entropy" else 0 for f in FEATURE_NAMES}

    raw = url.strip()
    norm = normalize_url(raw)

    scheme = ""
    try:
        parsed = urlparse(norm)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
    except Exception:
        hostname = ""
        path     = ""

    ip_flag = is_ip_address(hostname)

    if hostname:
        sub_count = max(0, hostname.count(".") - 1) if not ip_flag else 0
    else:
        sub_count = 0

    return {
        "url_length":           len(raw),
        "hostname_length":      len(hostname),
        "count_dots":           raw.count("."),
        "count_hyphens":        raw.count("-"),
        "count_at":             raw.count("@"),
        "count_qm":             raw.count("?"),
        "count_and":            raw.count("&"),
        "has_ip":               ip_flag,
        "entropy":              round(calculate_entropy(raw), 4),
        "count_slash":          raw.count("/"),
        "count_equal":          raw.count("="),
        "count_percent":        raw.count("%"),
        "count_digits":         sum(c.isdigit() for c in raw),
        "count_subdomains":     sub_count,
        "has_https":            1 if scheme == "https" else 0,
        "path_length":          len(path),
        "has_suspicious_words": 1 if any(kw in raw.lower() for kw in SUSPICIOUS_KEYWORDS) else 0,
    }


def extract_features_dataframe(urls: List[str]) -> pd.DataFrame:
    """Batch-extracts features and returns a DataFrame with columns in FEATURE_NAMES order."""
    return pd.DataFrame([extract_features(u) for u in urls])[FEATURE_NAMES]
