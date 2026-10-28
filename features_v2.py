"""
PhishEye feature set v2: 54 URL-only features.

Why URL-only? The extension must score a page BEFORE it loads, so we can't use the page-content
columns that ship with PhiUSIIL (HTML title, line count, etc.). Every feature below can be computed
from the URL string alone, so training and live inference use the exact same code.
"""
import ipaddress
import math
import re
from collections import Counter
from typing import Any, Dict, List
from urllib.parse import urlparse

FEATURE_NAMES_V2: List[str] = [
    # --- lengths (5) ---
    "url_length", "hostname_length", "path_length", "query_length", "fragment_length",
    # --- character counts (15) ---
    "count_dots", "count_hyphens", "count_at", "count_qm", "count_and", "count_equal",
    "count_percent", "count_slash", "count_underscore", "count_tilde", "count_hash",
    "count_colon", "count_semicolon", "count_comma", "count_plus",
    # --- ratios / entropy (8) ---
    "count_digits", "count_letters", "digit_ratio", "letter_ratio", "special_char_ratio",
    "uppercase_ratio", "entropy_url", "entropy_host",
    # --- more entropy + host structure (13) ---
    "entropy_path", "has_ip", "has_hex_ip", "has_https", "has_port", "count_subdomains",
    "host_digit_count", "host_hyphen_count", "host_digit_ratio", "tld_length", "is_risky_tld",
    "is_punycode", "vowel_ratio_host",
    # --- keywords / brand / tokens (9) ---
    "has_suspicious_words", "suspicious_word_count", "brand_in_subdomain_or_path", "count_params",
    "longest_token_len", "avg_token_len", "token_count", "longest_host_label", "path_depth",
    # --- tricks (4) ---
    "has_double_slash_in_path", "has_executable_ext", "is_shortener", "has_embedded_url",
]
assert len(FEATURE_NAMES_V2) == 54, len(FEATURE_NAMES_V2)

SUSPICIOUS_WORDS = (
    "login", "signin", "sign-in", "verify", "verification", "secure", "account", "update",
    "banking", "wallet", "confirm", "password", "authenticate", "credential", "recovery",
    "support", "service", "ebayisapi", "webscr", "suspend", "unlock", "billing", "invoice",
    "alert", "security", "validate", "restore", "limited",
)
RISKY_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "buzz", "cam", "icu", "rest", "fit",
    "surf", "click", "live", "space", "monster", "cyou", "sbs", "cfd", "support", "zip", "mov",
    "country", "kim", "loan", "men", "party", "review", "stream", "download", "racing",
}
BRANDS = (
    "paypal", "google", "microsoft", "apple", "amazon", "netflix", "facebook", "instagram",
    "whatsapp", "linkedin", "twitter", "hdfc", "icici", "sbi", "axis", "flipkart", "paytm",
    "phonepe", "outlook", "office365", "dropbox", "adobe", "ebay", "walmart", "chase",
    "wellsfargo", "bankofamerica", "dhl", "fedex", "ups", "irs", "binance", "coinbase", "metamask",
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id", "lnkd.in", "t.ly",
}
EXEC_EXT = re.compile(r"\.(exe|msi|scr|bat|cmd|apk|jar|ps1|vbs|hta|dmg|iso|zip|rar)(\?|$)", re.I)
MULTI_TLD = {"co.in", "ac.in", "gov.in", "org.in", "net.in", "co.uk", "org.uk", "ac.uk",
             "com.au", "co.jp", "com.br", "co.nz", "com.sg", "co.za"}
VOWELS = set("aeiou")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def registered_domain(host: str) -> str:
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in MULTI_TLD and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_url_v2(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def extract_features_v2(url: str) -> Dict[str, Any]:
    raw = (url or "").strip()
    out: Dict[str, Any] = {n: 0 for n in FEATURE_NAMES_V2}
    if not raw:
        return out
    norm = normalize_url_v2(raw)
    try:
        p = urlparse(norm)
        scheme, host, path = p.scheme.lower(), (p.hostname or "").lower(), p.path or ""
        query, frag = p.query or "", p.fragment or ""
        try:
            has_port = 1 if p.port else 0
        except ValueError:
            has_port = 0
    except ValueError:
        scheme, host, path, query, frag, has_port = "", "", "", "", "", 0

    low = raw.lower()
    n = max(1, len(raw))
    digits = sum(c.isdigit() for c in raw)
    letters = sum(c.isalpha() for c in raw)
    upper = sum(c.isupper() for c in raw)

    ip_flag = 0
    try:
        ipaddress.ip_address(host.strip("[]"))
        ip_flag = 1
    except ValueError:
        pass
    hex_ip = 1 if re.search(r"(^|//)0x[0-9a-f]+(\.|/|$)", low) or re.match(r"^\d{8,10}$", host) else 0

    labels = [l for l in host.split(".") if l]
    tld = labels[-1] if len(labels) > 1 and not ip_flag else ""
    reg = registered_domain(host)
    sub_part = host[: -len(reg)].rstrip(".") if reg and host.endswith(reg) else ""
    sub_count = len([l for l in sub_part.split(".") if l]) if not ip_flag else 0

    tokens = [t for t in re.split(r"[/._\-?=&#:;,+%~@]+", low) if t]
    host_letters = [c for c in host if c.isalpha()]
    host_digits = sum(c.isdigit() for c in host)

    sus_hits = sum(low.count(w) for w in SUSPICIOUS_WORDS)
    brand_flag = 0
    if not ip_flag:
        reg_name = reg.split(".")[0] if reg else ""
        tail = sub_part + "/" + path.lower() + "?" + query.lower()
        brand_flag = 1 if any(b in tail and b not in reg_name for b in BRANDS) else 0
        if not brand_flag and any(b in reg_name and reg_name != b for b in BRANDS):
            brand_flag = 1  # e.g. paypal-secure-login.com

    out.update({
        "url_length": len(raw), "hostname_length": len(host), "path_length": len(path),
        "query_length": len(query), "fragment_length": len(frag),
        "count_dots": raw.count("."), "count_hyphens": raw.count("-"), "count_at": raw.count("@"),
        "count_qm": raw.count("?"), "count_and": raw.count("&"), "count_equal": raw.count("="),
        "count_percent": raw.count("%"), "count_slash": raw.count("/"),
        "count_underscore": raw.count("_"), "count_tilde": raw.count("~"), "count_hash": raw.count("#"),
        "count_colon": raw.count(":"), "count_semicolon": raw.count(";"), "count_comma": raw.count(","),
        "count_plus": raw.count("+"),
        "count_digits": digits, "count_letters": letters,
        "digit_ratio": round(digits / n, 4), "letter_ratio": round(letters / n, 4),
        "special_char_ratio": round((n - digits - letters) / n, 4),
        "uppercase_ratio": round(upper / n, 4),
        "entropy_url": round(_entropy(raw), 4), "entropy_host": round(_entropy(host), 4),
        "entropy_path": round(_entropy(path), 4),
        "has_ip": ip_flag, "has_hex_ip": hex_ip, "has_https": 1 if scheme == "https" else 0,
        "has_port": has_port, "count_subdomains": sub_count,
        "host_digit_count": host_digits, "host_hyphen_count": host.count("-"),
        "host_digit_ratio": round(host_digits / max(1, len(host)), 4),
        "tld_length": len(tld), "is_risky_tld": 1 if tld in RISKY_TLDS else 0,
        "is_punycode": 1 if "xn--" in host else 0,
        "vowel_ratio_host": round(sum(c in VOWELS for c in host_letters) / max(1, len(host_letters)), 4),
        "has_suspicious_words": 1 if sus_hits else 0, "suspicious_word_count": sus_hits,
        "brand_in_subdomain_or_path": brand_flag,
        "count_params": len([q for q in query.split("&") if q]) if query else 0,
        "longest_token_len": max((len(t) for t in tokens), default=0),
        "avg_token_len": round(sum(len(t) for t in tokens) / max(1, len(tokens)), 3),
        "token_count": len(tokens),
        "longest_host_label": max((len(l) for l in labels), default=0),
        "path_depth": len([s for s in path.split("/") if s]),
        "has_double_slash_in_path": 1 if "//" in path else 0,
        "has_executable_ext": 1 if EXEC_EXT.search(path + ("?" + query if query else "")) else 0,
        "is_shortener": 1 if reg in SHORTENERS else 0,
        "has_embedded_url": 1 if re.search(r"https?(://|%3a%2f%2f)", (path + "?" + query).lower()) else 0,
    })
    return out
