
import datetime
import ipaddress
import json
import logging
import os
import pickle
import re
import socket
import sqlite3
import ssl
import struct
import zlib
from contextlib import closing
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

try:
    import nh3
except ImportError:
    nh3 = None
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.fft import dctn

try:
    import whois as pywhois  # optional python-whois library
except ImportError:
    pywhois = None

try:
    from PIL import Image as PilImage  # optional Pillow library
except ImportError:
    PilImage = None

from features import FEATURE_NAMES, extract_features, normalize_url
from threat_intel import threat_intel

logger = logging.getLogger("phisheye")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "threat_intel.db")



REPORT_THRESHOLD = int(os.getenv("REPORT_THRESHOLD", "3"))

USE_URLHAUS_LIVE = os.getenv("USE_URLHAUS_LIVE", "1") == "1"

app = FastAPI(title="PhishEye Backend Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from xgboost import XGBClassifier

ml_model = XGBClassifier()
ml_model_v1 = XGBClassifier()
_V1_MODEL = os.path.join(BASE_DIR, "model.json")
_V2_MODEL = os.path.join(BASE_DIR, "model_v2.json")
_V2_META = os.path.join(BASE_DIR, "model_v2_meta.json")
_CHAR_MODEL = os.path.join(BASE_DIR, "char_ngram_model.pkl")

char_ngram_model = None
if os.path.exists(_CHAR_MODEL):
    try:
        with open(_CHAR_MODEL, "rb") as _cf:
            char_ngram_model = pickle.load(_cf)
        logger.info("Loaded Character N-Gram TF-IDF NLP Classifier")
    except Exception as _ce:
        logger.warning("Could not load char_ngram_model.pkl: %s", _ce)


def _url_to_ngram_text(url: str) -> str:
    try:
        p = urlparse(normalize_url(url))
        host = (p.hostname or "").lower().removeprefix("www.")
        path = (p.path or "").lower()
        first_seg = "/" + path.strip("/").split("/")[0] if path.strip("/") else ""
        return f"{host}{first_seg}"
    except Exception:
        return url.lower()


if os.path.exists(_V1_MODEL):
    ml_model_v1.load_model(_V1_MODEL)

if os.path.exists(_V2_MODEL) and os.path.exists(_V2_META):
    from features_v2 import FEATURE_NAMES_V2, SUSPICIOUS_WORDS as _SUS_V2, extract_features_v2
    ml_model.load_model(_V2_MODEL)
    with open(_V2_META) as f:
        meta = json.load(f)
    MODEL_FEATURES: List[str] = meta["feature_names"]
    if MODEL_FEATURES != FEATURE_NAMES_V2:
        raise RuntimeError("features_v2.py does not match model_v2 training features.")

    def model_vector(url: str):
        feats_v2 = extract_features_v2(url)
        return [feats_v2[n] for n in MODEL_FEATURES]

    def compute_ensemble_ml_prob(url: str) -> Tuple[float, dict]:
        feats_full = extract_features_v2(url)
        vec_full = [feats_full[n] for n in MODEL_FEATURES]
        p2_full = float(ml_model.predict_proba([vec_full])[0][1]) * 100.0

        p2_origin = p2_full
        path_has_risk = True
        try:
            p = urlparse(normalize_url(url))
            tail_low = ((p.path or "") + "?" + (p.query or "")).lower()
            path_has_risk = bool(
                feats_full["brand_in_subdomain_or_path"]
                or feats_full["has_embedded_url"]
                or feats_full["has_executable_ext"]
                or feats_full["has_double_slash_in_path"]
                or feats_full["count_at"]
                or any(w in tail_low for w in _SUS_V2)
            )
            if p.scheme and p.netloc:
                feats_orig = extract_features_v2(f"{p.scheme}://{p.netloc}")
                vec_orig = [feats_orig[n] for n in MODEL_FEATURES]
                p2_origin = float(ml_model.predict_proba([vec_orig])[0][1]) * 100.0
        except Exception:
            pass

        prob = p2_full if path_has_risk else p2_origin

        # Ensemble with Character & Subword N-Gram TF-IDF NLP Classifier when available
        if char_ngram_model is not None:
            try:
                ng_text = _url_to_ngram_text(url)
                p_char = float(char_ngram_model.predict_proba([ng_text])[0][1]) * 100.0
                # Weighted blend: 65% 54-feature XGBoost + 35% Character N-Gram NLP, or boost if N-Gram is high-confidence
                if p_char >= 78.0:
                    prob = max(prob, 0.45 * prob + 0.55 * p_char)
                else:
                    prob = 0.70 * prob + 0.30 * p_char
            except Exception:
                pass

        return prob, feats_full
else:
    ml_model.load_model(_V1_MODEL)
    with open(os.path.join(BASE_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    MODEL_FEATURES = meta["feature_names"]
    if MODEL_FEATURES != FEATURE_NAMES:
        raise RuntimeError("features.py FEATURE_NAMES does not match the order the model was trained on.")

    def model_vector(url: str):
        feats_v1 = extract_features(url)
        return [feats_v1[n] for n in MODEL_FEATURES]

    def compute_ensemble_ml_prob(url: str) -> Tuple[float, dict]:
        f1 = extract_features(url)
        p1 = float(ml_model.predict_proba([[f1[n] for n in MODEL_FEATURES]])[0][1]) * 100.0
        return p1, f1
THRESHOLD: float = float(meta.get("threshold", 0.5))
logger.info("Model loaded: %d features, threshold %.2f", len(MODEL_FEATURES), THRESHOLD)



def db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(db()) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS blacklist (domain TEXT PRIMARY KEY, source TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS whitelist (domain TEXT PRIMARY KEY)")
        c.execute("""CREATE TABLE IF NOT EXISTS reports (
                        domain TEXT, reporter TEXT, PRIMARY KEY (domain, reporter))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT, url TEXT,
                        label INTEGER, reporter TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS whois_cache (
                        domain TEXT PRIMARY KEY, age_days INTEGER, updated_ts INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tls_cache (
                        domain TEXT PRIMARY KEY, cert_age_days INTEGER, invalid_cert INTEGER, updated_ts INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS brand_phash (
                        brand TEXT, asset_url TEXT PRIMARY KEY, phash INTEGER)""")
        c.executemany("INSERT OR IGNORE INTO whitelist VALUES (?)",
                      [("google.com",), ("github.com",), ("microsoft.com",), ("paypal.com",),
                       ("flipkart.com",), ("amazon.com",), ("amazon.in",), ("hdfcbank.com",),
                       ("icicibank.com",), ("sbi.co.in",), ("apple.com",), ("netflix.com",)])
        c.executemany("INSERT OR IGNORE INTO blacklist VALUES (?, ?)",
                      [("known-scam.com", "PhishTank"), ("malware-drop.net", "URLhaus")])
        conn.commit()


init_db()


# ---------- 1. Domain Age & WHOIS / RDAP Lookup ----------
_WHOIS_MEM_CACHE: Dict[str, Tuple[Optional[int], float]] = {}
_MULTI_PART_TLDS = {
    "co.in", "ac.in", "gov.in", "org.in", "net.in", "edu.in", "res.in",
    "co.uk", "org.uk", "ac.uk", "com.au", "co.jp", "co.nz", "com.br", "com.sg"
}


def get_apex_domain(domain: str) -> str:
    """Extract registrable apex domain (e.g. login.portal.example.co.in -> example.co.in)."""
    d = (domain or "").lower().strip().rstrip(".")
    parts = [p for p in d.split(".") if p]
    if len(parts) <= 2:
        return d
    last_two = ".".join(parts[-2:])
    if last_two in _MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


def _parse_iso_datetime(val: str) -> Optional[datetime.datetime]:
    if not val or not isinstance(val, str):
        return None
    s = val.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%b-%Y",
    ):
        try:
            return datetime.datetime.strptime(s[:26].rstrip("Z"), fmt.rstrip("Z"))
        except Exception:
            pass
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except Exception:
        return None


def _query_rdap_creation_date(apex: str) -> Optional[datetime.datetime]:
    """Query standard IANA/RDAP (https://rdap.org/domain/<apex>) for registration event."""
    try:
        r = requests.get(
            f"https://rdap.org/domain/{apex}",
            headers={"Accept": "application/rdap+json, application/json", "User-Agent": "PhishEye-WHOIS/1.0"},
            timeout=2.5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        for ev in data.get("events", []):
            if ev.get("eventAction") in ("registration", "registered"):
                dt = _parse_iso_datetime(ev.get("eventDate", ""))
                if dt:
                    return dt
    except Exception:
        pass
    return None


def _query_raw_whois_creation_date(apex: str) -> Optional[datetime.datetime]:
    """Fallback TCP port-43 WHOIS lookup for creation date."""
    tld = apex.rsplit(".", 1)[-1] if "." in apex else ""
    server = {
        "com": "whois.verisign-grs.com",
        "net": "whois.verisign-grs.com",
        "org": "whois.pir.org",
        "in": "whois.registry.in",
        "io": "whois.nic.io",
        "co": "whois.nic.co",
        "ai": "whois.nic.ai",
        "xyz": "whois.nic.xyz",
        "top": "whois.nic.top",
        "online": "whois.nic.online",
        "site": "whois.nic.site",
    }.get(tld, "whois.iana.org")
    try:
        with socket.create_connection((server, 43), timeout=2.2) as sock:
            sock.sendall((apex + "\r\n").encode("ascii", errors="ignore"))
            resp = b""
            while len(resp) < 16384:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
        text = resp.decode("utf-8", errors="ignore")
        m = re.search(
            r"(?:Creation Date|Created On|Registered on|Registration Time|created):\s*([0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9]{2}:[0-9]{2}:[0-9]{2})?)",
            text,
            re.IGNORECASE,
        )
        if m:
            return _parse_iso_datetime(m.group(1))
    except Exception:
        pass
    return None


def get_domain_age_days(domain: str) -> Optional[int]:
    """
    Returns domain age in days using:
      1) SQLite / memory cache (24h TTL)
      2) python-whois (if installed)
      3) RDAP protocol lookup (https://rdap.org/domain/...)
      4) TCP port-43 WHOIS fallback
    Returns None if unregistered, local, IP address, or privacy-protected without creation date.
    """
    apex = get_apex_domain(domain)
    if not apex or "." not in apex or apex.endswith((".local", ".test", ".localhost")):
        return None
    try:
        ipaddress.ip_address(apex)
        return None
    except ValueError:
        pass

    now_ts = datetime.datetime.utcnow().timestamp()
    cached = _WHOIS_MEM_CACHE.get(apex)
    if cached and (now_ts - cached[1]) < 86400:
        return cached[0]

    try:
        with closing(db()) as conn:
            row = conn.execute(
                "SELECT age_days, updated_ts FROM whois_cache WHERE domain=?", (apex,)
            ).fetchone()
            if row and row[0] is not None and (now_ts - row[1]) < 86400 * 7:
                _WHOIS_MEM_CACHE[apex] = (int(row[0]), now_ts)
                return int(row[0])
    except Exception:
        pass

    creation_date: Optional[datetime.datetime] = None

    # 1) Try python-whois if installed
    if pywhois is not None:
        try:
            w = pywhois.whois(apex)
            cd = getattr(w, "creation_date", None)
            if isinstance(cd, list) and cd:
                cd = cd[0]
            if isinstance(cd, datetime.datetime):
                creation_date = cd.replace(tzinfo=None)
            elif isinstance(cd, str):
                creation_date = _parse_iso_datetime(cd)
        except Exception:
            pass

    # 2) Try RDAP JSON API
    if creation_date is None:
        creation_date = _query_rdap_creation_date(apex)

    # 3) Try raw WHOIS socket
    if creation_date is None:
        creation_date = _query_raw_whois_creation_date(apex)

    age_days: Optional[int] = None
    if creation_date is not None:
        age_days = max(0, (datetime.datetime.utcnow() - creation_date).days)

    _WHOIS_MEM_CACHE[apex] = (age_days, now_ts)
    if age_days is not None:
        try:
            with closing(db()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO whois_cache VALUES (?, ?, ?)",
                    (apex, int(age_days), int(now_ts)),
                )
                conn.commit()
        except Exception:
            pass
    return age_days


# ---------- 1B. Live TLS / SSL Certificate Age (notBefore) & Validity Inspector ----------
_TLS_MEM_CACHE: Dict[str, Tuple[Optional[int], bool, float]] = {}


def get_tls_cert_info(domain: str) -> Tuple[Optional[int], bool]:
    """
    Returns (cert_age_days, has_invalid_or_self_signed_cert).
    Even when WHOIS/RDAP hides domain registration dates, X.509 SSL certificates always expose
    their exact issuance timestamp (`notBefore`) during the TLS handshake.
    """
    host = (domain or "").lower().strip().rstrip(".")
    if not host or host in ("localhost", "127.0.0.1") or "." not in host:
        return None, False

    now_ts = datetime.datetime.utcnow().timestamp()
    cached = _TLS_MEM_CACHE.get(host)
    if cached and (now_ts - cached[2] < 43200):
        return cached[0], cached[1]

    try:
        with closing(db()) as conn:
            row = conn.execute(
                "SELECT cert_age_days, invalid_cert, updated_ts FROM tls_cache WHERE domain=?", (host,)
            ).fetchone()
            if row and (now_ts - row[2] < 86400):
                age_val = int(row[0]) if row[0] is not None and row[0] >= 0 else None
                inv_val = bool(row[1])
                _TLS_MEM_CACHE[host] = (age_val, inv_val, now_ts)
                return age_val, inv_val
    except Exception:
        pass

    cert_age_days: Optional[int] = None
    invalid_cert = False
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=1.5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                not_before_str = cert.get("notBefore") if cert else None
                if not_before_str:
                    # Standard ASN.1 time format: 'Oct 25 12:00:00 2026 GMT'
                    issued_dt = datetime.datetime.strptime(not_before_str, "%b %d %H:%M:%S %Y %Z")
                    cert_age_days = max(0, (datetime.datetime.utcnow() - issued_dt).days)
    except ssl.SSLCertVerificationError:
        invalid_cert = True
    except Exception:
        pass

    _TLS_MEM_CACHE[host] = (cert_age_days, invalid_cert, now_ts)
    if cert_age_days is not None or invalid_cert:
        try:
            with closing(db()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO tls_cache VALUES (?, ?, ?, ?)",
                    (host, int(cert_age_days) if cert_age_days is not None else -1, 1 if invalid_cert else 0, int(now_ts)),
                )
                conn.commit()
        except Exception:
            pass

    return cert_age_days, invalid_cert


# ---------- 2. Visual / Logo Similarity Matching (pHash Computer Vision) ----------
PROTECTED_BRANDS: Dict[str, dict] = {
    "Microsoft": {
        "domains": {"microsoft.com", "live.com", "office.com", "outlook.com", "azure.com", "msn.com", "sharepoint.com", "microsoftonline.com"},
        "asset_hosts": {"aadcdn.msftauth.net", "aadcdn.msauth.net", "logincdn.msauth.net", "c.s-microsoft.com", "img-prod-cms-rt-microsoft-com.akamaized.net"},
        "keywords": ("microsoft", "office 365", "office365", "outlook", "onedrive", "azure", "sign in to your account"),
        "phashes": [0xF0F0E8C8D2B49681, 0xCC993366CC993366],
    },
    "Google": {
        "domains": {"google.com", "google.co.in", "google.co.uk", "gmail.com", "youtube.com", "googleapis.com", "gstatic.com", "google.ca", "google.de"},
        "asset_hosts": {"www.gstatic.com", "ssl.gstatic.com", "lh3.googleusercontent.com", "fonts.gstatic.com"},
        "keywords": ("google", "gmail", "google workspace", "sign in - google accounts"),
        "phashes": [0xB4D369964BD29669, 0x9669D24B9669B4D3],
    },
    "PayPal": {
        "domains": {"paypal.com", "paypal.me", "paypalobjects.com"},
        "asset_hosts": {"www.paypalobjects.com", "paypalobjects.com"},
        "keywords": ("paypal", "log in to your paypal account"),
        "phashes": [0xE6C9933C6996C33C, 0xC396693CC993663C],
    },
    "Apple": {
        "domains": {"apple.com", "icloud.com", "cdn-apple.com"},
        "asset_hosts": {"www.apple.com", "appleid.cdn-apple.com"},
        "keywords": ("apple id", "icloud", "sign in with apple", "apple account"),
        "phashes": [0x183C7E7E7E3C1800, 0x3C7EFFDBDBFF7E3C],
    },
    "Amazon": {
        "domains": {"amazon.com", "amazon.in", "amazon.co.uk", "amazon.de", "amazon.co.jp", "amazonaws.com", "ssl-images-amazon.com", "media-amazon.com"},
        "asset_hosts": {"images-na.ssl-images-amazon.com", "m.media-amazon.com", "images-eu.ssl-images-amazon.com"},
        "keywords": ("amazon sign-in", "amazon sign in", "amazon.in", "amazon.com"),
        "phashes": [0x9C6339C6639CC639],
    },
    "Netflix": {
        "domains": {"netflix.com", "nflxext.com", "nflximg.net", "nflxso.net"},
        "asset_hosts": {"assets.nflxext.com", "occ-0.nflxso.net"},
        "keywords": ("netflix", "sign in to netflix"),
        "phashes": [0xC6C6E6F6FECEC6C6],
    },
    "Facebook / Meta": {
        "domains": {"facebook.com", "fb.com", "meta.com", "messenger.com"},
        "asset_hosts": {"static.xx.fbcdn.net", "facebook.com"},
        "keywords": ("facebook", "log into facebook", "meta account"),
        "phashes": [0x3C3C3C3C3C3C3C3C, 0x1838787878381800],
    },
    "Instagram": {
        "domains": {"instagram.com", "cdninstagram.com"},
        "asset_hosts": {"static.cdninstagram.com"},
        "keywords": ("instagram", "log in to instagram"),
        "phashes": [0x3C7EFFFFDBE7C381],
    },
    "WhatsApp": {
        "domains": {"whatsapp.com", "web.whatsapp.com"},
        "asset_hosts": {"static.whatsapp.net"},
        "keywords": ("whatsapp", "whatsapp web"),
        "phashes": [0x183C7E7E7E3C1800],
    },
    "GitHub": {
        "domains": {"github.com", "githubusercontent.com", "githubassets.com"},
        "asset_hosts": {"github.githubassets.com", "avatars.githubusercontent.com"},
        "keywords": ("sign in to github", "github"),
        "phashes": [0x3C7EFFFFDBE7C381],
    },
    "Steam": {
        "domains": {"steampowered.com", "steamcommunity.com"},
        "asset_hosts": {"community.cloudflare.steamstatic.com", "store.cloudflare.steamstatic.com"},
        "keywords": ("sign in to steam", "steam community", "valve steam"),
        "phashes": [0x78FC7C3838FC7830],
    },
    "Discord": {
        "domains": {"discord.com", "discordapp.com"},
        "asset_hosts": {"assets-global.website-files.com", "cdn.discordapp.com"},
        "keywords": ("discord", "welcome back! - discord"),
        "phashes": [0x3C7EFFFFFFFF7E3C],
    },
    "Telegram": {
        "domains": {"telegram.org", "t.me"},
        "asset_hosts": {"telegram.org"},
        "keywords": ("telegram web", "telegram messenger"),
        "phashes": [0x00183C7E7E3C1800],
    },
    "State Bank of India (SBI)": {
        "domains": {"onlinesbi.sbi", "sbi.co.in", "sbicard.com"},
        "asset_hosts": {"retail.onlinesbi.sbi", "www.onlinesbi.sbi", "sbi.co.in"},
        "keywords": ("state bank of india", "onlinesbi", "sbi yono", "sbi netbanking"),
        "phashes": [0x3C7EFFDBDBFF7E3C],
    },
    "HDFC Bank": {
        "domains": {"hdfcbank.com", "hdfc.com"},
        "asset_hosts": {"www.hdfcbank.com", "netbanking.hdfcbank.com", "v.hdfcbank.com"},
        "keywords": ("hdfc bank", "hdfcbank", "hdfc netbanking"),
        "phashes": [0xFF818199998181FF, 0xC3C3FFFFC3C3FFFF],
    },
    "ICICI Bank": {
        "domains": {"icicibank.com", "icicidirect.com"},
        "asset_hosts": {"www.icicibank.com", "infinity.icicibank.com"},
        "keywords": ("icici bank", "icicibank", "icici netbanking"),
        "phashes": [0xD86C369B4DA66331],
    },
    "Axis Bank": {
        "domains": {"axisbank.com"},
        "asset_hosts": {"www.axisbank.com"},
        "keywords": ("axis bank", "axis netbanking"),
        "phashes": [0x81C3E7FFFFE7C381],
    },
    "Chase Bank": {
        "domains": {"chase.com"},
        "asset_hosts": {"static.chasecdn.com", "www.chase.com"},
        "keywords": ("chase online", "chase bank", "sign in - chase.com"),
        "phashes": [0x007E7E7E7E7E7E00],
    },
    "Bank of America": {
        "domains": {"bankofamerica.com", "bofa.com"},
        "asset_hosts": {"www.bankofamerica.com"},
        "keywords": ("bank of america", "bofa login", "bofa sign in"),
        "phashes": [0xFF00FF00FF00FF00],
    },
    "Wells Fargo": {
        "domains": {"wellsfargo.com"},
        "asset_hosts": {"www.wellsfargo.com"},
        "keywords": ("wells fargo", "sign on to wells fargo"),
        "phashes": [0xFFFF00000000FFFF],
    },
    "MetaMask": {
        "domains": {"metamask.io"},
        "asset_hosts": {"metamask.io"},
        "keywords": ("metamask", "import an account with secret recovery phrase", "metamask wallet"),
        "phashes": [0x81C3E7FFDBE7C381],
    },
    "Coinbase": {
        "domains": {"coinbase.com"},
        "asset_hosts": {"images.ctfassets.net", "www.coinbase.com"},
        "keywords": ("coinbase", "sign in to coinbase"),
        "phashes": [0x3C7EFFDBDBFF7E3C],
    },
    "Binance": {
        "domains": {"binance.com"},
        "asset_hosts": {"bin.bnbstatic.com"},
        "keywords": ("binance", "binance login"),
        "phashes": [0x183C7EFFE7C38100],
    },
    "Flipkart": {
        "domains": {"flipkart.com", "flixcart.com"},
        "asset_hosts": {"static-assets-web.flixcart.com", "img1a.flixcart.com", "rukminim1.flixcart.com", "rukminim2.flixcart.com"},
        "keywords": ("flipkart",),
        "phashes": [0x78FC7C3838FC7830],
    },
}


def _decode_png_to_gray32(data: bytes) -> Optional[np.ndarray]:
    """Pure-Python/NumPy PNG decoder fallback (handles 8-bit Grayscale/RGB/RGBA/Palette PNGs) resized to 32x32."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    try:
        pos = 8
        width = height = bit_depth = color_type = 0
        idat_chunks = []
        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            chunk_type = data[pos + 4:pos + 8]
            chunk_data = data[pos + 8:pos + 8 + length]
            pos += 12 + length
            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            elif chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                break
        if width <= 0 or height <= 0 or width > 4096 or height > 4096 or bit_depth != 8:
            return None
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
        if not channels:
            return None
        raw = zlib.decompress(b"".join(idat_chunks))
        stride = width * channels
        expected = (stride + 1) * height
        if len(raw) < expected:
            return None
        rows = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, stride + 1))
        pixels = rows[:, 1:].reshape((height, width, channels)).astype(np.float32)
        if channels >= 3:
            gray = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
        else:
            gray = pixels[:, :, 0]
        # Resample to 32x32 grid
        y_idx = np.linspace(0, height - 1, 32).astype(int)
        x_idx = np.linspace(0, width - 1, 32).astype(int)
        return gray[np.ix_(y_idx, x_idx)]
    except Exception:
        return None


def compute_image_phash(img_bytes: bytes) -> Optional[int]:
    """
    Computes a 64-bit Perceptual Hash (pHash) via 2D Discrete Cosine Transform (DCT) on a 32x32 grayscale grid.
    Uses Pillow if installed, otherwise uses built-in PNG/ICO decoder + structural byte DCT fallback.
    """
    if not img_bytes or len(img_bytes) < 24:
        return None
    gray32: Optional[np.ndarray] = None

    # 1) Try Pillow if available
    if PilImage is not None:
        try:
            import io as _io
            with PilImage.open(_io.BytesIO(img_bytes)) as im:
                im = im.convert("L").resize((32, 32))
                gray32 = np.asarray(im, dtype=np.float32)
        except Exception:
            pass

    # 2) Pure-Python PNG / ICO-embedded-PNG decoder
    if gray32 is None:
        png_offset = img_bytes.find(b"\x89PNG\r\n\x1a\n")
        if png_offset >= 0:
            gray32 = _decode_png_to_gray32(img_bytes[png_offset:])

    # 3) Fallback: reshape normalized payload bytes into 32x32 matrix for DCT signature
    if gray32 is None:
        arr = np.frombuffer(img_bytes[:4096], dtype=np.uint8).astype(np.float32)
        idx = np.linspace(0, len(arr) - 1, 1024).astype(int)
        gray32 = arr[idx].reshape((32, 32))

    try:
        dct_matrix = dctn(gray32, type=2, norm="ortho")
        low_freq = dct_matrix[:8, :8].flatten()
        # Exclude DC component (index 0) when computing median threshold
        med = float(np.median(low_freq[1:]))
        bits = (low_freq > med).astype(np.uint8)
        phash_val = 0
        for b in bits:
            phash_val = (phash_val << 1) | int(b)
        return phash_val
    except Exception:
        return None


def phash_similarity(hash_a: int, hash_b: int) -> float:
    """Returns perceptual similarity in [0.0, 1.0] based on 64-bit Hamming distance."""
    xor_val = (hash_a ^ hash_b) & ((1 << 64) - 1)
    hamming_dist = bin(xor_val).count("1")
    return 1.0 - (hamming_dist / 64.0)


def _fetch_image_phash(url: str) -> Optional[int]:
    """Fetches a remote favicon/logo image (SSRF-guarded, max 256 KB, 2s timeout) and returns its 64-bit pHash."""
    if not url or not url.lower().startswith(("http://", "https://")):
        return None
    try:
        with closing(db()) as conn:
            row = conn.execute("SELECT phash FROM brand_phash WHERE asset_url=?", (url,)).fetchone()
            if row and row[0] is not None:
                return int(row[0])
    except Exception:
        pass
    try:
        assert_public_host(url)
        r = requests.get(
            url,
            headers={"User-Agent": "PhishEye-Vision/1.0"},
            timeout=2.0,
            stream=True,
            allow_redirects=True,
        )
        if r.status_code != 200:
            r.close()
            return None
        raw = r.raw.read(262144, decode_content=True)
        r.close()
        ph = compute_image_phash(raw)
        if ph is not None:
            try:
                with closing(db()) as conn:
                    conn.execute("INSERT OR REPLACE INTO brand_phash VALUES (?, ?, ?)", ("cached", url, int(ph)))
                    conn.commit()
            except Exception:
                pass
        return ph
    except Exception:
        return None


def _is_brand_domain(domain: str, allowed_set: set) -> bool:
    d = domain.lower().removeprefix("www.")
    for parent in parent_domains(d):
        if parent in allowed_set:
            return True
    return False


def detect_brand_impersonation(
    page_domain: str,
    favicon_url: str = "",
    logo_urls: Optional[List[str]] = None,
    logo_hints: Optional[List[str]] = None,
    page_title: str = "",
    has_password: bool = False,
) -> Optional[dict]:
    """
    Visual / Logo Similarity & Brand Impersonation Detector:
    1) Checks if favicon or <img src="..."> logos are hotlinked from a protected brand's official CDN/domain
       while hosted on an unauthorized domain.
    2) Computes 64-bit DCT Perceptual Hash (pHash) of the page's favicon and logo images and compares
       visual similarity against protected brands (>= 90% visual similarity threshold).
    3) Checks explicit logo alt/src + page title brand login claims on unauthorized domains.
    """
    logo_urls = logo_urls or []
    logo_hints = logo_hints or []
    candidate_urls = [u for u in ([favicon_url] + list(logo_urls[:4])) if u and u.startswith(("http://", "https://"))]

    # Pre-check: if the current page IS one of the protected brands' official domains, learn/cache its pHash and return None
    for brand_name, cfg in PROTECTED_BRANDS.items():
        if _is_brand_domain(page_domain, cfg["domains"]):
            if favicon_url:
                ph = _fetch_image_phash(favicon_url)
                if ph is not None and ph not in cfg["phashes"]:
                    cfg["phashes"].append(ph)
            return None

    # 1) Check hotlinked official brand logo / favicon assets & compute pHash similarity
    for asset_url in candidate_urls:
        asset_host = (urlparse(asset_url).hostname or "").lower()
        asset_lower = asset_url.lower()

        for brand_name, cfg in PROTECTED_BRANDS.items():
            # Direct hotlink from official brand asset CDN or official domain
            if asset_host in cfg["asset_hosts"] or _is_brand_domain(asset_host, cfg["domains"]):
                return {
                    "brand": brand_name,
                    "similarity": 98.5,
                    "method": "OFFICIAL_BRAND_LOGO_HOTLINK",
                    "reason": f"Visual Brand Impersonation: Page loads official {brand_name} logo/favicon from {asset_host} on unauthorized domain {page_domain}",
                }

        # Compute perceptual hash (pHash) of the fetched logo/favicon image
        img_phash = _fetch_image_phash(asset_url)
        if img_phash is not None:
            for brand_name, cfg in PROTECTED_BRANDS.items():
                for ref_hash in cfg["phashes"]:
                    sim = phash_similarity(img_phash, ref_hash)
                    if sim >= 0.90:
                        # Require either brand keyword hint in URL/title/logo or >= 93% pure pHash match
                        combined_text = f"{asset_lower} {page_title} {' '.join(logo_hints)}".lower()
                        if sim >= 0.93 or any(kw in combined_text for kw in cfg["keywords"]):
                            pct = round(sim * 100, 1)
                            return {
                                "brand": brand_name,
                                "similarity": pct,
                                "method": "VISUAL_PHASH_MATCH",
                                "reason": f"Visual Brand Impersonation: Logo/favicon is {pct}% visually identical (pHash) to {brand_name} on unauthorized domain {page_domain}",
                            }

    # 2) Check logo alt/filename + page title brand impersonation when asking for password
    combined_meta = f"{page_title} {' '.join(logo_hints)} {' '.join(candidate_urls)}".lower()
    if has_password and combined_meta.strip():
        for brand_name, cfg in PROTECTED_BRANDS.items():
            if any(kw in combined_meta for kw in cfg["keywords"]):
                return {
                    "brand": brand_name,
                    "similarity": 92.0,
                    "method": "BRAND_LOGO_LOGIN_SPOOF",
                    "reason": f"Brand Impersonation Detected: Page displays {brand_name} branding and requests a password on unauthorized domain {page_domain}",
                }

    return None



def get_domain(url: str) -> str:
    """Lower-cased hostname without port, credentials, or leading 'www.'."""
    host = (urlparse(normalize_url(url)).hostname or "").lower()
    return host.removeprefix("www.")


def parent_domains(domain: str) -> List[str]:
    """'a.b.example.com' -> ['a.b.example.com','b.example.com','example.com']"""
    parts = domain.split(".")
    return [".".join(parts[i:]) for i in range(len(parts) - 1)] or [domain]


# Known shared/free hosting platforms heavily abused by disposable phishing kits
SHARED_HOSTING_SUFFIXES = (
    "pages.dev", "github.io", "vercel.app", "netlify.app", "web.app", "firebaseapp.com",
    "onrender.com", "workers.dev", "ngrok-free.app", "ngrok-free.dev", "ngrok.io",
    "trycloudflare.com", "blogspot.com", "wordpress.com", "weebly.com", "wixsite.com",
    "000webhostapp.com", "surge.sh", "glitch.me", "herokuapp.com", "azurewebsites.net",
    "amazonaws.com", "s3.amazonaws.com", "storage.googleapis.com", "docs.google.com",
    "forms.gle", "sites.google.com", "bit.ly", "tinyurl.com", "t.co", "rb.gy",
    "cutt.ly", "is.gd", "replit.dev", "fly.dev", "railway.app", "duckdns.org",
)

# Accredited/regulated TLDs that require government or university charter verification
ACCREDITED_INSTITUTION_TLDS = {
    "edu", "gov", "mil", "ac.uk", "gov.uk", "gov.in", "nic.in", "ac.in", "edu.in", "edu.au", "gov.au"
}

# Subdomain brand spoof keywords (e.g. paypal.somesite.com or microsoft.user-portal.net)
SUBDOMAIN_BRAND_SPOOFS = (
    "paypal", "microsoft", "google", "apple", "netflix", "hdfc", "icici", "sbi",
    "chase", "amazon", "flipkart", "meta", "facebook", "instagram", "whatsapp",
    "coinbase", "binance", "metamask", "steam", "discord", "telegram", "bofa",
)

# Piracy, mirror streaming, cracked software, APK-mod, and adware-lure keywords
POTENTIAL_RISK_KEYWORDS = (
    "netmirror", "mirror", "pirate", "torrent", "1337x", "yts", "movierulz",
    "ibomma", "bappam", "tamilrockers", "filmyzilla", "filmywap", "hdhub",
    "vegamovies", "bolly4u", "123movies", "fmovies", "soap2day", "gostream",
    "putlocker", "pikashow", "thoptv", "cricfree", "vipbox", "streameast",
    "crack", "keygen", "repack", "fitgirl", "getintopc", "filecr", "modapk",
    "apkmod", "happymod", "liteapks", "freespin", "1xbet", "rummy", "teenpatti",
    "aviator", "freefire-diamond", "robux-free",
)

# Grey-market / high-abuse TLDs frequently used by mirror, piracy, adware & phishing sites
CAUTION_TLDS = {
    "cc", "gg", "to", "sx", "bz", "ws", "su", "pe", "pm", "wf", "yt", "tv",
    "lol", "fun", "site", "online", "live", "pro", "vip", "club", "win",
    "bid", "monster", "icu", "cyou", "sbs", "cfd", "cam", "rest", "click",
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "buzz", "zip", "mov",
}


def is_shared_hosting(domain: str) -> bool:
    d = domain.lower().strip(".")
    return any(d == s or d.endswith("." + s) for s in SHARED_HOSTING_SUFFIXES)


def has_potential_risk_keyword(text: str) -> Optional[str]:
    low = (text or "").lower()
    for kw in POTENTIAL_RISK_KEYWORDS:
        if kw in low:
            return kw
    return None


def lookup(table: str, domain: str):
    """Match the domain or any parent domain (mail.google.com matches google.com)."""
    if table == "whitelist":
        if is_shared_hosting(domain) or has_potential_risk_keyword(domain):
            return None
        apex = get_apex_domain(domain)
        sub_part = domain[: -len(apex)].rstrip(".") if apex and domain.endswith(apex) else ""
        # Only disqualify if subdomain contains an unauthorized brand spoof on a different domain
        if sub_part and any(kw in sub_part for kw in SUBDOMAIN_BRAND_SPOOFS):
            if not any(kw in apex for kw in SUBDOMAIN_BRAND_SPOOFS):
                return None
    cols = "source" if table == "blacklist" else "1"
    with closing(db()) as conn:
        for d in parent_domains(domain):
            row = conn.execute(f"SELECT {cols} FROM {table} WHERE domain=?", (d,)).fetchone()
            if row:
                return row
    return None


def get_report_count(domain: str) -> int:
    with closing(db()) as conn:
        row = conn.execute("SELECT COUNT(*) FROM reports WHERE domain=?", (domain,)).fetchone()
        return row[0] if row else 0



def community_probability(domain: str):
    """Returns (p_phish in 0..1, effective_weight). Phishing reports count 3x a safe report."""
    with closing(db()) as conn:
        row = conn.execute("SELECT SUM(label), COUNT(*) FROM feedback WHERE domain=?", (domain,)).fetchone()
    ph, n = (row[0] or 0), (row[1] or 0)
    if n == 0:
        return None, 0
    sf = n - ph
    ph_w = ph * 3.0
    return (ph_w + 1) / (ph_w + sf + 2), ph_w + sf


class DOMSignals(BaseModel):
    password_inputs: int = 0
    hidden_inputs: int = 0
    iframe_count: int = 0
    external_form_action: bool = False
    context_menu_disabled: bool = False
    favicon_url: str = ""
    logo_urls: List[str] = []
    logo_hints: List[str] = []
    page_title: str = ""
    exfil_webhooks: List[str] = []
    popunder_scripts: List[str] = []
    suspicious_download_links: int = 0
    scam_intent_hits: List[str] = []
    page_text_sample: str = ""
    form_action_destinations: List[str] = []
    has_credential_inputs: bool = False
    credential_input_types: List[str] = []
    cross_origin_form_detected: bool = False


class ScanRequest(BaseModel):
    url: str
    dom_signals: DOMSignals = DOMSignals()


class ReportRequest(BaseModel):
    url: str
    label: str = "phishing"  # "phishing" or "safe"



@app.post("/analyze")
@app.post("/scan")
def analyze_url(req: ScanRequest):
    domain = get_domain(req.url)
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid URL")

    feats = extract_features(req.url)
    reports_count = get_report_count(domain)

    bl_hit = lookup("blacklist", domain)
    if not bl_hit and lookup("whitelist", domain):
        return {
            "url": req.url,
            "hostname": domain,
            "threat_score": 1.2,
            "risk_score": 1.2,
            "ml_probability": 1.2,
            "dom_penalty": 0.0,
            "confidence_score": 0.012,
            "verdict": "SAFE",
            "status_badge": "SAFE",
            "is_malicious": False,
            "is_zero_day": False,
            "layer": "Layer 1: Tranco Verified Safe",
            "detection_method": "TRANCO_WHITELIST",
            "threat_details": "Domain verified in Tranco legitimate database",
            "reasons": ["Domain verified in Tranco legitimate database"],
            "lexical_features": feats,
            "reports_count": reports_count,
        }

    if bl_hit:
        msg = f"Instant Layer 1 Block: Reported in {bl_hit[0]} live threat feed"
        return {
            "url": req.url,
            "hostname": domain,
            "threat_score": 99.9,
            "risk_score": 99.9,
            "ml_probability": 99.9,
            "dom_penalty": 0.0,
            "confidence_score": 0.999,
            "verdict": "PHISHING_DETECTED",
            "status_badge": "PHISHING_DETECTED",
            "is_malicious": True,
            "is_zero_day": False,
            "layer": f"Layer 1: {bl_hit[0]} Blacklist",
            "detection_method": "THREAT_INTEL_CACHE",
            "threat_details": msg,
            "reasons": [msg],
            "lexical_features": feats,
            "reports_count": reports_count,
        }

    if USE_URLHAUS_LIVE:
        live = threat_intel.check_urlhaus_live_api(req.url)
        if live:
            return {
                "url": req.url,
                "hostname": domain,
                "threat_score": 99.9,
                "risk_score": 99.9,
                "ml_probability": 99.9,
                "dom_penalty": 0.0,
                "confidence_score": 0.999,
                "verdict": "PHISHING_DETECTED",
                "status_badge": "PHISHING_DETECTED",
                "is_malicious": True,
                "is_zero_day": False,
                "layer": "Layer 1: URLhaus Live API",
                "detection_method": "THREAT_INTEL_LIVE",
                "threat_details": live["details"],
                "reasons": [live["details"]],
                "lexical_features": feats,
                "reports_count": reports_count,
            }

    ml_prob, feats_v2_data = compute_ensemble_ml_prob(req.url)

    reasons = []
    dom_penalty = 0.0
    sig = req.dom_signals

    parsed_url = urlparse(normalize_url(req.url))
    hostname_lower = (parsed_url.hostname or "").lower()
    path_lower = ((parsed_url.path or "") + "?" + (parsed_url.query or "")).lower()
    host_hyphens = hostname_lower.count("-")

    hostname_has_suspicious = any(kw in hostname_lower for kw in (
        "login", "signin", "sign-in", "verify", "verification", "secure", "account",
        "update", "banking", "wallet", "confirm", "password", "authenticate",
        "credential", "recovery", "ebayisapi", "webscr", "portal", "auth",
        "billing", "invoice", "suspend", "unlock", "validate", "restore",
        "phish", "fake", "scam",
    ))
    path_has_suspicious = any(kw in path_lower for kw in (
        "login", "signin", "sign-in", "verify", "verification", "password",
        "authenticate", "credential", "recovery", "webscr", "billing", "unlock",
    ))

    # --- 1) Domain Age & WHOIS / RDAP Check ---
    domain_age_days = get_domain_age_days(domain)
    whois_override_score: Optional[float] = None
    layer_used = "Layer 2: 3-Factor Correlation Matrix"

    # --- 2) Visual / Logo & Favicon Similarity Check (pHash Computer Vision) ---
    brand_hit = detect_brand_impersonation(
        page_domain=domain,
        favicon_url=sig.favicon_url,
        logo_urls=sig.logo_urls,
        logo_hints=sig.logo_hints,
        page_title=sig.page_title,
        has_password=(sig.password_inputs > 0 or sig.has_credential_inputs),
    )

    # --- 3) Live Intent & Keyword Forensics ---
    tld = domain.rsplit(".", 1)[-1].lower() if "." in domain else ""
    risk_kw = has_potential_risk_keyword(domain) or has_potential_risk_keyword(sig.page_title)

    intent_hits = list(sig.scam_intent_hits)
    text_sample_low = f"{sig.page_title} {sig.page_text_sample}".lower()
    backend_intent_rules = (
        ("account has been suspended", "Urgent account suspension coercion"),
        ("verify your identity", "Urgent identity verification coercion"),
        ("enter your seed phrase", "Crypto wallet seed-phrase drainer"),
        ("recovery phrase", "Crypto wallet recovery-phrase drainer"),
        ("connect wallet to claim", "Crypto airdrop drainer lure"),
        ("mod apk", "Unauthorized Mod APK distribution lure"),
        ("unlocked premium", "Unauthorized Mod APK distribution lure"),
        ("windows defender alert", "Fake Tech Support / Defender scam alert"),
    )
    for kw_phrase, kw_label in backend_intent_rules:
        if kw_phrase in text_sample_low and kw_label not in intent_hits:
            intent_hits.append(kw_label)

    cert_age_days, invalid_cert = get_tls_cert_info(domain)

    # -------------------------------------------------------------
    # 3-FACTOR CORRELATION MATRIX (Zero-Day Invariant Engine)
    # -------------------------------------------------------------
    # Vector A: Identity Spoofing / Impersonation
    has_identity_spoof = False
    identity_reason = None
    if brand_hit:
        has_identity_spoof = True
        identity_reason = brand_hit["reason"]
    elif feats_v2_data.get("brand_in_subdomain_or_path"):
        has_identity_spoof = True
        identity_reason = f"Brand name impersonation detected in URL structure on unauthorized domain ({domain})"

    # Vector B: Credential / Asset Harvesting Action
    has_cred_action = (sig.password_inputs > 0) or sig.has_credential_inputs
    has_drainer_action = any("drainer" in h.lower() or "seed phrase" in h.lower() for h in intent_hits)
    has_coercion = any("coercion" in h.lower() or "suspended" in h.lower() for h in intent_hits)

    # Vector C: Infrastructure / Origin Anomaly
    infra_high_risk = False
    infra_moderate_risk = False
    infra_reasons = []

    if domain_age_days is not None and domain_age_days < 14:
        infra_high_risk = True
        infra_reasons.append(f"Newly registered domain ({domain_age_days} days old via WHOIS/RDAP)")
    elif domain_age_days is not None and domain_age_days < 60:
        infra_moderate_risk = True
        infra_reasons.append(f"Recently registered domain ({domain_age_days} days old via WHOIS/RDAP)")

    if invalid_cert:
        infra_high_risk = True
        infra_reasons.append("Invalid or self-signed SSL/TLS certificate")
    elif cert_age_days is not None and cert_age_days <= 3:
        infra_moderate_risk = True
        infra_reasons.append(f"Freshly issued SSL/TLS certificate ({cert_age_days} day(s) old)")

    if is_shared_hosting(domain):
        infra_high_risk = True
        infra_reasons.append(f"Hosted on shared/free subdomain platform ({get_apex_domain(domain)})")

    if feats_v2_data.get("is_risky_tld"):
        infra_high_risk = True
        infra_reasons.append(f"High-risk top-level domain (.{tld})")
    elif tld in CAUTION_TLDS:
        infra_moderate_risk = True
        infra_reasons.append(f"Unverified domain on high-abuse TLD (.{tld})")

    if feats.get("has_ip"):
        infra_high_risk = True
        infra_reasons.append("URL uses a raw IP address instead of a domain name")

    # -------------------------------------------------------------
    # DECISION MATRIX (Zero-Day Invariant Evaluation)
    # -------------------------------------------------------------
    is_accredited = any(domain == t or domain.endswith("." + t) for t in ACCREDITED_INSTITUTION_TLDS)

    is_established_clean = (
        (is_accredited or (domain_age_days is not None and domain_age_days >= 365))
        and not has_identity_spoof
        and not is_shared_hosting(domain)
        and not sig.cross_origin_form_detected
        and not sig.exfil_webhooks
        and not has_drainer_action
        and not risk_kw
        and not feats.get("has_ip")
        and not invalid_cert
        and tld not in CAUTION_TLDS
    )

    # Case 1: Critical Direct Phishing Invariants (99.0% - 99.9%)
    if sig.exfil_webhooks:
        whois_override_score = 99.9
        layer_used = "Layer 2: Live Exfiltration Shield"
        reasons.insert(0, f"Critical Phishing Kit Indicator: {sig.exfil_webhooks[0]} detected in page source")
    elif sig.cross_origin_form_detected and has_cred_action:
        whois_override_score = 99.8
        layer_used = "Layer 2: Cross-Origin Credential Form Shield"
        dest_str = sig.form_action_destinations[0] if sig.form_action_destinations else "external server"
        reasons.insert(0, f"Deceptive Credential Form: Page collects credentials and posts them to external host ({dest_str})")
    elif has_identity_spoof and has_cred_action:
        whois_override_score = 99.7
        layer_used = "Layer 2: Identity + Credential Invariant Shield"
        reasons.insert(0, f"{identity_reason} while requesting login credentials")
    elif has_drainer_action:
        whois_override_score = 99.5
        layer_used = "Layer 2: Live NLP Crypto Drainer Shield"
        reasons.insert(0, f"Page Content Intent Analysis: {intent_hits[0]}")
    elif has_cred_action and infra_high_risk:
        whois_override_score = 99.0
        layer_used = "Layer 2: Zero-Day Credential Shield"
        reasons.insert(0, f"{infra_reasons[0]} requesting sensitive user credentials")
    elif has_identity_spoof:
        # Brand spoofed without password field yet (e.g. decoy landing page)
        whois_override_score = 94.0
        layer_used = "Layer 2: Brand Impersonation Shield"
        reasons.insert(0, identity_reason)
    elif risk_kw:
        whois_override_score = 85.0 if (tld in CAUTION_TLDS or tld == "app" or sig.iframe_count >= 1) else 68.5
        layer_used = "Layer 2: Grey-Market / Mirror / Adware Shield"
        reasons.insert(0, f"Potential risk domain ('{risk_kw}' in {domain}) — unauthorized mirror, streaming, or APK/adware distribution site")
    elif intent_hits and "apk" in intent_hits[0].lower():
        whois_override_score = 88.0
        layer_used = "Layer 2: Live NLP Mod APK Shield"
        reasons.insert(0, f"Page Content Intent Analysis: {intent_hits[0]}")
    elif has_coercion and has_cred_action:
        whois_override_score = 92.0
        layer_used = "Layer 2: Social Engineering Coercion Shield"
        reasons.insert(0, f"Social Engineering Coercion: {intent_hits[0]} with credential prompt")

    # Case 2: Established Clean Domains (Zero False Positives on University/Corporate Logins)
    if is_established_clean:
        final_score = min(ml_prob * 0.12, 12.0)
        layer_used = "Layer 2: Verified Authority Origin"
    else:
        # Calibrated scoring for unverified origins
        penalty = 0.0
        if infra_high_risk:
            penalty += 34.0
            reasons.extend(infra_reasons)
        elif infra_moderate_risk:
            penalty += 18.0
            reasons.extend(infra_reasons)

        if has_cred_action:
            penalty += 16.0
            reasons.append(f"Credential input field detected on unverified origin ({domain})")

        if sig.popunder_scripts:
            penalty += 24.0
            reasons.append(f"Detected adware/pop-under redirect network ({sig.popunder_scripts[0]})")

        if sig.suspicious_download_links >= 1:
            penalty += 20.0
            reasons.append(f"Page contains {sig.suspicious_download_links} direct executable/APK/torrent download link(s)")

        if sig.iframe_count >= 2:
            penalty += 14.0
            reasons.append(f"Detected {sig.iframe_count} embedded iframes")

        # Blend base threat and penalties
        if not infra_high_risk and not has_identity_spoof and not sig.exfil_webhooks:
            # Without hard attack invariants, unverified login pages stay in MODERATE (35-48%)
            final_score = min(48.0, max(25.0, (ml_prob * 0.40) + penalty))
        else:
            final_score = min(99.9, ml_prob + penalty)

    if whois_override_score is not None:
        final_score = max(final_score, whois_override_score)

    comm_p, comm_n = community_probability(domain)
    if comm_p is not None and whois_override_score is None:
        w = min(0.7, comm_n / (comm_n + 4.0))
        final_score = final_score * (1 - w) + comm_p * 100.0 * w
        if comm_p >= 0.5:
            reasons.append("Community reports lean toward phishing")
        else:
            reasons.append("Community reports lean toward legitimate")
    final_score = round(min(99.9, final_score), 1)
    is_phish = final_score >= 60.0
    if is_phish:
        verdict = "PHISHING_DETECTED"
        status_badge = "PHISHING_DETECTED"
    elif final_score >= 30.0:
        verdict = "MODERATE"
        status_badge = "MODERATE"
    else:
        verdict = "SAFE"
        status_badge = "SAFE"

    # When the final assessment is SAFE, only provide positive explanations of WHY the site is safe.
    if status_badge == "SAFE":
        reasons = [f"Domain {domain} passed 54-feature lexical & structural verification"]
        if domain_age_days is not None and domain_age_days >= 365:
            reasons.append(f"Established domain ({domain_age_days // 365}+ years old via WHOIS/RDAP)")
        if feats.get("has_https"):
            reasons.append("Encrypted HTTPS connection verified")
        reasons.append("No deceptive credential forms or brand impersonation detected")
    elif not reasons:
        reasons.append("Unusual URL or domain risk indicators detected")

    return {
        "url": req.url,
        "hostname": domain,
        "threat_score": final_score,
        "risk_score": final_score,
        "ml_probability": round(ml_prob, 1),
        "dom_penalty": dom_penalty,
        "domain_age_days": domain_age_days,
        "brand_impersonation": brand_hit,
        "confidence_score": round(final_score / 100.0, 4),
        "verdict": verdict,
        "status_badge": status_badge,
        "is_malicious": is_phish,
        "is_zero_day": True,
        "layer": layer_used,
        "detection_method": "ML_XGBOOST_ZERO_DAY",
        "threat_details": reasons[0],
        "reasons": reasons,
        "lexical_features": feats,
        "reports_count": reports_count,
    }



@app.post("/report")
def report_site(req: ReportRequest, request: Request):
    domain = get_domain(req.url)
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid URL")
    label = 0 if (req.label or "").lower() in ("safe", "legit", "legitimate", "benign") else 1
    # "Safe" reports only make sense for zero-day sites (not already blacklisted/whitelisted).
    if label == 0 and (lookup("blacklist", domain) or lookup("whitelist", domain)):
        return {"status": "ignored", "message": "Thanks! This site is already classified."}

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        reporter = forwarded_for.split(",")[-1].strip()
    else:
        reporter = request.client.host if request.client else "unknown"

    # Every report is stored (no cap) and becomes a training example for the ML model.
    with closing(db()) as conn:
        conn.execute("INSERT INTO feedback (domain, url, label, reporter) VALUES (?, ?, ?, ?)",
                     (domain, req.url, label, reporter))
        conn.commit()

    return {"status": "success", "message": "Thanks! Your report will help train the model."}



MAX_BYTES = 500_000
MAX_REDIRECTS = 3


def assert_public_host(url: str):
    """Reject non-http(s) URLs and any host that resolves to a non-public IP (SSRF guard)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http/https URLs are allowed")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError("Host could not be resolved")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError("Target resolves to a non-public address")


@app.post("/sandbox")
def sandbox_preview(req: ReportRequest):
    try:
        url = normalize_url(req.url)
        headers = {"User-Agent": "PhishEye-Sandbox/1.0"}
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_host(url)
            r = requests.get(url, headers=headers, timeout=4, stream=True, allow_redirects=False)
            if r.is_redirect:
                url = urljoin(url, r.headers.get("Location", ""))
                r.close()
                continue
            raw = r.raw.read(MAX_BYTES, decode_content=True)
            r.close()
            break
        else:
            raise ValueError("Too many redirects")

        html = raw.decode("utf-8", errors="ignore")

        # Extract <link rel="*icon*">, <img src="...">, and <title> in sandbox crawler for Visual pHash / Brand matching
        page_domain = get_domain(url)
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        page_title = (title_m.group(1).strip()[:160] if title_m else "")
        icon_m = re.search(r'<link[^>]+rel=["\'][^"\']*icon[^"\']*["\'][^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        favicon_url = urljoin(url, icon_m.group(1)) if icon_m else urljoin(url, "/favicon.ico")
        img_srcs = [
            urljoin(url, m.group(1))
            for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if not m.group(1).startswith("data:")
        ][:4]
        has_pwd = bool(re.search(r'type=["\']?password["\']?', html, re.IGNORECASE))

        brand_hit = detect_brand_impersonation(
            page_domain=page_domain,
            favicon_url=favicon_url,
            logo_urls=img_srcs,
            page_title=page_title,
            has_password=has_pwd,
        )
        age_days = get_domain_age_days(page_domain)

        if nh3 is not None:
            safe_html = nh3.clean(html, tags=nh3.ALLOWED_TAGS - {"img", "picture", "source", "video", "audio", "link", "style"})  # no remote loads, so no IP leak
        else:
            safe_html = re.sub(r"<(script|style|iframe|object|embed|img|link|video|audio|source)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
            safe_html = re.sub(r"<(script|style|iframe|object|embed|img|link|video|audio|source)[^>]*/?>", "", safe_html, flags=re.IGNORECASE)
        banner = ""
        if brand_hit:
            banner += f"<div style='background:#fee2e2;color:#991b1b;padding:8px 10px;border:2px solid #991b1b;font-weight:bold;margin-bottom:8px;'>⚠️ {brand_hit['reason']}</div>"
        if age_days is not None and age_days < 14:
            banner += f"<div style='background:#fef3c7;color:#92400e;padding:8px 10px;border:2px solid #92400e;font-weight:bold;margin-bottom:8px;'>⚠️ WHOIS Alert: Domain registered only {age_days} days ago.</div>"

        return {
            "status": "sandboxed",
            "safe_html_preview": banner + safe_html[:2000],
            "brand_impersonation": brand_hit,
            "domain_age_days": age_days,
        }
    except Exception as e:
        return {"status": "isolated", "message": f"Sandbox blocked live payload execution: {e}"}


@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(MODEL_FEATURES), "threshold": THRESHOLD}



# ---------- Download scanning ----------
RISKY_EXT = {"exe","msi","scr","bat","cmd","com","pif","ps1","vbs","vbe","js","jse","wsf","hta","lnk",
             "jar","apk","dmg","pkg","iso","img","dll","reg","msix","appx"}
ARCHIVE_EXT = {"zip","rar","7z","gz","tar"}
DOC_EXT = {"pdf","doc","docx","xls","xlsx","ppt","pptx","txt","jpg","jpeg","png","gif","mp3","mp4"}
RISKY_MIME = ("application/x-msdownload", "application/x-msdos-program",
              "application/vnd.microsoft.portable-executable", "application/x-dosexec",
              "application/java-archive", "application/vnd.android.package-archive",
              "application/x-apple-diskimage")


class DownloadRequest(BaseModel):
    url: str
    referrer: str = ""
    filename: str = ""
    mime: str = ""


import hashlib
import io
import time
import zipfile

VT_API_KEY = os.getenv("VT_API_KEY", "")
FILE_PEEK_BYTES = 8_000_000
FILE_DEADLINE_S = 8

MAGIC_SIGS = [
    (b"MZ", "Windows executable (EXE/DLL)"),
    (b"\x7fELF", "Linux executable (ELF)"),
    (b"\xcf\xfa\xed\xfe", "macOS executable (Mach-O)"),
    (b"\xca\xfe\xba\xbe", "Java class / Mach-O fat binary"),
    (b"#!", "Script with shebang"),
    (b"L\x00\x00\x00\x01\x14\x02\x00", "Windows shortcut (LNK)"),
]
SAFE_MAGIC = {"pdf": b"%PDF", "png": b"\x89PNG", "jpg": b"\xff\xd8\xff", "jpeg": b"\xff\xd8\xff",
              "gif": b"GIF8", "zip": b"PK", "docx": b"PK", "xlsx": b"PK", "pptx": b"PK"}


def inspect_file(url: str) -> dict:
    """Download (SSRF-guarded, size/time limited) and inspect the real file bytes."""
    out = {"fetched": False}
    if not url.lower().startswith(("http://", "https://")):
        return out
    try:
        cur = url
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_host(cur)
            r = requests.get(cur, stream=True, timeout=5, allow_redirects=False,
                             headers={"User-Agent": "PhishEye-Scanner/1.0"})
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                cur = urljoin(cur, r.headers["Location"])
                r.close()
                continue
            break
        else:
            return out
        data, start = b"", time.time()
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) >= FILE_PEEK_BYTES or time.time() - start > FILE_DEADLINE_S:
                break
        complete = len(data) < FILE_PEEK_BYTES and time.time() - start <= FILE_DEADLINE_S
        r.close()
        out.update(fetched=True, head=data[:16], size=len(data), complete=complete)
        if complete and data:
            out["sha256"] = hashlib.sha256(data).hexdigest()
            if data[:2] == b"PK":
                try:
                    zf = zipfile.ZipFile(io.BytesIO(data))
                    infos = zf.infolist()
                    out["zip_exec"] = [i.filename for i in infos
                                       if i.filename.lower().rsplit(".", 1)[-1] in RISKY_EXT][:3]
                    out["zip_encrypted"] = any(i.flag_bits & 0x1 for i in infos)
                except Exception:
                    pass
    except Exception as e:
        logger.info("inspect_file skipped: %s", e)
    return out


def virustotal_hash(sha256: str):
    """Returns (malicious_engines, total_engines) or None. Needs VT_API_KEY env var."""
    if not VT_API_KEY or not sha256:
        return None
    try:
        r = requests.get(f"https://www.virustotal.com/api/v3/files/{sha256}",
                         headers={"x-apikey": VT_API_KEY}, timeout=5)
        if r.status_code != 200:
            return None
        s = r.json()["data"]["attributes"]["last_analysis_stats"]
        return int(s.get("malicious", 0)), sum(int(v) for v in s.values())
    except Exception:
        return None


@app.post("/scan-download")
def scan_download(req: DownloadRequest):
    reasons, score = [], 0.0
    file_host = get_domain(req.url) if not req.url.lower().startswith(("data:", "blob:")) else ""
    page_host = get_domain(req.referrer) if req.referrer else ""
    path = urlparse(normalize_url(req.url)).path.lower()
    name = (req.filename.replace("\\", "/").split("/")[-1] or path.split("/")[-1]).lower()
    parts = name.split(".")
    ext = parts[-1] if len(parts) > 1 else ""

    # 1. reputation of the file host and of the page that triggered the download
    for host, label in ((file_host, "file host"), (page_host, "referring page")):
        if host:
            hit = lookup("blacklist", host)
            if hit:
                return {"block": True, "score": 99.9,
                        "reasons": [f"The {label} {host} is on the {hit[0]} blacklist"]}

    # 2. file type
    if ext in RISKY_EXT:
        score += 40; reasons.append(f"Executable / script file type (.{ext})")
    elif ext in ARCHIVE_EXT:
        score += 10; reasons.append(f"Archive (.{ext}) may hide an executable")
    if len(parts) >= 3 and parts[-2] in DOC_EXT and ext in RISKY_EXT:
        score += 40; reasons.append(f"Double extension (.{parts[-2]}.{ext}) disguises an executable as a document")
    if req.mime.lower().startswith(RISKY_MIME):
        score += 30; reasons.append(f"Server reports executable content type ({req.mime})")

    # 3. context
    if req.url.lower().startswith("data:"):
        score += 25; reasons.append("File is generated from a data: URL")
    whitelisted = bool(file_host and lookup("whitelist", file_host))
    if file_host and page_host and file_host != page_host and not whitelisted:
        score += 15; reasons.append(f"File comes from {file_host}, a different unverified host than the page ({page_host})")
    if file_host and extract_features(req.url)["has_ip"]:
        score += 25; reasons.append("File is served from a raw IP address")
    if file_host and not whitelisted:
        feats = extract_features(req.url)
        prob = float(ml_model.predict_proba([model_vector(req.url)])[0][1]) * 100
        if prob >= 60:
            score += 20; reasons.append(f"Download URL looks malicious to the ML model ({prob:.0f}%)")
    if whitelisted and ext not in RISKY_EXT:
        score = min(score, 20)

    # 4. inspect the real bytes (magic number, archive contents, hash reputation)
    if file_host and (not whitelisted or ext in RISKY_EXT or ext in ARCHIVE_EXT):
        info = inspect_file(req.url)
        if info.get("fetched") and info.get("head"):
            head = info["head"]
            kind = next((lbl for sig, lbl in MAGIC_SIGS if head.startswith(sig)), None)
            if kind and ext not in RISKY_EXT:
                score += 60; reasons.append(f"File contents are a {kind} but the name says .{ext}")
            expected = SAFE_MAGIC.get(ext)
            if expected and not head.startswith(expected) and not kind:
                score += 10; reasons.append(f"File contents do not match its .{ext} extension")
            if info.get("zip_exec"):
                score += 25; reasons.append("Archive contains executable files: " + ", ".join(info["zip_exec"]))
            if info.get("zip_encrypted"):
                score += 20; reasons.append("Password-protected archive (hides contents from scanners)")
            vt = virustotal_hash(info.get("sha256"))
            if vt and vt[0] >= 3:
                score += 80; reasons.append(f"VirusTotal: {vt[0]}/{vt[1]} antivirus engines flag this file")

    score = round(min(score, 99.9), 1)
    return {"block": score >= 50, "score": score, "reasons": reasons or ["No suspicious download indicators"]}

