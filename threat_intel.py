

import os, logging, requests
from typing import Dict, Any, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger("threat_intel")

URLHAUS_API = "https://urlhaus-api.abuse.ch/v1/url/"
TIMEOUT     = 2.5   


class ThreatIntelEngine:

    def __init__(self, auth_key: Optional[str] = None):
        self.auth_key = auth_key or os.getenv("URLHAUS_AUTH_KEY", "").strip()
        self.cached_malicious_urls: Set[str]    = set()
        self.cached_malicious_domains: Set[str] = set()
        self.source_metadata: Dict[str, str]    = {}

    
    def load_feeds(self, url_feed_dict: Dict[str, str]):
        """Bulk-load {url: source_label} into O(1) lookup sets."""
        for u, src in url_feed_dict.items():
            norm = u.strip().rstrip("/")
            self.cached_malicious_urls.add(norm)
            self.source_metadata[norm] = src
            try:
                host = urlparse(u).netloc
                if host:
                    self.cached_malicious_domains.add(host.lower())
            except Exception:
                pass
        logger.info("Loaded %d URLs into threat cache.", len(self.cached_malicious_urls))

    
    def check_local_cache(self, url: str) -> Optional[Dict[str, Any]]:
        clean = url.strip().rstrip("/")
        if clean in self.cached_malicious_urls:
            return dict(
                is_malicious=True,
                source=self.source_metadata.get(clean, "Threat Intel Feed"),
                confidence=1.0, is_zero_day=False,
                details="Direct URL match in active threat blacklist.",
            )
        try:
            domain = urlparse(url).netloc.lower()
            if domain and domain in self.cached_malicious_domains:
                return dict(
                    is_malicious=True,
                    source="Malicious Domain Blacklist",
                    confidence=0.99, is_zero_day=False,
                    details=f"Host '{domain}' in threat feed.",
                )
        except Exception:
            pass
        return None

    
    def check_urlhaus_live_api(self, url: str) -> Optional[Dict[str, Any]]:
        headers = {"User-Agent": "FastAPI-Phishing-Scanner/1.0"}
        if self.auth_key:
            headers["Auth-Key"] = self.auth_key
        try:
            r = requests.post(URLHAUS_API, data={"url": url.strip()},
                              headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data.get("query_status") == "ok":
                    return dict(
                        is_malicious=True,
                        source="URLhaus (Live API)",
                        threat_type=data.get("threat", "malware_download"),
                        status=data.get("url_status", "active"),
                        tags=data.get("tags", []),
                        confidence=1.0, is_zero_day=False,
                        details=f"URLhaus: {data.get('threat')} ({data.get('url_status')})",
                    )
        except Exception as e:
            logger.warning("URLhaus API error: %s", e)
        return None

    
    def evaluate(self, url: str, check_live_api: bool = True) -> Dict[str, Any]:
        
        hit = self.check_local_cache(url)
        if hit:
            return hit
        
        if check_live_api:
            live = self.check_urlhaus_live_api(url)
            if live:
                self.cached_malicious_urls.add(url.strip().rstrip("/"))
                self.source_metadata[url.strip().rstrip("/")] = "URLhaus (Live API)"
                return live
        
        return dict(
            is_malicious=False, is_zero_day=True,
            source=None, confidence=0.0,
            details="Not in PhishTank / URLhaus. Routing to XGBoost Zero-Day classifier.",
        )



threat_intel = ThreatIntelEngine()
