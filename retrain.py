"""
Retrain and upgrade PhishEye ML models:
1) Updates the 54-feature XGBoost model (model_v2.json) with:
   - User reports from threat_intel.db (feedback table, weighted 15x)
   - Hard-negative legitimate deep-path URLs (cancels path-length false-positive bias)
   - Mirror / Piracy / APK-Mod / Crypto-Drainer / Scam lure URLs
2) Trains a fast Character & Subword N-Gram TF-IDF NLP Classifier (char_ngram_model.pkl)
   that catches unseen phishing, mirror, DGA, and scam character patterns automatically.

Usage: python retrain.py
"""
import csv
import os
import pickle
import random
import sqlite3
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features_v2 import FEATURE_NAMES_V2, extract_features_v2, normalize_url_v2 as normalize_url

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
DB_PATH = os.path.join(BASE, "threat_intel.db")
V2_MODEL_PATH = os.path.join(BASE, "model_v2.json")
CHAR_MODEL_PATH = os.path.join(BASE, "char_ngram_model.pkl")

LEGIT_PATHS = [
    "/wiki/Main_Page", "/wiki/Computer_security", "/search?q=laptop+deals",
    "/products/electronics/phones", "/docs/getting-started/installation",
    "/blog/2026/10/engineering-update", "/about-us", "/contact",
    "/user/profile/settings", "/articles/technology-news-today",
    "/watch?v=dQw4w9WgXcQ", "/repos/owner/project/issues/42",
    "/dp/B08N5WRWNW/ref=sr_1_1", "/courses/machine-learning/lectures/1",
]

SYNTHETIC_RISK_DOMAINS = [
    "https://netmirror.app", "https://netmirror.cc", "https://netmirror.gg",
    "https://netmirror.tv/watch/free", "https://hdhub4u.tv/movies/download",
    "https://movierulz.lol/telugu-movies", "https://ibomma.cc/telugu",
    "https://vegamovies.nl/download-hd", "https://123movies.sc/watch-online",
    "https://fmovies.to/home", "https://soap2day.rs/movie",
    "https://tamilrockers.ws/download", "https://filmyzilla.vin/bollywood",
    "https://pikashow.cc/download-apk", "https://thoptv.live/apk",
    "https://happymod.pro/download-mod-apk", "https://liteapks.com/mod-unlocked",
    "https://modapk-unlimited.xyz/download.apk", "https://fitgirl-repacks.cc/game-crack",
    "https://getintopc-crack.site/keygen", "https://1337x.to/torrent/download",
    "https://claim-usdt-airdrop.pages.dev/connect", "https://metamask-restore-seed.vercel.app",
    "https://sbi-kyc-pan-update.netlify.app/login", "https://hdfc-netbanking-verify.web.app",
    "https://telegram-bot-auth-verify.duckdns.org/signin", "https://freefire-diamond-generator.lol",
    "https://1xbet-freespin-bonus.icu/play", "https://rummy-teenpatti-gold.sbs/apk",
]


def url_to_ngram_text(url: str) -> str:
    """Extract normalized hostname + path tokens for character & subword n-gram analysis."""
    try:
        p = urlparse(normalize_url(url))
        host = (p.hostname or "").lower().removeprefix("www.")
        path = (p.path or "").lower()
        # Keep only domain + first segment of path so normal long article paths don't dominate
        first_seg = "/" + path.strip("/").split("/")[0] if path.strip("/") else ""
        return f"{host}{first_seg}"
    except Exception:
        return url.lower()


def load_training_sample():
    urls, labels, weights = [], [], []

    # 1) Load user feedback from SQLite (highest priority, weight 15x)
    if os.path.exists(DB_PATH):
        try:
            rows = sqlite3.connect(DB_PATH).execute("SELECT url, label FROM feedback").fetchall()
            for u, l in rows:
                if u:
                    urls.append(u.strip())
                    labels.append(int(l))
                    weights.append(15.0 if int(l) == 1 else 12.0)
            print(f"[+] Loaded {len(rows)} user feedback reports from threat_intel.db")
        except Exception as e:
            print(f"[!] Feedback table read warning: {e}")

    # 2) Add curated grey-market / mirror / APK / crypto-drainer / phishing kit samples
    for u in SYNTHETIC_RISK_DOMAINS:
        urls.append(u)
        labels.append(1)
        weights.append(6.0)

    # 3) Load Tranco legitimate domains + create Hard Negatives with deep paths
    tranco_path = os.path.join(DATA, "tranco.csv")
    if os.path.exists(tranco_path):
        rng = random.Random(42)
        with open(tranco_path, newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.reader(f)):
                if i >= 25000 or len(r) < 2:
                    break
                dom = r[1].strip().lower()
                base_u = f"https://{dom}"
                urls.append(base_u)
                labels.append(0)
                weights.append(1.0)
                # Add a deep-path legitimate URL for 40% of domains to cancel path-length bias
                if i % 2 == 0:
                    deep_u = base_u + rng.choice(LEGIT_PATHS)
                    urls.append(deep_u)
                    labels.append(0)
                    weights.append(1.5)

    # 4) Load OpenPhish & URLhaus & PhiUSIIL sample
    openphish_path = os.path.join(DATA, "openphish.txt")
    if os.path.exists(openphish_path):
        with open(openphish_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                u = line.strip()
                if u.startswith("http"):
                    urls.append(u)
                    labels.append(1)
                    weights.append(2.0)

    phi_path = os.path.join(DATA, "phiusiil.csv")
    if os.path.exists(phi_path):
        df = pd.read_csv(phi_path, usecols=lambda c: c.lower() in ("url", "label"), nrows=35000)
        df.columns = [c.lower() for c in df.columns]
        for u, l in zip(df["url"].astype(str), df["label"]):
            is_phish = 0 if int(l) == 1 else 1
            urls.append(u)
            labels.append(is_phish)
            weights.append(1.0)

    return urls, np.array(labels, dtype=int), np.array(weights, dtype=float)


def main():
    print("[*] Building training corpus (User Feedback + Hard Negatives + Mirror/Scam + Threat Feeds)...")
    urls, y, w = load_training_sample()
    print(f"[*] Total training samples: {len(urls):,} (Threat={int((y == 1).sum()):,}, Legit={int((y == 0).sum()):,})")

    # --- Part A: Train Character & Subword N-Gram TF-IDF NLP Classifier ---
    print("[*] Training Character N-Gram TF-IDF NLP Classifier (3-5 char n-grams)...")
    texts = [url_to_ngram_text(u) for u in urls]
    char_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=45000, min_df=2, sublinear_tf=True)),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=40, random_state=42)),
    ])
    char_pipe.fit(texts, y, clf__sample_weight=w)
    with open(CHAR_MODEL_PATH, "wb") as f:
        pickle.dump(char_pipe, f)
    print(f"[+] Saved Character N-Gram NLP model -> {CHAR_MODEL_PATH}")

    # --- Part B: Fine-tune / Boost 54-Feature XGBoost Model (model_v2.json) on Feedback + Hard Negatives ---
    if os.path.exists(V2_MODEL_PATH):
        print("[*] Fine-tuning 54-feature XGBoost model (model_v2.json) on Hard Negatives + Mirror/Scam + User Reports...")
        # Select a focused fine-tuning slice: all feedback + all synthetic risk + 8,000 balanced samples
        idx_priority = [i for i, wt in enumerate(w) if wt >= 1.5]
        rng = random.Random(42)
        if len(idx_priority) > 12000:
            idx_priority = rng.sample(idx_priority, 12000)
        X_ft = np.array(
            [[extract_features_v2(urls[i])[n] for n in FEATURE_NAMES_V2] for i in idx_priority],
            dtype=np.float32,
        )
        y_ft = y[idx_priority]
        w_ft = w[idx_priority]

        base_xgb = XGBClassifier()
        base_xgb.load_model(V2_MODEL_PATH)
        tuned_xgb = XGBClassifier(
            n_estimators=45,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        tuned_xgb.fit(X_ft, y_ft, sample_weight=w_ft, xgb_model=base_xgb.get_booster())
        tuned_xgb.save_model(V2_MODEL_PATH)
        print(f"[+] Updated 54-feature XGBoost model -> {V2_MODEL_PATH}")


if __name__ == "__main__":
    main()
