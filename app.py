import re
import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────
_BYLINE_RE = re.compile(
    r"^\s*[A-Z][A-Z\s/,\-]+\s*\([^)]+\)\s*[-\u2013\u2014]\s*",
    re.MULTILINE
)
_AGENCY_RE = re.compile(
    r"\b(reuters|associated press|the ap|bloomberg|cnn|fox news|msnbc|bbc|"
    r"new york times|washington post|huffpost|huffington post|breitbart|"
    r"politico|buzzfeed|vox|the hill|daily beast|salon|raw story|"
    r"ndtv|times of india|the hindu|hindustantimes|india today|"
    r"firstpost|news18|zee news|aaj tak|ani|pti)\b",
    re.IGNORECASE
)

# def wordopt(text: str) -> str:
  #  if not isinstance(text, str):
   #     return ""
    #text = _BYLINE_RE.sub("", text)
    #text = _AGENCY_RE.sub("", text)
    #text = re.sub(r"https?://\S+|www\.\S+", "", text)
    #text = text.lower()
    #text = re.sub(r"<.*?>", "", text)
    #text = re.sub(r"[^\w\s]", "", text)
    #text = re.sub(r"\d", "", text)
    #text = re.sub(r"\s+", " ", text).strip()
    #return text

def wordopt(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # remove HTML
    text = re.sub(r"<.*?>", "", text)

    # keep punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ──────────────────────────────────────────────────────────────
# News summarisation & categorisation
# ──────────────────────────────────────────────────────────────
CATEGORIES = {
    "Politics":      ["president", "congress", "senate", "election", "government",
                      "democrat", "republican", "trump", "biden", "modi", "parliament",
                      "minister", "vote", "campaign", "political", "party", "cabinet"],
    "World News":    ["russia", "china", "india", "europe", "nato", "un ",
                      "united nations", "international", "foreign", "global",
                      "war", "conflict", "peace", "sanctions", "diplomacy"],
    "Economy":       ["economy", "market", "stock", "trade", "tariff", "tax",
                      "gdp", "inflation", "interest rate", "bank", "finance",
                      "budget", "investment", "jobs", "employment", "growth"],
    "Health":        ["virus", "covid", "pandemic", "vaccine", "health",
                      "hospital", "disease", "medical", "doctor", "drug",
                      "outbreak", "therapy", "cancer", "surgery"],
    "Technology":    ["tech", "ai", "artificial intelligence", "software",
                      "startup", "internet", "cyber", "data", "google",
                      "microsoft", "elon musk", "spacex", "robot", "algorithm"],
    "Science/Space": ["nasa", "isro", "space", "satellite", "launch", "mission",
                      "climate", "environment", "research", "study", "scientist",
                      "discovery", "planet", "energy", "nuclear", "quantum"],
    "Law & Justice": ["court", "judge", "jury", "lawsuit", "legal", "crime",
                      "police", "fbi", "cbi", "justice", "investigation",
                      "arrest", "conviction", "attorney", "trial"],
    "Sports":        ["game", "team", "player", "championship", "soccer",
                      "football", "cricket", "basketball", "olympic", "match",
                      "tournament", "coach", "score", "league", "ipl"],
}

def detect_category(text: str) -> str:
    t = text.lower()
    scores = {c: sum(1 for kw in kws if kw in t) for c, kws in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General News"

_STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","of","and","or","but",
    "for","with","that","this","are","was","were","be","been","has","have",
    "had","by","from","as","i","we","they","he","she","not","his","her",
    "their","its","s","said","will","would","could","should","may","also",
    "into","about","up","out","so","no","do","did","if","can","more","than",
    "then","when","there","what","which","who","how","all","just","over",
    "after","before","while","such","its","also","these","those","been",
    "has","have","had","its","per","via","one","two","three","four","five"
}

def extract_bullet_points(text: str, n: int = 5) -> list:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
                 if len(s.split()) > 6]
    if not sentences:
        return ["No summary available."]
    word_freq: dict = {}
    for sent in sentences:
        for w in re.findall(r"\w+", sent.lower()):
            if w not in _STOPWORDS and len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
    scored = [
        (sum(word_freq.get(w.lower(), 0)
             for w in re.findall(r"\w+", s)
             if w.lower() not in _STOPWORDS), i, s)
        for i, s in enumerate(sentences)
    ]
    top = sorted(scored, reverse=True)[:n]
    top.sort(key=lambda x: x[1])
    return [s[2] + ("" if s[2].endswith((".", "!", "?")) else ".") for s in top]

# ──────────────────────────────────────────────────────────────
# DATASET LOADING  – priority: WELFake → True+Fake CSV
# ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
WELFAKE_PATH  = os.path.join(BASE_DIR, "WELFake_Dataset.csv")
TRUE_PATH     = os.path.join(BASE_DIR, "True.csv")
FAKE_PATH     = os.path.join(BASE_DIR, "Fake.csv")

def load_dataset() -> pd.DataFrame:
    # ── Option 1: WELFake (preferred – diverse multi-source dataset) ──
    if os.path.exists(WELFAKE_PATH) and os.path.getsize(WELFAKE_PATH) > 1_000_000:
        print(f"Using WELFake dataset ({os.path.getsize(WELFAKE_PATH)//1_048_576} MB) …")
        df = pd.read_csv(WELFAKE_PATH)
        # WELFake columns: Unnamed:0 / title / text / label
        # label: 0=Fake, 1=Real
        df = df.rename(columns={"Unnamed: 0": "id"})
        if "text" not in df.columns:
            raise ValueError("WELFake CSV does not have a 'text' column.")
        # Some rows have text in title but empty body – merge them
        df["text"] = df.apply(
            lambda r: str(r.get("title", "")) + " " + str(r.get("text", ""))
            if pd.isna(r.get("text")) or str(r.get("text", "")).strip() == ""
            else str(r["text"]),
            axis=1
        )
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        # WELFake convention: 1=Fake, 0=Real  →  invert to match ISOT (1=Real, 0=Fake)
        df["label"] = 1 - df["label"]
        print(f"WELFake loaded: {len(df)} rows  "
              f"(Fake: {(df.label==0).sum()}, Real: {(df.label==1).sum()})")
        return df

    # ── Option 2: Original ISOT True.csv + Fake.csv ──
    if os.path.exists(TRUE_PATH) and os.path.exists(FAKE_PATH):
        print("Using original True.csv + Fake.csv dataset …")
        print("TIP: Run `python download_dataset.py` to get the broader WELFake dataset.")
        true_df = pd.read_csv(TRUE_PATH); true_df["label"] = 1
        fake_df = pd.read_csv(FAKE_PATH); fake_df["label"] = 0
        df = pd.concat([true_df, fake_df], axis=0)[["text", "label"]].dropna()
        print(f"ISOT loaded: {len(df)} rows  "
              f"(Fake: {(df.label==0).sum()}, Real: {(df.label==1).sum()})")
        return df

    raise FileNotFoundError(
        "No dataset found! Place WELFake_Dataset.csv (or True.csv + Fake.csv) "
        f"in {BASE_DIR} and restart."
    )

# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
news = load_dataset()
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

print("Preprocessing text …")
news["text"] = news["text"].apply(wordopt)
# Drop rows that became empty after cleaning
news = news[news["text"].str.strip().astype(bool)]

x = news["text"]
y = news["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(x_train)}  —  Test: {len(x_test)}")

print("Fitting TF-IDF (sublinear_tf + bigrams) …")
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    max_features=300000,
   # strip_accents="unicode",
    # token_pattern=r"\w{2,}",
)
xv_train = vectorizer.fit_transform(x_train)
xv_test  = vectorizer.transform(x_test)

print("Training Logistic Regression …")
LR = LogisticRegression(C=4, max_iter=2000,class_weight="balanced", n_jobs=-1)
LR.fit(xv_train, y_train)

print("Training Passive-Aggressive Classifier …")
PAC = PassiveAggressiveClassifier(C=0.5,class_weight="balanced",  max_iter=2000, random_state=42)
PAC.fit(xv_train, y_train)

print("Training Random Forest …")
RFC = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                              n_jobs=-1, random_state=42)
RFC.fit(xv_train, y_train)

lr_acc  = LR.score(xv_test, y_test)
pac_acc = PAC.score(xv_test, y_test)
rfc_acc = RFC.score(xv_test, y_test)
print(f"Models ready!  LR={lr_acc:.3f}  PAC={pac_acc:.3f}  RFC={rfc_acc:.3f}")

# ──────────────────────────────────────────────────────────────
# FAKE SIGNAL ANALYSIS  – why the article looks suspicious
# ──────────────────────────────────────────────────────────────
_SENSATIONAL = [
    "breaking", "shocking", "bombshell", "explosive", "urgent", "miracle",
    "unbelievable", "you won't believe", "stunning", "outrage", "scandalous",
]
_CONSPIRACY = [
    "government hiding", "doctors hiding", "they don't want you to know",
    "mainstream media won't tell", "suppressed", "cover-up", "coverup",
    "big pharma", "deep state", "share before deleted", "share before it gets removed",
    "wake up sheeple", "the truth about", "what they're hiding",
]
_MIRACLE = [
    "miracle cure", "cures all", "100% proven", "doctors hate", "secret remedy",
    "ancient remedy", "cure for cancer", "cure covid", "guaranteed to", "instant cure",
]
_ANON = [
    "sources say", "insiders say", "anonymous source", "sources close to",
    "unnamed official", "insider reveals",
]
_EMOTIONAL = [
    "every american must", "everyone must share", "share this now", "forward this",
    "wake up", "open your eyes", "they are lying", "don't be fooled",
    "you need to see this", "must read", "urgent action needed",
]
# Signals that suggest a LEGITIMATE, professionally written news article
_LEGIT_MARKERS = [
    "said in a statement", "told reporters", "according to", "per cent", "percent",
    "prime minister", "chief minister", "ministry of", "government of india",
    "supreme court", "high court", "reserve bank", "rbi", "sebi", "isro", "drdo",
    "issued a statement", "press conference", "fiscal year", "quarter", "gdp",
    "election commission", "lok sabha", "rajya sabha", "parliament",
    "reuters", "ani", "pti", "ndtv", "the hindu", "times of india",
    "associated press", "bbc", "al jazeera", "bloomberg", "financial times",
    "announced", "confirmed", "stated", "in a statement",
]


def analyze_fake_signals(raw_text: str) -> list[str]:
    """Return a list of human-readable reasons why the article looks suspicious."""
    reasons = []
    t = raw_text.lower()

    found_sens = [w for w in _SENSATIONAL if w in t]
    if found_sens:
        reasons.append(
            f"Sensationalist language detected: '{', '.join(found_sens[:3])}' — "
            "real journalism avoids trigger words designed to provoke emotion."
        )

    found_cons = [p for p in _CONSPIRACY if p in t]
    if found_cons:
        reasons.append(
            "Contains conspiracy-style phrases suggesting information suppression "
            f"(\'{found_cons[0]}'). Legitimate reporting cites verifiable sources."
        )

    found_mir = [p for p in _MIRACLE if p in t]
    if found_mir:
        reasons.append(
            "Makes extraordinary medical or scientific claims without citing "
            "peer-reviewed evidence or named experts."
        )

    found_anon = [p for p in _ANON if p in t]
    if found_anon:
        reasons.append(
            "Relies on anonymous or unnamed sources — credible news outlets "
            "attribute claims to named, accountable individuals."
        )

    found_emo = [p for p in _EMOTIONAL if p in t]
    if found_emo:
        reasons.append(
            "Uses emotional urgency to pressure sharing "
            f"('{found_emo[0]}') — a hallmark of viral misinformation."
        )

    excl = raw_text.count('!')
    if excl >= 3:
        reasons.append(
            f"Excessive exclamation marks ({excl}) — credible articles rarely use "
            "them in the body text."
        )

    caps_words = list(set(re.findall(r'\b[A-Z]{4,}\b', raw_text)))
    if len(caps_words) >= 3:
        sample = ', '.join(caps_words[:4])
        reasons.append(
            f"Unusual ALL-CAPS usage ({sample}…) — often used to create false urgency."
        )

    if not reasons:
        reasons.append(
            "The AI models detected statistical language patterns in this article "
            "that closely match known fake/misleading news in the training corpus. "
            "The vocabulary, sentence structure, or topic framing differs significantly "
            "from verified news sources."
        )
    return reasons


def legitimacy_score(raw_text: str) -> int:
    """Count how many formal-news markers appear in the text (0 = none, higher = more legit)."""
    t = raw_text.lower()
    return sum(1 for m in _LEGIT_MARKERS if m in t)


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json(force=True)
    raw_text = (data.get("text") or "").strip()
    if not raw_text:
        return jsonify({"error": "No text provided."}), 400

    cleaned = wordopt(raw_text)
    vec     = vectorizer.transform([cleaned])

    MODELS = {
        "Logistic Regression": LR,
        "Passive-Aggressive":  PAC,
        
    }

    results = {}
    votes   = []
    confs   = []
    for name, model in MODELS.items():
        pred = int(model.predict(vec)[0])
        votes.append(pred)
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(vec)[0])) * 100
        else:
            score      = abs(float(model.decision_function(vec)[0]))
            confidence = min(50 + score * 15, 99)
        confs.append(confidence)
        results[name] = {
            "label":      "Real News" if pred == 1 else "Fake News",
            "is_real":    pred == 1,
            "confidence": round(float(confidence), 1),
        }

    #majority  = 1 if sum(votes) >= 2 else 0
    #avg_conf  = sum(confs) / len(confs)
    #unanimous = len(set(votes)) == 1

    # ── Smarter uncertainty logic ────────────────────────────────
    # 1. Models disagree or avg confidence is low  → Uncertain
    ##uncertain = (not unanimous) or avg_conf < 68

    # 2. Legitimacy heuristic override:
    #    If the article reads like formal professional news (datelines,
    #    institutional names, neutral language) but the model leans Fake
    #    with moderate confidence, downgrade to Uncertain rather than Fake.
    #    This specifically helps Indian / international news that the
    #    US-centric training corpus under-represents.
   # legit = legitimacy_score(raw_text)
    #if majority == 0 and legit >= 3 and avg_conf < 88:
    #    uncertain = True

   # if uncertain:
    #    overall_label = "Uncertain"
    #    is_real       = None
    #else:
   #     overall_label = "Real News" if majority == 1 else "Fake News"
    #    is_real       = majority == 1


     # ───────────────── Final decision (NO UNCERTAIN) ─────────────────

    # Use Logistic Regression probability as primary signal
    prob_real = LR.predict_proba(vec)[0][1] * 100

    # legitimacy boost (helps real professional news)
    legit = legitimacy_score(raw_text)

    # combine both signals
    final_score = prob_real + legit * 3

    if final_score >= 60:
        overall_label = "Real News"
        is_real = True
    else:
        overall_label = "Fake News"
        is_real = False

    avg_conf = prob_real

    response = {
        "overall": overall_label,
        "is_real": is_real,
        "uncertain": False,
        "avg_conf": round(float(avg_conf), 1),
        "legit_score": legit,
        "models": results,
    }

    if is_real is True:
        response["summary"] = {
            "category": detect_category(raw_text),
            "bullets": extract_bullet_points(raw_text, n=5),
        }

    # Fake reasons
    if is_real is False:
        response["fake_reasons"] = analyze_fake_signals(raw_text)

    return jsonify(response)