"""
fakenews_improved.py
═════════════════════════════════════════════════════════════════
Improved Fake News Detection Pipeline
(Run this script as an alternative to the notebook.
 It trains the same models as app.py and prints classification reports.)

Key improvements over the original notebook
────────────────────────────────────────────
1. wordopt() now STRIPS Reuters/AP bylines like "WASHINGTON (Reuters) –"
   and wipes common news-agency names so the model cannot cheat by recognising
   journalistic formatting rather than actual content.
2. TfidfVectorizer uses sublinear_tf=True + bigrams + min_df for better features.
3. Replaced DecisionTreeClassifier with PassiveAggressiveClassifier, which is
   specifically designed for streaming-text tasks and avoids overfitting.
4. All 44 898 articles from True.csv + Fake.csv are used correctly.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ──────────────────────────────────────────────
# 1. Load datasets
# ──────────────────────────────────────────────
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
true_df["label"] = 1   # 1 = Real
fake_df["label"] = 0   # 0 = Fake

news = pd.concat([fake_df, true_df], axis=0)
news = news.drop(["title", "subject", "date"], axis=1)
news = news[news["text"].str.strip().astype(bool)]   # drop empty rows
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(news)} articles  —  "
      f"Fake: {(news.label==0).sum()}  |  Real: {(news.label==1).sum()}")

# ──────────────────────────────────────────────
# 2. Improved Preprocessing
# ──────────────────────────────────────────────
# "CITY (Agency) –" patterns that appear only in True.csv corrupt the model
_BYLINE_RE = re.compile(
    r"^\s*[A-Z][A-Z\s/,\-]+\s*\([^)]+\)\s*[-\u2013\u2014]\s*",
    re.MULTILINE
)
# Wire-service / media names that are strong but spurious signals
_AGENCY_RE = re.compile(
    r"\b(reuters|associated press|the ap|bloomberg|cnn|fox news|msnbc|bbc|"
    r"new york times|washington post|huffpost|huffington post|breitbart|"
    r"politico|buzzfeed|vox|the hill|daily beast|salon|raw story)\b",
    re.IGNORECASE
)


def wordopt(text: str) -> str:
    text = _BYLINE_RE.sub("", text)          # remove "CITY (Agency) –"
    text = _AGENCY_RE.sub("", text)          # remove news-agency names
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Preprocessing …")
news["text"] = news["text"].apply(wordopt)

x = news["text"]
y = news["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {len(x_train)}  —  Test: {len(x_test)}")

# ──────────────────────────────────────────────
# 3. TF-IDF Vectoriser (improved settings)
# ──────────────────────────────────────────────
print("Fitting TF-IDF (sublinear_tf + bigrams) …")
vectorizer = TfidfVectorizer(
    sublinear_tf=True,        # log1p(tf) — dampens very common terms
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=3,                 # ignore terms in < 3 documents
    max_features=150_000,
    strip_accents="unicode",
    token_pattern=r"\w{2,}",  # only multi-char tokens
)
xv_train = vectorizer.fit_transform(x_train)
xv_test  = vectorizer.transform(x_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ──────────────────────────────────────────────
# 4. Model Training
# ──────────────────────────────────────────────
print("\n──  Logistic Regression ──")
LR = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print(f"Accuracy: {accuracy_score(y_test, pred_lr):.4f}")
print(classification_report(y_test, pred_lr, target_names=["Fake", "Real"]))

print("\n──  Passive-Aggressive Classifier ──")
PAC = PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42)
PAC.fit(xv_train, y_train)
pred_pac = PAC.predict(xv_test)
print(f"Accuracy: {accuracy_score(y_test, pred_pac):.4f}")
print(classification_report(y_test, pred_pac, target_names=["Fake", "Real"]))

print("\n──  Random Forest (200 trees) ──")
RFC = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
print(f"Accuracy: {accuracy_score(y_test, pred_rfc):.4f}")
print(classification_report(y_test, pred_rfc, target_names=["Fake", "Real"]))

# ──────────────────────────────────────────────
# 5. Manual testing helper
# ──────────────────────────────────────────────
def output_label(n): return "✅ Real News" if n == 1 else "🚨 Fake News"


def manual_testing(article: str):
    cleaned = wordopt(article)
    vec = vectorizer.transform([cleaned])
    pred_lr_m  = LR.predict(vec)[0]
    pred_pac_m = PAC.predict(vec)[0]
    pred_rfc_m = RFC.predict(vec)[0]
    print(f"\n  LR  → {output_label(pred_lr_m)}")
    print(f"  PAC → {output_label(pred_pac_m)}")
    print(f"  RFC → {output_label(pred_rfc_m)}")
    majority = 1 if pred_lr_m + pred_pac_m + pred_rfc_m >= 2 else 0
    print(f"\n  ═══ Majority Verdict: {output_label(majority)} ═══")


if __name__ == "__main__":
    print("\n\nType a news article to test (blank line to quit):")
    while True:
        lines, line = [], input()
        while line:
            lines.append(line)
            line = input()
        if not lines:
            break
        manual_testing(" ".join(lines))
