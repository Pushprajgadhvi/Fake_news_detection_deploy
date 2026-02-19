# 📰 Fake News Detection Using Machine Learning

This project uses machine learning models to detect whether a news article is **real** or **fake**, based on the article's text. It includes models like Logistic Regression (LR) and Random Forest Classifier (RFC), trained on a labeled dataset of news articles.

---

## Problem Statement

With the rapid growth of online media, misinformation spreads quickly. Manual verification is slow and expensive. This project develops an automated system capable of identifying fake news articles based on textual content.

---

## Dataset

We use a labeled dataset containing news articles with the following fields:

* title
* text
* label (FAKE / REAL)

Typical dataset sources:

* Kaggle Fake News Dataset
* ISOT Fake News Dataset

---

## Project Pipeline

### 1. Data Preprocessing

* Lowercasing text
* Removing punctuation
* Removing stopwords
* Tokenization
* Lemmatization

### 2. Feature Engineering

Text is converted into numerical vectors using:

**TF-IDF Vectorization**

* Unigrams and bigrams
* Frequency importance weighting

### 3. Model Training

We train machine learning classifiers on vectorized text:

* Passive Aggressive Classifier
* Logistic Regression
* Linear SVM

### 4. Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 5. Prediction System

User can input a news article → model predicts:

**REAL NEWS** or **FAKE NEWS**

---

## Results

Typical performance achieved:

* Accuracy: ~92–98% (depending on dataset)
* Balanced precision & recall

---

## Future Improvements

* Deep learning models (LSTM / Transformers)
* Detect sarcasm & satire articles
* Multi-language support
* Web interface deployment

---

## Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* nltk / spacy

---

## Learning Outcomes

This project demonstrates:

* End-to-end NLP pipeline
* Text vectorization techniques
* Classification modeling
* Model evaluation and validation
* Real-world misinformation detection application

---
