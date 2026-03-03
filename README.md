# 📰 Fake News Detection Web Application

A Machine Learning powered web application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) and a trained Scikit-learn model.

The system provides a Flask REST API and a responsive web interface for real-time predictions.

---

## 🚀 Features

- Real-time Fake vs Real news classification
- Clean and responsive user interface
- RESTful API endpoint
- TF-IDF based NLP feature extraction
- Pretrained model integration
- Deployment-ready project structure

---

## 🧠 Machine Learning Pipeline

1. Text Preprocessing
   - Lowercasing
   - URL removal
   - Punctuation cleaning
   - Whitespace normalization

2. Feature Engineering
   - TF-IDF Vectorization
   - Sparse matrix transformation

3. Model
   - Logistic Regression (or any trained classifier)
   - Serialized using Pickle

4. Inference Flow

   User Input  
   ↓  
   Clean Text  
   ↓  
   TF-IDF Transform  
   ↓  
   Model Prediction  
   ↓  
   JSON Response  

---

## 📁 Project Structure

```bash
fake-news-detection/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate Environment

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🔌 API Documentation

### POST /predict

Request Body (JSON):

```json
{
  "text": "Your news article text here"
}
```

Response:

```json
{
  "prediction": "Real News"
}
```

---

## 🧰 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- TF-IDF

---

## 🌍 Deployment Options

This project can be deployed on:

- Render
- Railway
- Hugging Face Spaces
- Docker
- AWS
- Azure

---

## 📈 Future Improvements

- Add confidence probability score
- Compare multiple ML models
- Integrate live News API
- Convert backend to FastAPI
- Add Docker containerization

---
