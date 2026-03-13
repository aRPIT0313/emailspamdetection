# 📧 Email Spam Detection using NLP & Naive Bayes

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4A90D9?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=flat)

> An NLP pipeline that classifies emails as **Spam or Ham** using text preprocessing, TF-IDF vectorization, and Naive Bayes — implemented with an interactive Streamlit interface.
---

## 🎯 Problem Statement

Spam emails waste time and pose security risks (phishing, scams). The challenge in spam detection is high precision — wrongly flagging a legitimate email as spam (false positive) is costly. This project compares three Naive Bayes variants to find the best precision-recall tradeoff.

---

## 🏗️ Project Structure

```
email-spam-detection/
├── data/
│   └── mail.csv                  # 2010 raw emails, cleaned to 1787
├── notebooks/
│   ├── 01_EDA.ipynb              # Distribution, wordcloud, stats
│   ├── 02_preprocessing.ipynb    # NLP cleaning pipeline
│   └── 03_modelling.ipynb        # Model training & comparison
├── src/
│   ├── preprocess.py             # transform_mail() function
│   └── model.py
├── app/
│   ├── app.py                    # Streamlit interface
│   ├── model.pkl                 # Saved MultinomialNB model
│   └── vectorizer.pkl            # Saved TF-IDF vectorizer
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

### 1. Dataset
- **Source:** `mail.csv` — 2010 emails, cleaned to **1787 after deduplication**
- **Columns:** `result` (spam/ham label) + `mail` (email text)
- **Class distribution:** visualized via pie chart

### 2. EDA — Key Insight
Spam messages are consistently **longer** than legitimate emails — higher character count, word count, and sentence count. This guided feature selection.

| Feature | Spam (avg) | Ham (avg) |
|---|---|---|
| Characters | — | — |
| Words | — | — |
| Sentences | — | — |

> 📝 Fill in from your EDA notebook output.

### 3. NLP Preprocessing Pipeline

Every email goes through 5 steps via `transform_mail()`:

| Step | Operation | Tool |
|---|---|---|
| 1 | Lowercasing | Python built-in |
| 2 | Tokenization | `nltk.word_tokenize()` |
| 3 | Remove special characters | `isalnum()` filter |
| 4 | Stopword removal | `nltk.corpus.stopwords` |
| 5 | Stemming | `PorterStemmer` |

```python
def transform_mail(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
```

### 4. Feature Extraction

Two techniques compared:

| Technique | Shape | Purpose |
|---|---|---|
| CountVectorizer (BoW) | 1787 × 3770 | Word frequency matrix |
| **TF-IDF Vectorizer** ✅ | 1787 × 3770 | Weights rare but meaningful words higher |

TF-IDF selected — reduces noise from common words like "the", "is", boosts discriminative words like "lottery", "winner", "claim".

**Train/Test Split:** 60% / 40% (`random_state=2`)

---

## 📊 Model Results

| Model | Accuracy | Precision |
|---|---|---|
| GaussianNB | 87% | 0.73 |
| BernoulliNB | 91% | **0.99** |
| **MultinomialNB** ✅ | **95%** | **0.91** |

**Selected: Multinomial Naive Bayes**

Why? Best balance between accuracy (95%) and precision (0.91). BernoulliNB has higher precision but lower accuracy — it over-flags spam. MultinomialNB is optimized for word-count features, which is exactly what TF-IDF produces.

> In spam detection, **precision is the critical metric** — a false positive (legitimate email flagged as spam) is a user trust problem.

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/email-spam-detection.git
cd email-spam-detection
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/app.py
```

### Predict via code
```python
import pickle
from src.preprocess import transform_mail

model = pickle.load(open('app/model.pkl', 'rb'))
vectorizer = pickle.load(open('app/vectorizer.pkl', 'rb'))

email = "Congratulations! You've won a ₹50,000 prize. Click now to claim."
processed = transform_mail(email)
vector = vectorizer.transform([processed])
result = model.predict(vector)
print("SPAM" if result[0] == 1 else "NOT SPAM")
```

---

## 🌐 Streamlit App Flow

```
User enters email text
        ↓
transform_mail() — 5-step NLP preprocessing
        ↓
TF-IDF vectorization
        ↓
MultinomialNB.predict()
        ↓
Display: SPAM 🚨 or NOT SPAM ✅
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| NLP | NLTK (tokenization, stopwords, stemming) |
| ML | scikit-learn (GaussianNB, MultinomialNB, BernoulliNB) |
| Vectorization | TF-IDF, CountVectorizer |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | Streamlit |
| Serialization | Pickle |

---

## 🔮 Future Improvements

- [ ] Replace PorterStemmer with lemmatization for better semantic accuracy
- [ ] Try BERT embeddings for contextual understanding
- [ ] Add confidence score display in the UI
- [ ] Extend to multi-class: spam / phishing / promotional / legitimate

---
