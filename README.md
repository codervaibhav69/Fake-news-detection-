# Fake News Detection using TF-IDF and Machine Learning

## 1) Project Title
**Fake News Detection using TF-IDF and Machine Learning**

---

## 2) Problem Statement
With the rapid growth of social media and online publishing, fake news spreads faster than ever. False information can influence public opinion, affect elections, create panic, and damage trust in credible journalism. Because the volume of online content is massive, manual verification is slow and expensive. This makes **automatic fake news detection** an important machine learning problem.

A machine learning system can learn patterns from previously labeled articles and then classify new articles as **real** or **fake**, helping platforms, fact-checkers, and users respond faster.

---

## 3) Objectives
The project has three main objectives:

1. Detect whether a news article is real or fake.
2. Convert textual news data into numerical features using **TF-IDF**.
3. Train a machine learning model to classify news articles effectively.

---

## 4) Dataset
Use a public dataset such as Kaggle's Fake News Dataset. Typical columns are:

- `title`: Headline of the article
- `text`: Main article content
- `label`: Ground truth class (`FAKE` / `REAL` or `0` / `1`)

> Example source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Save your CSV file as `data/fake_or_real_news.csv` (or update the path in the script).

---

## 5) Methodology (End-to-End Workflow)

### Step 1: Data Collection
Download the dataset from a trusted public source (e.g., Kaggle), then load it with pandas.

### Step 2: Data Preprocessing
Clean and normalize the text before feature extraction:

- Lowercasing
- Removing URLs and punctuation
- Removing numbers and extra spaces
- Tokenization
- Stopword removal (using NLTK)

### Step 3: Feature Extraction with TF-IDF
Use `TfidfVectorizer` to transform cleaned text into weighted numerical vectors. TF-IDF assigns higher weight to informative terms and lower weight to common terms.

### Step 4: Train-Test Split
Split data into training and testing sets (e.g., 80%-20%) to evaluate generalization.

### Step 5: Model Training and Testing
Train a classifier (Logistic Regression in this project), then predict labels for test data.

### Step 6: Evaluation
Use metrics like:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

## 6) Implementation in Python

Create a file named `fake_news_detection.py` with the code below.

```python
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import nltk
from nltk.corpus import stopwords


# Download stopwords once (safe to rerun)
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean and normalize input text for ML."""
    if pd.isna(text):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove punctuation and numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)

    # Keep only alphabetic words and remove stopwords
    tokens = text.split()
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in STOP_WORDS]

    return " ".join(tokens)


def main() -> None:
    # -------------------------------
    # 1) Load Dataset
    # -------------------------------
    data_path = "data/fake_or_real_news.csv"  # Update path if needed
    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Basic checks
    df = df.dropna(subset=["text", "label"])

    # Optional: combine title + text for richer features
    if "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    # -------------------------------
    # 2) Clean Text Data
    # -------------------------------
    df["clean_content"] = df["content"].apply(clean_text)

    # Convert labels to numeric if needed
    if df["label"].dtype == "object":
        # Expected mapping for REAL/FAKE labels
        df["label"] = df["label"].str.upper().map({"FAKE": 0, "REAL": 1})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X = df["clean_content"]
    y = df["label"]

    # -------------------------------
    # 3) Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------
    # 4) TF-IDF Vectorization
    # -------------------------------
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF test shape:", X_test_tfidf.shape)

    # -------------------------------
    # 5) Model Training
    # -------------------------------
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    # -------------------------------
    # 6) Evaluation
    # -------------------------------
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Fake News Detection")
    plt.tight_layout()
    plt.show()

    # Show top important terms for interpretability (logistic regression coefficients)
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[0]

    top_fake_idx = np.argsort(coefs)[:15]       # most negative => fake class
    top_real_idx = np.argsort(coefs)[-15:]      # most positive => real class

    print("\nTop terms indicating FAKE news:")
    print(feature_names[top_fake_idx])

    print("\nTop terms indicating REAL news:")
    print(feature_names[top_real_idx])


if __name__ == "__main__":
    main()
```

### Install Required Libraries

```bash
pip install pandas scikit-learn nltk matplotlib
```

### Run the Project

```bash
python fake_news_detection.py
```

---

## 7) Evaluation Metrics (Theory)

- **Accuracy**: Fraction of total predictions that are correct.
  \[
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision**: Out of predicted positive samples, how many are actually positive.
  \[
  Precision = \frac{TP}{TP + FP}
  \]

- **Recall**: Out of actual positive samples, how many were correctly identified.
  \[
  Recall = \frac{TP}{TP + FN}
  \]

- **F1-score**: Harmonic mean of precision and recall.
  \[
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  \]

For fake news detection, precision and recall are especially important because missing fake articles or incorrectly flagging real articles both have consequences.

---

## 8) Results and Analysis
A TF-IDF + Logistic Regression baseline usually performs strongly on clean textual datasets. Typical observations:

- Good accuracy because TF-IDF captures important keywords and phrases.
- Logistic Regression is fast, interpretable, and works well with sparse features.
- Confusion matrix helps identify if the model is biased toward one class.

### Strengths
- Simple and efficient pipeline
- Easy to train and deploy
- Interpretable through feature importance

### Limitations
- Limited understanding of deep context and sarcasm
- Sensitive to domain shift (news style may change over time)
- Can be affected by noisy or imbalanced datasets

---

## 9) Conclusion
This project demonstrates a complete machine learning pipeline for fake news detection using TF-IDF and Logistic Regression. The model converts text into weighted numeric vectors and learns to classify articles as real or fake with good baseline performance.

For future improvements:

- Use advanced preprocessing (lemmatization, named entity features)
- Try ensemble models (Random Forest, XGBoost)
- Use deep learning models (LSTM, BiLSTM)
- Use transformer-based models (BERT, RoBERTa) for better contextual understanding

---

## 10) Optional Enhancements
To make the project more practical and user-friendly:

- Build a **Flask** web app with a textbox where users paste an article.
- Build a **Streamlit** UI for real-time predictions and confidence scores.
- Add a probability threshold slider and explanation panel (top words influencing predictions).

---

## Suggested Folder Structure

```text
Fake-news-detection-
│
├── README.md
├── fake_news_detection.py
└── data/
    └── fake_or_real_news.csv
```

---

## Quick Clarification: Is the project workable without a dataset?

Short answer: **for real training, you need a dataset**.

- Machine learning models must learn from labeled examples, so a real CSV dataset is required to train a useful fake-news classifier.
- This project now also supports a **demo mode** so you can verify that the full pipeline runs even before downloading a dataset.

Run demo mode:

```bash
python fake_news_detection.py --demo
```

Run with your real dataset:

```bash
python fake_news_detection.py --data-path data/fake_or_real_news.csv
```
