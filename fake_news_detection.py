import argparse
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords


# Download stopwords once (safe to rerun)
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean and normalize input text for ML."""
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)  # Remove numbers

    tokens = text.split()
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in STOP_WORDS]
    return " ".join(tokens)


def build_demo_dataset() -> pd.DataFrame:
    """Return a tiny in-memory dataset so the pipeline can run without external files."""
    data = {
        "title": [
            "Scientists confirm climate report",
            "Shocking secret cure hidden from public",
            "Government releases annual budget details",
            "Celebrity cloned in secret lab",
            "WHO publishes vaccination effectiveness study",
            "Aliens elected as city mayor overnight",
            "Stock market closes higher after policy update",
            "Miracle coin doubles money every hour",
            "University report shows improved literacy rates",
            "Fake portal claims moon made of cheese",
        ],
        "text": [
            "Peer-reviewed report confirms long-term warming trends.",
            "Anonymous blog says hospitals hide a miracle herb treatment.",
            "The finance ministry published audited numbers for the fiscal year.",
            "A viral post claims a movie star was cloned by scientists.",
            "Clinical data shows vaccines reduce severe disease significantly.",
            "Satirical social account posted that aliens won local elections.",
            "Investors reacted positively after central bank policy statement.",
            "Scam website promises unrealistic crypto returns with no risk.",
            "Education board data indicates literacy has improved this year.",
            "Conspiracy forum insists moon rocks are made of dairy products.",
        ],
        "label": ["REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE"],
    }
    return pd.DataFrame(data)


def load_dataset(data_path: str, use_demo: bool) -> pd.DataFrame:
    """Load dataset from CSV or use a demo dataset."""
    if use_demo:
        print("Using built-in demo dataset (small, only for pipeline testing).")
        return build_demo_dataset()

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Download a fake-news CSV dataset and place it at this path, or run with --demo."
        )

    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake News Detection using TF-IDF + Logistic Regression")
    parser.add_argument(
        "--data-path",
        default="data/fake_or_real_news.csv",
        help="Path to CSV with columns such as title, text, and label.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a tiny built-in demo dataset (for quick testing).",
    )
    args = parser.parse_args()

    # 1) Load Dataset
    df = load_dataset(args.data_path, args.demo)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain at least 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()

    # Combine title + text if title exists
    if "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    # 2) Clean Text Data
    df["clean_content"] = df["content"].apply(clean_text)

    # Convert labels to numeric if needed
    if df["label"].dtype == "object":
        df["label"] = df["label"].str.upper().map({"FAKE": 0, "REAL": 1})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if df["label"].nunique() < 2:
        raise ValueError("Need at least two classes in label column for training.")

    X = df["clean_content"]
    y = df["label"]

    # 3) Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4) TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF test shape:", X_test_tfidf.shape)

    # 5) Model Training
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    # 6) Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Fake News Detection")
    plt.tight_layout()
    plt.show()

    # Top important terms for interpretability
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[0]

    top_fake_idx = np.argsort(coefs)[:10]
    top_real_idx = np.argsort(coefs)[-10:]

    print("\nTop terms indicating FAKE news:")
    print(feature_names[top_fake_idx])

    print("\nTop terms indicating REAL news:")
    print(feature_names[top_real_idx])


if __name__ == "__main__":
    main()
