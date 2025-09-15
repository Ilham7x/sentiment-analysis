import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from utils_text import preprocess

RAW_PATH = Path("data/raw/amazon_reviews.csv")
PROC_PATH = Path("data/processed/reviews_clean.csv")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
CHART_DIR = Path("outputs/charts")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)
CHART_DIR.mkdir(exist_ok=True, parents=True)

def load_dataset() -> pd.DataFrame:
    """
    Try to load a CSV from data/raw. If not present, build a tiny sample
    so the pipeline is runnable.
    """
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    # Fallback mini dataset
    print("No data/raw/amazon_reviews.csv found. Using a tiny fallback sample.")
    sample = {
        "review_text": [
            "Amazing product, works perfectly and arrived fast!",
            "Terrible quality. Broke after one week.",
            "It’s okay, not great, not terrible.",
            "Loved it! Excellent build and battery life.",
            "Worst purchase ever. Do not recommend.",
            "Decent for the price, but shipping was slow."
        ],
        "rating": [5, 1, 3, 5, 1, 3]
    }
    return pd.DataFrame(sample)


def pick_text_and_label(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:

    if "reviews.text" in df.columns:
        if "reviews.title" in df.columns:
            X = (
                df["reviews.title"].astype(str).fillna("") + ". " +
                df["reviews.text"].astype(str).fillna("")
            )
        else:
            X = df["reviews.text"].astype(str).fillna("")
    else:
        # Fallbacks if someone uses different headers
        text_col_candidates = [
            "reviewText", "review_text", "review_body", "body", "text",
            "content", "comment", "review"
        ]
        text_col = next((c for c in text_col_candidates if c in df.columns), None)
        if text_col is None:
            raise ValueError("Could not find a text column (expected 'reviews.text').")
        X = df[text_col].astype(str).fillna("")

    # ---- RATING -> SENTIMENT ----
    if "reviews.rating" not in df.columns:
        raise ValueError("Could not find 'reviews.rating' column in the CSV.")

    def to_numeric_rating(val):
        """Handle both numeric and string like '4.0 out of 5 stars'."""
        import re
        if pd.isna(val):
            return None
        try:
            return float(val)
        except Exception:
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            return float(m.group(1)) if m else None

    rnum = df["reviews.rating"].apply(to_numeric_rating)

    def map_rating(r):
        if r is None:
            return "neutral"
        if r <= 2:
            return "negative"
        elif r >= 4:
            return "positive"
        else:
            return "neutral"

    y = rnum.apply(map_rating)

    vc = y.value_counts()
    print("Label distribution:\n", vc)
    if len(vc) < 2:
        raise ValueError(
            f"Only one class after mapping: {vc.index.tolist()}. "
            "Check 'reviews.rating' values."
        )

    return X, y


def preprocess_and_save(X: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"text": X, "label": y})
    df["text_clean"] = df["text"].apply(preprocess)
    df.to_csv(PROC_PATH, index=False)
    print(f"Saved cleaned data -> {PROC_PATH}")
    return df

def train_and_eval(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF + Logistic Regression pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=50000,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            solver="saga",
            class_weight="balanced" 
        ))
    ])

    pipe.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["negative","neutral","positive"],
                yticklabels=["negative","neutral","positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix -> {CHART_DIR / 'confusion_matrix.png'}")

    # Save misclassified examples for error analysis
    mis = pd.DataFrame({
        "text": X_test,
        "true": y_test,
        "pred": y_pred
    })
    mis = mis[mis["true"] != mis["pred"]]
    mis.to_csv(OUT_DIR / "misclassified.csv", index=False)
    print(f"Saved misclassified examples -> {OUT_DIR / 'misclassified.csv'}")

    # Save artifacts (entire pipeline)
    joblib.dump(pipe, MODEL_DIR / "tfidf_logreg_pipeline.joblib")
    print(f"Saved model pipeline -> {MODEL_DIR / 'tfidf_logreg_pipeline.joblib'}")

def main():
    df_raw = load_dataset()
    X, y = pick_text_and_label(df_raw)
    df_clean = preprocess_and_save(X, y)
    train_and_eval(df_clean)

if __name__ == "__main__":
    main()