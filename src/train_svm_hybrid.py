import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

PROC_PATH = Path("data/processed/reviews_clean.csv")  
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
CHART_DIR = Path("outputs/charts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_clean_data() -> pd.DataFrame:
    if not PROC_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PROC_PATH}"
        )
    df = pd.read_csv(PROC_PATH)
    if not {"text_clean", "label"}.issubset(df.columns):
        raise ValueError("Clean file must contain 'text_clean' and 'label' columns.")
    return df

def hybrid_resample(X_train: pd.Series, y_train: pd.Series):
    rng = np.random.RandomState(RANDOM_STATE)

    # current counts
    counts = Counter(y_train)
    print("Before hybrid sampling:", counts)

    TARGETS = {
        "positive": min(counts.get("positive", 0), 12000),  # (undersample)
        "neutral":  max(counts.get("neutral", 0), 3000),    # (oversample)
        "negative": max(counts.get("negative", 0), 2000),   # (oversample)
    }

    # Build new indices by sampling per class
    idxs_new = []
    for label, target in TARGETS.items():
        idxs = np.where(y_train.values == label)[0]
        if len(idxs) == 0:
            continue
        if target <= len(idxs):
            # undersample without replacement
            chosen = rng.choice(idxs, size=target, replace=False)
        else:
            # oversample with replacement
            chosen = rng.choice(idxs, size=target, replace=True)
        idxs_new.append(chosen)

    if not idxs_new:
        raise RuntimeError("No indices selected during hybrid sampling.")
    idxs_new = np.concatenate(idxs_new)
    rng.shuffle(idxs_new)

    X_res = X_train.iloc[idxs_new].reset_index(drop=True)
    y_res = y_train.iloc[idxs_new].reset_index(drop=True)

    print("After hybrid sampling:", Counter(y_res))
    return X_res, y_res

def train_and_eval(df: pd.DataFrame):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    # Hybrid resample on TRAIN ONLY
    X_res, y_res = hybrid_resample(X_train, y_train)

    # TF-IDF + Linear SVM
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
        ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_res, y_res)

    # Evaluate on untouched test set
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title("Confusion Matrix (Hybrid Sampling, Linear SVM)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_png = CHART_DIR / "confusion_matrix_svm_hybrid.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Saved confusion matrix -> {out_png}")

    # Save model
    model_path = MODEL_DIR / "tfidf_svm_hybrid.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved model -> {model_path}")

def main():
    df = load_clean_data()
    train_and_eval(df)

if __name__ == "__main__":
    main()