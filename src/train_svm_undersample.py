import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
from pathlib import Path

PROC_PATH = Path("data/processed/reviews_clean.csv")
OUT_DIR = Path("outputs")
CHART_DIR = Path("outputs/charts")
MODEL_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CHART_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(PROC_PATH)
    X, y = df["text_clean"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Before undersampling:", Counter(y_train))

    # Undersample majority (positive class)
    rus = RandomUnderSampler(sampling_strategy="not minority", random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train.to_frame(), y_train)

    print("After undersampling:", Counter(y_train_res))

    # Train pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
        ("clf", LinearSVC())
    ])
    pipe.fit(X_train_res["text_clean"], y_train_res)

    # Evaluate
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=["negative","neutral","positive"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["negative","neutral","positive"],
                yticklabels=["negative","neutral","positive"])
    plt.title("Confusion Matrix (SVM Undersample)")
    plt.savefig(CHART_DIR / "confusion_matrix_svm_undersampled.png")
    plt.close()

    joblib.dump(pipe, MODEL_DIR / "tfidf_svm_undersampled.joblib")
    print("Saved model -> models/tfidf_svm_undersampled.joblib")

if __name__ == "__main__":
    main()