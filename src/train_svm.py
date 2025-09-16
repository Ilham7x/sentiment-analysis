import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

PROC_PATH = Path("data/processed/reviews_clean.csv")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
CHART_DIR = Path("outputs/charts")
for p in [MODEL_DIR, OUT_DIR, CHART_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def load_clean():
    if not PROC_PATH.exists():
        raise FileNotFoundError("file not found")
    df = pd.read_csv(PROC_PATH)
    return df

def train_and_eval(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            sublinear_tf=True,      
            max_df=0.9,             
            min_df=5,                
            max_features=100000
        )),
        ("svm", LinearSVC(
            C=1.0,
            class_weight="balanced",  # counter imbalanced classes
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    labels = ["negative","neutral","positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (LinearSVC)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix_svm.png")
    plt.close()
    print(f"Saved confusion matrix -> {CHART_DIR / 'confusion_matrix_svm.png'}")

    joblib.dump(pipe, MODEL_DIR / "tfidf_svm_pipeline.joblib")
    print(f"Saved model pipeline -> {MODEL_DIR / 'tfidf_svm_pipeline.joblib'}")


def main():
    df = load_clean()
    train_and_eval(df)

if __name__ == "__main__":
    main()
