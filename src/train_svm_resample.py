import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

PROC_PATH = Path("data/processed/reviews_clean.csv")
CHART_DIR = Path("outputs/charts")
MODEL_DIR = Path("models")
CHART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(PROC_PATH)
    X = df["text_clean"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline
    pipe = ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=50000,
            min_df=2
        )),
        # oversample minority classes to the count of the majority)
        ("over", RandomOverSampler(sampling_strategy='auto', random_state=42)),
        ("clf", LinearSVC(C=1.0))  
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (SVM + Oversampling)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path = CHART_DIR / "confusion_matrix_svm_oversampled.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")

    model_path = MODEL_DIR / "tfidf_svm_oversampled.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved model -> {model_path}")

if __name__ == "__main__":
    main()