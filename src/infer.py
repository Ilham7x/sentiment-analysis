import sys
import joblib
from pathlib import Path

MODEL_PATH = Path("models/tfidf_logreg_pipeline.joblib")

def predict(review_text: str) -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train first: python src/train_baseline.py")
    pipe = joblib.load(MODEL_PATH)
    return pipe.predict([review_text])[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py \"Your review text here\"")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    label = predict(text)
    print(f"Predicted sentiment: {label}")
