from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import hashlib

RAW_PATH   = Path("data/raw/amazon_reviews.csv")
CLEAN_PATH = Path("data/processed/reviews_clean.csv")   
MODEL_PATH = Path("models/tfidf_svm_hybrid.joblib")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / "predictions.csv"

def build_combined_text(row):
    title = str(row.get("reviews.title", "") or "").strip()
    body  = str(row.get("reviews.text", "") or "").strip()
    if title and body:
        sep = "" if title.endswith((".", "!", "?", "…")) else "."
        return f"{title}{sep} {body}".strip()
    elif body:
        return body
    else:
        return title

def first_asin(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        parts = [p.strip(" '\"") for p in s[1:-1].split(",") if p.strip()]
        return parts[0] if parts else np.nan
    return s

def make_uid(row):
    key = f"{row.get('product_id','')}|{row.get('date','')}|{row.get('text','') or row.get('review_text','')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

def main():
    raw = pd.read_csv(RAW_PATH)
    raw["combined_text"] = raw.apply(build_combined_text, axis=1)

    clean = pd.read_csv(CLEAN_PATH)  

    merged = clean.merge(
        raw,
        left_on="text",
        right_on="combined_text",
        how="inner",
        suffixes=("_clean", "_raw"),
    )
    if merged.empty:
        raise RuntimeError(
            "Join produced 0 rows. Check that clean 'text' = (reviews.title + '. ' + reviews.text)."
        )

    # 4) Metadata for Power BI
    if "reviews.date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["reviews.date"], errors="coerce").dt.date
    elif "reviews.dateSeen" in merged.columns:
        merged["date"] = pd.to_datetime(merged["reviews.dateSeen"], errors="coerce").dt.date
    else:
        merged["date"] = pd.NaT

    merged["product_id"] = merged["asins"].apply(first_asin) if "asins" in merged.columns else np.nan
    merged["brand_out"]  = merged.get("brand", np.nan)
    merged["rating_out"] = merged.get("reviews.rating", np.nan)

    pipe = joblib.load(MODEL_PATH)
    yhat = pipe.predict(merged["text_clean"].astype(str))

    label_to_num = {"negative": -1, "neutral": 0, "positive": 1}
    sent_num = [label_to_num.get(lbl, 0) for lbl in yhat]

    merged["review_uid"] = merged.apply(make_uid, axis=1)

    # 7) Export Power BI csv
    out = pd.DataFrame({
        "review_id": merged["review_uid"],      
        "date": merged["date"],
        "product_id": merged["product_id"],
        "product_name": merged.get("name"),
        "brand": merged["brand_out"],
        "rating": merged["rating_out"],
        "review_text": merged["text"],          
        "pred_label": yhat,
        "sentiment_score": sent_num,
    })

    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved {OUT_CSV} with {len(out)} rows.")

if __name__ == "__main__":
    main()
