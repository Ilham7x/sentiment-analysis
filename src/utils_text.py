import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def basic_clean(text: str) -> str:
    if text is None:
        return ""
    t = text.lower().strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)          # URLs
    t = re.sub(r"<.*?>", " ", t)                     # HTML tags
    t = re.sub(r"[^a-z\s]", " ", t)                  # keep letters + spaces
    t = re.sub(r"\s+", " ", t)                       # collapse whitespace
    return t.strip()

def tokenize(text: str):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    return [w for w in tokens if w not in STOP_WORDS and len(w) > 1]

def lemmatize(tokens):
    return [LEMMATIZER.lemmatize(w) for w in tokens]

def preprocess(text: str, do_stopwords=True, do_lemmatize=True) -> str:
    t = basic_clean(text)
    toks = tokenize(t)
    if do_stopwords:
        toks = remove_stopwords(toks)
    if do_lemmatize:
        toks = lemmatize(toks)
    return " ".join(toks)