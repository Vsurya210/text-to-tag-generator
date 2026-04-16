"""
text_to_tag.py
Text-to-Tag Generator (TF-IDF + Cosine Similarity)

Features:
 - Load tags from tags.csv (tag,description)
 - Optionally load labeled documents for evaluation (dataset.csv)
 - CLI demo + interactive mode
 - Precision@k evaluation helper

This version is safe to run in Jupyter/VSCode and in terminal.
If /mnt/data/tags.csv and /mnt/data/dataset.csv exist, they will be used by default.
Dependencies: pandas, scikit-learn, nltk, numpy
"""

import re
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import os
import sys

# NLP libs
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vectorizer + similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (quiet)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess(text: str, keep_numbers: bool = False) -> str:
    """Lowercase, remove urls/emails, punctuation, numbers (optional), stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|mailto:\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    if not keep_numbers:
        text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    clean = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        lemma = LEMMATIZER.lemmatize(t)
        clean.append(lemma)
    return " ".join(clean)

class TextToTag:
    def __init__(self, tags: List[dict],
                 ngram_range=(1,2), max_df=0.9, min_df=1):
        """
        tags: list of {"tag": "machine learning", "description": "..." }
        """
        if not tags:
            raise ValueError("Provide at least one tag.")
        self.tags = tags
        self.tag_names = [t['tag'] for t in tags]
        # Build tag docs (name + description) and preprocess
        self.tag_docs_raw = [f"{t['tag']}. {t.get('description','')}" for t in tags]
        self.tag_docs = [preprocess(doc) for doc in self.tag_docs_raw]

        # TF-IDF vectorizer fit on tag docs so tags are in the same vector space
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
        self.tag_matrix = self.vectorizer.fit_transform(self.tag_docs)

    def predict(self, text: str, top_k: int = 5, min_score: float = 0.03) -> List[Tuple[str, float]]:
        """
        Return list of (tag, score) sorted by score desc.
        min_score filters out weak matches.
        """
        proc = preprocess(text)
        if not proc.strip():
            return []
        vec = self.vectorizer.transform([proc])
        sims = cosine_similarity(vec, self.tag_matrix).flatten()
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            score = float(sims[i])
            if score >= min_score:
                results.append((self.tag_names[i], score))
        return results

# ---------- Helpers to load CSVs ----------
def load_tags_from_csv(path: str) -> List[dict]:
    """
    CSV columns: tag,description
    """
    df = pd.read_csv("tags.csv")
    tags = []
    for _, row in df.iterrows():
        tags.append({"tag": str(row['tag']).strip(), "description": str(row.get('description','')).strip()})
    return tags

def load_dataset_csv(path: str) -> List[dict]:
    """
    CSV columns: id,text,tags  (tags as semicolon ';' separated ground-truth)
    Returns list of {"id":..., "text":..., "tags":[...]}
    """
    df = pd.read_csv("dataset.csv")
    docs = []
    for _, row in df.iterrows():
        raw_tags = str(row.get('tags','')).strip()
        true_tags = [t.strip() for t in raw_tags.split(';') if t.strip()]
        docs.append({"id": row.get('id'), "text": str(row.get('text','')), "tags": true_tags})
    return docs

# ---------- Evaluation metrics ----------
def precision_at_k(preds: List[List[str]], trues: List[List[str]], k: int = 3) -> float:
    """
    preds: list per doc of predicted tags (ordered)
    trues: list per doc of ground truth tags
    Precision@k averaged over docs
    """
    if len(preds) != len(trues):
        raise ValueError("preds and trues must have same length")
    scores = []
    for p, t in zip(preds, trues):
        topk = p[:k]
        if not topk:
            scores.append(0.0)
            continue
        correct = sum([1 for tag in topk if tag in t])
        scores.append(correct / k)
    return float(np.mean(scores))

def evaluate_model(model: TextToTag, dataset: List[dict], top_k: int = 3) -> dict:
    preds = []
    trues = []
    for item in dataset:
        text = item['text']
        true_tags = item['tags']
        out = model.predict(text, top_k=top_k)
        pred_tags = [tag for tag, _ in out]
        preds.append(pred_tags)
        trues.append(true_tags)
    p_at_k = precision_at_k(preds, trues, k=top_k)
    return {"precision@{}".format(top_k): p_at_k, "n_docs": len(dataset)}

# ---------- Sample tags embedded if no CSV ----------
DEFAULT_TAGS = [
    {"tag": "machine learning", "description": "supervised and unsupervised learning classification regression model"},
    {"tag": "data science", "description": "data cleaning exploratory analysis visualization pandas numpy"},
    {"tag": "natural language processing", "description": "text processing tokenization lemmatization nlp text mining"},
    {"tag": "computer vision", "description": "image processing cnn object detection image classification"},
    {"tag": "web development", "description": "frontend backend api flask django javascript html css"},
    {"tag": "devops", "description": "docker kubernetes ci/cd deployment cloud automation"},
    {"tag": "database", "description": "sql nosql query indexing database design"},
    {"tag": "statistics", "description": "probability hypothesis testing distribution statistical inference"},
    {"tag": "information retrieval", "description": "search ranking tf-idf cosine similarity indexing"},
    {"tag": "nlp applications", "description": "chatbot summarization classification information extraction"}
]

# ---------- CLI demo ----------
def demo_cli(tags_csv: Optional[str] = None, dataset_csv: Optional[str] = None):
    if tags_csv and os.path.exists(tags_csv):
        tags = load_tags_from_csv(tags_csv)
    else:
        tags = DEFAULT_TAGS
    model = TextToTag(tags)
    print("=== Text-to-Tag Generator Demo ===")
    if dataset_csv and os.path.exists(dataset_csv):
        dataset = load_dataset_csv(dataset_csv)
        print(f"Loaded dataset with {len(dataset)} examples from {dataset_csv}")
        print("Evaluating model (Precision@3)...")
        try:
            res = evaluate_model(model, dataset, top_k=3)
            print(res)
        except Exception as e:
            print("Evaluation failed:", e)
    print("\nTry sample texts or enter 'exit'\n")
    while True:
        try:
            q = input("Enter text to tag: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        preds = model.predict(q, top_k=6)
        if not preds:
            print("No tags found (try more descriptive text).")
            continue
        print("Predicted tags (tag: score):")
        for tag, score in preds:
            print(f" - {tag} : {score:.3f}")
    print("Bye!")

# ---------- Main: Jupyter-safe argparse handling ----------
def _default_uploaded_paths():
    # paths where you uploaded files in this environment
    tags_path = "/mnt/data/tags.csv"
    dataset_path = "/mnt/data/dataset.csv"
    return tags_path if os.path.exists(tags_path) else None, dataset_path if os.path.exists(dataset_path) else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Text-to-Tag Generator CLI (Jupyter-safe)")
    parser.add_argument("--tags", type=str, default="tags.csv", help="path to tags.csv")
    parser.add_argument("--dataset", type=str, default=None, help="optional dataset.csv for evaluation")

    # Detect Jupyter/interactive kernel args and handle gracefully
    jupyter_args = any("ipykernel_launcher" in arg or arg.startswith("--f=") for arg in sys.argv)
    if jupyter_args:
        # ignore external kernel args and prefer uploaded files if available
        args = parser.parse_args(args=[])
        uploaded_tags, uploaded_dataset = _default_uploaded_paths()
        if uploaded_tags:
            args.tags = uploaded_tags
        if uploaded_dataset:
            args.dataset = uploaded_dataset
    else:
        args = parser.parse_args()

    demo_cli(tags_csv=args.tags if os.path.exists(args.tags) else None,
             dataset_csv=args.dataset)
