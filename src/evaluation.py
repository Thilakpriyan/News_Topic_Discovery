# src/evaluation.py

from gensim.models import CoherenceModel
from gensim import corpora
import pandas as pd
from pathlib import Path
import joblib

def get_top_words(model, feature_names, n_top_words=10):
    topics = []
    for topic in model.components_:
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]
        ]
        topics.append(top_words)
    return topics


def evaluate_lda_model_sklearn(lda_model, texts, feature_names):
    dictionary = corpora.Dictionary(texts)
    topics = get_top_words(lda_model, feature_names)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    )

    coherence_score = coherence_model.get_coherence()

    print("📊 MODEL EVALUATION RESULTS (SKLEARN LDA)")
    print("-" * 50)
    print(f"✅ Topic Coherence Score : {coherence_score:.4f}")
    print("⚠️ Perplexity Score     : Not applicable for sklearn LDA")

    return coherence_score


if __name__ == "__main__":
    from preprocessing import preprocess_texts
    from model import lda_model   # ✅ ONLY import model

    BASE_DIR = Path(__file__).resolve().parent.parent

    # Load dataset
    data_path = BASE_DIR / "data" / "raw" / "bbc_news" / "bbc-text.csv"
    print("📂 Loading dataset from:", data_path)
    df = pd.read_csv(data_path)

    # Preprocess
    cleaned_texts = preprocess_texts(df["text"])
    tokenized_texts = [text.split() for text in cleaned_texts]

    # ✅ LOAD TF-IDF VECTORIZER (CORRECT NAME)
    # Load TF-IDF vectorizer (CORRECT PATH)
    tfidf_path = BASE_DIR / "src" / "models" / "tfidf_vectorizer.pkl"
    tfidf = joblib.load(tfidf_path)

    feature_names = tfidf.get_feature_names_out()

    evaluate_lda_model_sklearn(
        lda_model,
        tokenized_texts,
        feature_names
    )
topic_8 = lda_model.components_[7]
top_words = topic_8.argsort()[:-15:-1]

for i in top_words:
    print(feature_names[i])
