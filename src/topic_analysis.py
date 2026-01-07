# src/topic_analysis.py

import pandas as pd
import joblib

# Load trained objects from Phase 4
lda_model = joblib.load("models/lda_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
X = joblib.load("models/doc_term_matrix.pkl")


def display_topics(model, feature_names, num_top_words=10):
    """
    Display top keywords for each topic
    """
    for topic_idx, topic in enumerate(model.components_):
        print(f"\n🔹 Topic {topic_idx + 1}:")
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-num_top_words - 1:-1]
        ]
        print(", ".join(top_words))


def get_dominant_topics(model, X):
    """
    Get dominant topic for each document
    """
    topic_dist = model.transform(X)
    dominant_topics = topic_dist.argmax(axis=1)

    return pd.DataFrame({
        "Document_Index": range(len(dominant_topics)),
        "Dominant_Topic": dominant_topics
    })


if __name__ == "__main__":
    feature_names = tfidf.get_feature_names_out()

    # 1️⃣ Display topic keywords
    display_topics(lda_model, feature_names, num_top_words=10)

    # 2️⃣ Document–Topic analysis
    doc_topics = get_dominant_topics(lda_model, X)
    print("\n📌 Dominant topics (first 5 documents):")
    print(doc_topics.head())
