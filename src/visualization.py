"""
PHASE 7: Topic Visualization & Analysis
Project: News / Document Topic Discovery
Author: Thilakpriyan
"""

# ===============================
# 1. IMPORTS
# ===============================
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ===============================
# 2. PATH SETUP (CASE A)
# models folder is inside src/
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load saved artifacts
lda_model = joblib.load(os.path.join(MODEL_DIR, "lda_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
X = joblib.load(os.path.join(MODEL_DIR, "doc_term_matrix.pkl"))

feature_names = vectorizer.get_feature_names_out()


# ===============================
# 3. TOP WORDS BAR CHART
# ===============================
def plot_top_words(model, feature_names, n_top_words=10):
    """
    Plot top N words for each topic using bar charts
    """
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = topic[top_indices]

        plt.figure(figsize=(8, 4))
        plt.barh(top_words[::-1], weights[::-1])
        plt.title(f"Topic {topic_idx + 1} – Top Words")
        plt.xlabel("Word Weight")
        plt.tight_layout()
        plt.show()


# ===============================
# 4. WORD CLOUD PER TOPIC
# ===============================
def plot_wordcloud(model, feature_names, topic_idx):
    """
    Generate word cloud for a specific topic
    """
    word_freq = {
        feature_names[i]: model.components_[topic_idx][i]
        for i in range(len(feature_names))
    }

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=20
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud – Topic {topic_idx + 1}")
    plt.show()


# ===============================
# 5. DOCUMENT–TOPIC HEATMAP
# ===============================
def plot_document_topic_heatmap(model, X):
    """
    Visualize document-topic distribution
    """
    doc_topic_dist = model.transform(X)

    plt.figure(figsize=(10, 6))
    plt.imshow(doc_topic_dist, aspect="auto", cmap="viridis")
    plt.colorbar(label="Topic Probability")
    plt.xlabel("Topics")
    plt.ylabel("Documents")
    plt.title("Document–Topic Distribution Heatmap")
    plt.show()

    return doc_topic_dist


# ===============================
# 6. DOMINANT TOPIC PER DOCUMENT
# ===============================
def print_dominant_topics(doc_topic_dist, n_docs=10):
    """
    Print dominant topic for first N documents
    """
    dominant_topics = np.argmax(doc_topic_dist, axis=1)

    print("\nSample Document → Dominant Topic")
    for i in range(n_docs):
        print(f"Document {i + 1} → Topic {dominant_topics[i] + 1}")


# ===============================
# 7. RUN ALL VISUALIZATIONS
# ===============================
if __name__ == "__main__":
    print("Running PHASE 7: Visualization & Analysis")

    # 1. Top words per topic
    plot_top_words(lda_model, feature_names, n_top_words=10)

    # 2. WordClouds
    for i in range(lda_model.n_components):
        plot_wordcloud(lda_model, feature_names, i)

    # 3. Document-topic heatmap
    doc_topic_dist = plot_document_topic_heatmap(lda_model, X)

    # 4. Dominant topic display
    print_dominant_topics(doc_topic_dist)

    print("\nPHASE 7 COMPLETED SUCCESSFULLY")
