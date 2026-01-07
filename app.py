import os
import joblib
import numpy as np
import streamlit as st

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="News Topic Prediction",
    layout="centered"
)

st.title("📰 News Topic Prediction System")
st.markdown(
    "Enter a news paragraph below. "
    "The system predicts the most relevant topic using a trained LDA model."
)

# ===============================
# Load Model & Vectorizer
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")

lda_model = joblib.load(os.path.join(MODEL_DIR, "lda_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

feature_names = vectorizer.get_feature_names_out()

# ===============================
# Topic Names
# ===============================
topic_names = {
    1: "Business",
    2: "International Economy & Trade",
    3: "Home Entertainment & Media Technology",
    4: "Online Media, Blogs & Betting",
    5: "International Sports & Global Events",
    6: "Movies, Awards & Actors",
    7: "Social Media, Celebrities & Online Platforms",
    8: "General News & Mixed Content",
    9: "Telecommunications & Digital Regulation",
    10: "Gaming Industry & Financial Markets"
}

def get_topic_name(topic_id):
    return topic_names.get(topic_id, f"Topic {topic_id}")

# ===============================
# Inference Logic
# ===============================
def predict_topics(text):
    text_vec = vectorizer.transform([text])
    probs = lda_model.transform(text_vec)[0]
    return probs

# ===============================
# User Input
# ===============================
st.subheader("✍️ Enter News Paragraph")

user_text = st.text_area(
    "Paste a complete news paragraph (minimum 40 words recommended):",
    height=220
)

# ===============================
# Prediction Output
# ===============================
if st.button("🔍 Predict Topic"):

    if len(user_text.split()) < 40:
        st.warning("⚠ Please enter at least 40 words for better prediction.")
        st.stop()

    probs = predict_topics(user_text)

    # Show Top-3 topics
    top_topics = np.argsort(probs)[-3:][::-1]

    st.success("✅ Predicted Topics")

    for idx in top_topics:
        topic_id = idx + 1
        st.write(
            f"**Topic {topic_id} ({get_topic_name(topic_id)})** "
            f"→ Probability: {probs[idx]:.3f}"
        )
