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

    "This system predicts the most relevant topic for a **new news article** "
    "using a trained **LDA topic model**."
)

# ===============================
# Load Trained Model
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

lda_model = joblib.load(os.path.join(MODEL_DIR, "lda_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


# ===============================
# Topic Names
# ===============================
topic_names = {
    1: "Technology",
    2: "Business / Economy",
    3: "Sports",
    4: "Sports (Athletics)",
    5: "General News / Politics / Entertainment"
}



def get_topic_name(topic_id):
    return topic_names.get(topic_id, f"Topic {topic_id}")

# ===============================
# Inference Function
# ===============================
def infer_topics(text):
    vec = vectorizer.transform([text])
    probs = lda_model.transform(vec)[0]
    return probs

# ===============================
# USER INPUT
# ===============================
st.subheader("✍️ Enter News Article Text")

user_text = st.text_area(
    "Paste a full news article (minimum 40 words recommended):",
    height=220
)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Topic"):

    if len(user_text.split()) < 40:
        st.warning("⚠ Please enter at least 40 words for better prediction.")
        st.stop()

    probs = infer_topics(user_text)

    # Show Top-3 Topics
    top_topics = np.argsort(probs)[-3:][::-1]

    st.success("✅ Predicted Topics")

    for idx in top_topics:
        topic_id = idx + 1
        st.write(
            f"**Topic {topic_id} ({get_topic_name(topic_id)})** "
            f"→ Probability: {probs[idx]:.3f}"
        )

    # Top words of best topic
    best_topic = top_topics[0]
    st.subheader("🔑 Keywords of Predicted Topic")

