import pandas as pd
from preprocessing import preprocess_texts
from feature_extraction import get_features
from sklearn.decomposition import LatentDirichletAllocation

# Load dataset
df = pd.read_csv("../data/raw/bbc_news/bbc-text.csv")   # adjust path if needed

# Raw text column
raw_text = df['text']

# PHASE 2: Preprocessing
clean_text = preprocess_texts(raw_text)

# PHASE 3: Feature Extraction
X, tfidf = get_features(clean_text)

# PHASE 4: LDA Model Training
lda_model = LatentDirichletAllocation(
    n_components=5,
    random_state=67,
    learning_method='batch'
)

lda_model.fit(X)
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(lda_model, "models/lda_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(X, "models/doc_term_matrix.pkl")

print("✅ Phase 4 completed: LDA model trained")
print(type(clean_text))
print(X.shape)
print(lda_model.components_.shape)

print("✅ Model and vectorizer saved successfully")
feature_names = tfidf.get_feature_names_out()

for idx, topic in enumerate(lda_model.components_):
    print(f"\nTopic {idx + 1}:")
    print([feature_names[i] for i in topic.argsort()[:-11:-1]])

