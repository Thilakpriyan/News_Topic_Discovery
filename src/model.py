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
    n_components=50,
    random_state=67
)

lda_model.fit(X)

print("✅ Phase 4 completed: LDA model trained")
print(type(clean_text))
print(X.shape)
print(lda_model.components_.shape)
