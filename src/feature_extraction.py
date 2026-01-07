from sklearn.feature_extraction.text import TfidfVectorizer

def get_features(clean_text):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2)
    X = tfidf.fit_transform(clean_text)
    return X, tfidf
