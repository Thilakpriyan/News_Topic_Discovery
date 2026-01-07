import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_texts(text_series):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_text = []

    for text in text_series:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        clean_text.append(" ".join(tokens))

    return clean_text
