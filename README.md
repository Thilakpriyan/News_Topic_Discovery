News Topic Discovery and Prediction System
1. Project Description

This project implements a News Topic Discovery and Prediction system using unsupervised machine learning techniques. The system learns hidden topics from a collection of news articles and predicts the most relevant topic for a new, unseen paragraph provided by the user.

The project does not rely on labeled data. Instead, it uses Latent Dirichlet Allocation (LDA) to identify patterns in word usage and group documents based on similar themes.

2. Dataset

The project uses a news article dataset containing multiple categories of news content. The dataset is used only for training and analysis purposes. No labels are required for model training.

3. Project Workflow

The project is implemented in multiple phases, each representing a key step in building a topic modeling system.

4. Phase 1: Problem Understanding and Dataset Preparation

In this phase, the concept of topic modeling is studied and a suitable news dataset is selected. The dataset structure and size are analyzed to ensure it is appropriate for unsupervised learning.

Output:
Raw dataset ready for preprocessing.

5. Phase 2: Text Preprocessing

Raw text data cannot be used directly for machine learning. Therefore, natural language processing techniques are applied to clean and normalize the text.

Steps performed:

Convert text to lowercase

Remove punctuation and special characters

Remove stopwords

Tokenize text into words

Apply lemmatization

Output:
Cleaned and normalized text data.

6. Phase 3: Feature Extraction

Machine learning models require numerical input. In this phase, textual data is converted into numerical form using TF-IDF vectorization.

Output:
Document–Term Matrix representing the importance of words in documents.

7. Phase 4: Model Training

Latent Dirichlet Allocation (LDA) is trained using the document–term matrix. The model is configured with 10 topics to ensure better interpretability and reduced topic overlap.

Although LDA is not a predictive model in the traditional sense, training is required to learn the latent topic distributions from the dataset.

Output:
Trained LDA topic model.

8. Phase 5: Topic Extraction and Interpretation

After training, the model produces a set of topics, each represented by a list of important keywords. These keywords are analyzed manually to understand the theme of each topic.

Since the model is unsupervised, topic names are assigned based on keyword interpretation rather than predefined labels.

Output:
Interpreted and named topics.

9. Phase 6: Model Evaluation

Traditional accuracy metrics are not applicable because no ground-truth labels exist. Instead, evaluation is performed using:

Topic coherence (qualitative analysis)

Human interpretability of keywords

Distribution of topics across documents

Output:
Validated topic quality.

10. Phase 7: Visualization and Analysis

To understand the model behavior, offline visualizations are created. These include:

Bar charts of top keywords per topic

Word clouds for topic interpretation

Document–topic distribution analysis

These visualizations are used only for analysis and are not part of the deployed web application.

Output:
Visual insights into discovered topics.

11. Phase 8: Web Application for Topic Prediction

A Streamlit web application is developed to allow users to input a new news paragraph. The trained LDA model predicts the most relevant topic for the input text.

The web interface focuses only on prediction and does not display dataset visualizations.

Output:
User-input topic prediction system.

12. Final Topic Naming

Topics are manually named based on dominant keywords extracted from the LDA model. The topic names reflect the underlying patterns learned by the model rather than strict predefined categories.

13. Project Structure
News_Topic_Discovery/
│
├── app.py                  # Streamlit application for prediction
├── src/
│   ├── visualization.py    # Offline topic visualization
│   ├── topic_analysis.py   # Topic keyword extraction
│   └── models/
│       ├── lda_model.pkl
│       ├── tfidf_vectorizer.pkl
│       └── doc_term_matrix.pkl
│
├── data/
│   └── raw/
│
└── README.md

14. How to Run the Project

Run the web application:

streamlit run app.py


Run offline visualization:

python src/visualization.py

15. Conclusion

This project demonstrates how unsupervised learning techniques can be applied to analyze and organize large collections of textual data. Despite the absence of labeled data, the system is able to discover meaningful topics and provide real-time topic prediction for new documents