📰 News Topic Discovery & Prediction System

An unsupervised machine learning project that discovers hidden topics from news articles using Latent Dirichlet Allocation (LDA) and predicts the most relevant topic for new user-provided news paragraphs through a Streamlit web application.

📌 Project Overview

In the modern digital era, large volumes of textual data such as news articles, blogs, and reports are generated every day. Manually organizing and understanding these documents is time-consuming and inefficient.

This project implements an automated News Topic Discovery and Prediction system using unsupervised machine learning. The system learns latent topics from a news dataset and allows users to input a new paragraph to identify its most relevant topic based on learned word patterns.

The project focuses on topic discovery, interpretability, and real-time inference, without relying on labeled data or GPU-intensive models.

🎯 Objectives

Automatically discover hidden topics from large text datasets

Eliminate the need for manual labeling

Apply unsupervised machine learning techniques

Interpret topics using keyword analysis

Predict topics for unseen user input text

Build a clean and user-friendly web interface

🧩 Project Phases
🔹 Phase 1: Problem Understanding & Dataset Collection

Studied topic modeling concepts

Selected a news dataset (BBC News articles)

Analyzed dataset size and structure

Output: Dataset ready for preprocessing

🔹 Phase 2: Data Preprocessing (NLP)

Converted text to lowercase

Removed punctuation and special characters

Removed stopwords

Performed tokenization and lemmatization

Why:
Raw text is noisy and cannot be directly used by ML models.

Output: Clean and normalized text data

🔹 Phase 3: Feature Extraction

Converted text into numerical form using TF-IDF Vectorization

Why:
Machine learning models operate only on numerical data.

Output: Document–Term Matrix

🔹 Phase 4: Model Training

Trained Latent Dirichlet Allocation (LDA) model

Used 10 topics (n_components = 10) for better interpretability

Model trained in an unsupervised manner

Output: Trained topic discovery model

🔹 Phase 5: Topic Extraction & Interpretation

Extracted top keywords for each topic

Analyzed word distributions

Manually assigned topic names based on dominant keywords

Important Note:
Since LDA is unsupervised, topic labels are inferred by humans, not learned by the model.

🔹 Phase 6: Model Evaluation

Evaluated using:

Topic coherence (qualitative)

Human interpretability

Accuracy is not applicable due to lack of ground-truth labels

Output: Validated topic quality

🔹 Phase 7: Visualization & Analysis (Offline)

Visualized:

Top keywords per topic (bar charts)

Word clouds

Document–topic distributions

Used Matplotlib and WordCloud

Purpose:
To understand and interpret discovered topics, not for prediction.

🔹 Phase 8: Web Application (Prediction Only)

Built a Streamlit web app

Users can input a new news paragraph

App predicts top matching topics with probabilities

Displays keywords of the predicted topic for explainability

⚠ Dataset visualizations are not shown in the web app.

🧠 Final Topic Naming (Based on Keyword Analysis)
Topic ID	Assigned Name
Topic 1	Movies, Awards & Actors
Topic 2	International Economy & Trade
Topic 3	Home Entertainment & Media Technology
Topic 4	Online Media, Blogs & Betting
Topic 5	International Sports & Global Events
Topic 6	Sports & Popular Culture
Topic 7	Social Media, Celebrities & Online Platforms
Topic 8	General News & Mixed Content
Topic 9	Telecommunications & Digital Regulation
Topic 10	Gaming Industry & Financial Markets

🌐 Web Application Features

Text input for new news paragraphs

Topic prediction using trained LDA model

Displays Top-3 matching topics with probabilities

Clean and minimal UI

Designed for real-time inference only

🛠 Technologies Used
Category	Tools
Language	Python
NLP	NLTK
ML Model	Scikit-learn (LDA)
Vectorization	TF-IDF
Visualization	Matplotlib, WordCloud
Web App	Streamlit

📂 Project Structure
News_Topic_Discovery/
│
├── app.py                    # Streamlit web app (prediction only)
│
├── src/
│   ├── visualization.py      # Phase 7 offline analysis
│   ├── topic_analysis.py     # Topic keyword extraction
│   └── models/
│       ├── lda_model.pkl
│       ├── tfidf_vectorizer.pkl
│       └── doc_term_matrix.pkl
│
├── data/
│   └── raw/
│
├── README.md

▶ How to Run the Project
🔹 Run the Web App
streamlit run app.py

🔹 Run Offline Visualization
python src/visualization.py