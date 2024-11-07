
# imdb_sentiment_analysis.py

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')

# Load IMDb dataset
def load_imdb_dataset():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=50000)
    return X_train, y_train, X_test, y_test

# Decode reviews from integer-encoded format to text
def decode_review(review, index_to_word):
    return " ".join([index_to_word.get(i, "?") for i in review])

def decode_reviews(X_data):
    word_index = imdb.get_word_index()
    index_to_word = {index + 3: word for word, index in word_index.items()}
    index_to_word[0] = "<PAD>"
    index_to_word[1] = "<START>"
    index_to_word[2] = "<UNK>"
    index_to_word[3] = "<UNUSED>"
    return [decode_review(review, index_to_word) for review in X_data]

# Remove stopwords from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

# Feature extraction using TF-IDF
def extract_features_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

# Model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Main function to run all steps
def main():
    # Load and decode data
    X_train, y_train, X_test, y_test = load_imdb_dataset()
    X_train_text = decode_reviews(X_train)
    X_test_text = decode_reviews(X_test)

    # Remove stopwords
    X_train_text = [remove_stopwords(review) for review in X_train_text]
    X_test_text = [remove_stopwords(review) for review in X_test_text]

    # Extract TF-IDF features
    X_train_tfidf, X_test_tfidf = extract_features_tfidf(X_train_text, X_test_text)

    # Initialize models
    nb_model = MultinomialNB()
    svm_model = SVC(kernel='linear')
    logreg_model = LogisticRegression(max_iter=1000)

    # Train and evaluate models
    evaluate_model(nb_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    evaluate_model(svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    evaluate_model(logreg_model, X_train_tfidf, X_test_tfidf, y_train, y_test)

if __name__ == "__main__":
    main()
