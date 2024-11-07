
# imdb_sentiment_analysis.py
import re
import numpy as np
from tensorflow.keras.datasets import imdb
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow as tf

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bs4 import BeautifulSoup
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()
nb_model_embeddings = GaussianNB()
nb_model = MultinomialNB()


def tokenize_text(text): # tokenizing all words in text -cw
    return word_tokenize(text)

def remove_punctuation_numbers(text):
    return re.sub(r'[^a-zA-Z\s]', '', text) # removing punctuation and numbersv-cw

def to_lowercase(text):
    return text.lower() # just a method to make the text lower case -cw

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text() # I wanna take out any kinda html tags that could be in the data -cw

def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


def preprocess_text(text): # grabbing all my functions i created above and putting them into this method -cw
    text = to_lowercase(text)
    text = remove_html(text)
    text = remove_punctuation_numbers(text)
    text = remove_stopwords(text) #this function is located below
    text = stem_text(text) 
    return text
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
    return ' '.join([word for word in text.split() if word not in stop_words])

# Feature extraction using TF-IDF
def extract_features(X_train, X_test): # changed name of this method to accomodate both functions -cw
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # added bag of words -cw
    vectorizer_bow = CountVectorizer(max_features=5000)
    X_train_bow = vectorizer_bow.fit_transform(X_train)
    X_test_bow = vectorizer_bow.transform(X_test)
    return X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow

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

def train_word2vec(X_train_tokens): # training word embeddings on the data set
    model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model

def get_feature_vectors(reviews, model, vector_size):
    feature_vectors = []
    for review in reviews:
        vector = np.zeros(vector_size)
        count = 0
        for word in review:
            if word in model.wv:
                vector += model.wv[word]
                count += 1
        if count != 0:
            vector /= count
        feature_vectors.append(vector)
    return np.array(feature_vectors)

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
    return embeddings_index

def get_embedding_matrix(word_index, embeddings_index, vector_size):
    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def train_feedforward_nn(X_train, y_train, input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    return model


def train_lstm(X_train, y_train, max_sequence_length, vocab_size, embedding_matrix):
    model = Sequential()
    vector_size = 100
    model.add(Embedding(input_dim=vocab_size, output_dim=vector_size, weights=[embedding_matrix],
                        input_length=max_sequence_length, trainable=False))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    return model

# Main function to run all steps
def main():
    # Load and decode data
    X_train, y_train, X_test, y_test = load_imdb_dataset()
    X_train_text = decode_reviews(X_train)
    X_test_text = decode_reviews(X_test)

    # Preprocess text data
    X_train_text = [preprocess_text(review) for review in X_train_text]
    X_test_text = [preprocess_text(review) for review in X_test_text]

    # Extract features using TF-IDF and BoW
    X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow = extract_features(X_train_text, X_test_text)

    # Initialize models
    nb_model = MultinomialNB()
    svm_model = LinearSVC()
    logreg_model = LogisticRegression(max_iter=1000)

    # Evaluate models with TF-IDF features
    print("Evaluating models with TF-IDF features:")
    evaluate_model(nb_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    evaluate_model(svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    evaluate_model(logreg_model, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # Evaluate models with BoW features
    print("Evaluating models with BoW features:")
    evaluate_model(nb_model, X_train_bow, X_test_bow, y_train, y_test)
    evaluate_model(svm_model, X_train_bow, X_test_bow, y_train, y_test)
    evaluate_model(logreg_model, X_train_bow, X_test_bow, y_train, y_test)

    # Tokenize text data for Word2Vec
    X_train_tokens = [tokenize_text(review) for review in X_train_text]
    X_test_tokens = [tokenize_text(review) for review in X_test_text]

    # Train Word2Vec model
    word2vec_model = train_word2vec(X_train_tokens)

    # Create averaged word embeddings for reviews
    vector_size = 100  # Same as in Word2Vec model
    X_train_embeddings = get_feature_vectors(X_train_tokens, word2vec_model, vector_size)
    X_test_embeddings = get_feature_vectors(X_test_tokens, word2vec_model, vector_size)

    # Evaluate models with Word Embeddings
    print("Evaluating models with Word Embeddings:")
    nb_model_embeddings = GaussianNB()
    svm_model_embeddings = LinearSVC()
    logreg_model_embeddings = LogisticRegression(max_iter=1000)

    evaluate_model(nb_model_embeddings, X_train_embeddings, X_test_embeddings, y_train, y_test)
    evaluate_model(svm_model_embeddings, X_train_embeddings, X_test_embeddings, y_train, y_test)
    evaluate_model(logreg_model_embeddings, X_train_embeddings, X_test_embeddings, y_train, y_test)

    # Train and evaluate Feedforward Neural Network
    print("Training Feedforward Neural Network with Word Embeddings:")
    input_dim = X_train_embeddings.shape[1]
    nn_model = train_feedforward_nn(X_train_embeddings, y_train, input_dim)
    loss, accuracy = nn_model.evaluate(X_test_embeddings, y_test)
    print(f"Feedforward Neural Network Accuracy: {accuracy:.4f}")

    # Prepare data for LSTM
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train_text)
    X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_text)
    max_sequence_length = 500
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)
    vocab_size = len(tokenizer.word_index) + 1

    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, vector_size))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # Train and evaluate LSTM model
    print("Training LSTM Neural Network:")
    lstm_model = train_lstm(X_train_padded, y_train, max_sequence_length, vocab_size, embedding_matrix)
    loss, accuracy = lstm_model.evaluate(X_test_padded, y_test)
    print(f"LSTM Neural Network Accuracy: {accuracy:.4f}")




if __name__ == "__main__":
    main()
