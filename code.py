# Disaster Tweet Classification Script

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. Configuration
MAX_WORDS = 5000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 5

# 3. Load Data
def load_data(train_path='train.csv', test_path='test.csv'):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# 4. Exploratory Data Analysis (EDA)
def plot_distributions(train_data):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=train_data)
    plt.title("Disaster (1) vs. Non-Disaster (0) Tweets Distribution")
    plt.show()

    train_data['text_length'] = train_data['text'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['text_length'], bins=30, kde=True)
    plt.title("Tweet Length Distribution")
    plt.xlabel("Tweet Length")
    plt.ylabel("Frequency")
    plt.show()

def plot_wordcloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(text_data))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Common Words in Disaster-Related Tweets")
    plt.show()

# 5. Data Cleaning
def clean_text(text, stopwords):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return " ".join([word for word in text.split() if word not in stopwords])

def preprocess_data(train_data):
    stopwords = set(["i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", 
                     "what", "which", "who", "this", "that", "am", "is", "are", "was", "were", 
                     "be", "been", "have", "has", "do", "does", "did", "a", "an", "the", "and", 
                     "but", "if", "or", "because", "as", "until", "of", "at", "by", "for", "with", 
                     "about", "between", "into", "through", "during", "before", "after", "to", 
                     "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                     "further", "then", "once", "here", "there", "when", "where", "why", "how", 
                     "all", "any", "both", "each", "few", "more", "most", "other", "some", "no", 
                     "not", "only", "same", "so", "too", "very", "can", "will", "just"])
    
    train_data['cleaned_text'] = train_data['text'].apply(lambda x: clean_text(x, stopwords))
    return train_data

# 6. Tokenize and Pad Sequences for LSTM
def tokenize_and_pad_sequences(text_data, max_words, max_length):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences, tokenizer

# 7. Define LSTM Model
def build_lstm_model(input_dim, embedding_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 8. Train LSTM Model
def train_lstm_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
    return history

# 9. Evaluate Model
def evaluate_model(model, X_val, y_val):
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

# 10. TF-IDF and Random Forest Classifier
def train_tfidf_rf_model(train_data):
    tfidf = TfidfVectorizer(max_features=MAX_WORDS)
    X_tfidf = tfidf.fit_transform(train_data['cleaned_text']).toarray()
    X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = train_test_split(X_tfidf, train_data['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train_tfidf, y_train_tfidf)
    y_pred_tfidf = rf_model.predict(X_val_tfidf)
    
    print("Random Forest Classifier Results:")
    print("Accuracy:", accuracy_score(y_val_tfidf, y_pred_tfidf))
    print("Confusion Matrix:\n", confusion_matrix(y_val_tfidf, y_pred_tfidf))
    print("Classification Report:\n", classification_report(y_val_tfidf, y_pred_tfidf))

# 11. Main Execution
if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()

    # Exploratory Data Analysis
    plot_distributions(train_data)

    # Preprocess data to create the 'cleaned_text' column
    train_data = preprocess_data(train_data)

    # Plot word cloud using the 'cleaned_text' column
    plot_wordcloud(train_data[train_data['target'] == 1]['cleaned_text'])

    # Tokenize and pad sequences for LSTM
    X, tokenizer = tokenize_and_pad_sequences(train_data['cleaned_text'], MAX_WORDS, MAX_SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X, train_data['target'], test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Build and train LSTM model
    lstm_model = build_lstm_model(input_dim=MAX_WORDS, embedding_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
    print("Training LSTM Model...")
    train_lstm_model(lstm_model, X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS)

    # Evaluate LSTM Model
    evaluate_model(lstm_model, X_val, y_val)

    # Train and evaluate TF-IDF + Random Forest model
    print("\nTraining Random Forest with TF-IDF...")
    train_tfidf_rf_model(train_data)
