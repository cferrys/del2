import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# Load the training data
train_data = pd.read_csv('train.csv')

# 1. Basic Data Overview
print("Dataset Information:")
train_data.info()
print("\nSample Data:")
print(train_data.head())

# 2. EDA - Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=train_data)
plt.title("Disaster (1) vs. Non-Disaster (0) Tweets Distribution")
plt.show()

# EDA - Tweet Length Distribution
train_data['text_length'] = train_data['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(train_data['text_length'], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Length of Tweet")
plt.ylabel("Frequency")
plt.show()

# 3. Data Cleaning and Visualization
# Define basic stopwords (without external packages)
basic_stopwords = set(["i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", 
                       "what", "which", "who", "this", "that", "am", "is", "are", "was", "were", 
                       "be", "been", "have", "has", "do", "does", "did", "a", "an", "the", "and", 
                       "but", "if", "or", "because", "as", "until", "of", "at", "by", "for", "with", 
                       "about", "between", "into", "through", "during", "before", "after", "to", 
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                       "further", "then", "once", "here", "there", "when", "where", "why", "how", 
                       "all", "any", "both", "each", "few", "more", "most", "other", "some", "no", 
                       "not", "only", "same", "so", "too", "very", "can", "will", "just"])

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return " ".join([word for word in text.split() if word not in basic_stopwords])

train_data['cleaned_text'] = train_data['text'].apply(clean_text)

# Word Cloud for Disaster Tweets
disaster_text = " ".join(train_data[train_data['target'] == 1]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(disaster_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Common Words in Disaster-Related Tweets")
plt.show()
