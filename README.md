# Disaster Tweet Classification

This project classifies tweets related to natural disasters using a Natural Language Processing (NLP) model. The goal is to determine whether a tweet is disaster-related (`1`) or not (`0`). This README provides an overview of the dataset, project structure, and steps included in the notebook.

## Project Structure
- `train.csv`: Training dataset containing labeled tweets.
- `test.csv`: Test dataset with tweets to classify.
- `sample_submission.csv`: Sample submission file for Kaggle competition.
- `disaster_tweet_classification.ipynb`: Jupyter notebook containing data loading, exploration, preprocessing, model training, and evaluation code.

## Steps

### 1. **Data Loading**
   - Load the training data (`train.csv`) using Pandas for analysis and preprocessing.

### 2. **Exploratory Data Analysis (EDA)**
   - **Target Distribution**: Visualizes the distribution of disaster vs. non-disaster tweets.
   - **Tweet Length Analysis**: Analyzes and visualizes tweet lengths to understand text characteristics.
   - **Common Words**: Generates a word cloud for disaster-related tweets, highlighting frequent terms.

### 3. **Data Cleaning**
   - **Text Cleaning**: Removes URLs, punctuation, and common stopwords to prepare text for modeling.

### 4. **Model Building and Training**
   - **Text Embeddings**: Converts text into numerical form using a word embedding method.
   - **Neural Network Architecture**: Builds an RNN-based model (e.g., LSTM or GRU) to classify tweets.
   - **Hyperparameter Tuning**: Tests various configurations to improve model performance.

### 5. **Evaluation**
   - Evaluates model performance using metrics like accuracy, precision, and recall.

## Usage

To run the project:

1. Clone the repository and open `disaster_tweet_classification.ipynb` in Jupyter Notebook.
2. Ensure all dependencies are installed (e.g., `pandas`, `matplotlib`, `seaborn`, `wordcloud`).
3. Follow each code cell in the notebook to load data, perform EDA, clean the text, train the model, and evaluate results.

## Results
The notebook generates various visualizations and outputs metrics summarizing model performance. Future work includes tuning the model further and testing additional text-processing methods.


