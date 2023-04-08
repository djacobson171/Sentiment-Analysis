# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

# Step 2: Load the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/danielpinelis/Downloads/archive (3)/all-data.csv', encoding='ISO-8859-1', header=None)

# Step 3: Explore and visualize the data
print(df.head())
print(df[0].value_counts())

plt.figure(figsize=(8,6))
sns.countplot(x=0, data=df)
plt.title('Sentiment Distribution')
plt.show()

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[1], df[0], test_size=0.2, random_state=42)

# Step 5: Define the preprocessing steps
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text into individual words
    words = word_tokenize(text)

    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stem words
    words = [stemmer.stem(word) for word in words]

    # Join words back into text
    text = ' '.join(words)

    return text

# Step 6: Define the machine learning pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=preprocess_text)),
    ('classifier', MultinomialNB())
])

# Step 7: Train and evaluate the model
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)

# Step 8: Fine-tune the model with GridSearchCV
param_grid = {
    'vectorizer__max_df': [0.9, 1.0],
    'vectorizer__min_df': [1, 2],
    'vectorizer__ngram_range': [(1,1), (1,2)],
    'classifier__alpha': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Accuracy:", grid.best_score_)
print("Best cross-validation score:", grid.best_score_)
print("Test score:", grid.score(X_test, y_test))

# Step 9: Use Vader to analyze sentiment of input text
analyzer = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    # Calculate Vader compound score
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']

    # Classify sentiment based on compound score
    if compound_score > 0.05:
        return 'positive'
    elif compound_score < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Step 10: Allow the user to input new text data and predict its sentiment
while True:
    text = input("Enter a news article: ")
    prediction = pipeline.predict([text])[0]
    print("Sentiment:", prediction)
    isCorrect = input("Was the sentiment correct? (y/n/quit) ")
    
    # Step 11: Display the most informative features for the classifier
    if isCorrect.lower() == 'info':
        top_words = 10
        classifier = pipeline.named_steps['classifier']
        vectorizer = pipeline.named_steps['vectorizer']
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
        top = zip(coefs_with_fns[:top_words], coefs_with_fns[:-(top_words + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    
    # Step 12: Ask for user feedback if prediction is incorrect and incorporate into training data
    elif isCorrect.lower() == 'n':
        feedback = input("Please enter the correct sentiment (positive/negative/neutral): ")
        if feedback in ['positive', 'negative', 'neutral']:
            df = df.append({0: feedback, 1: text}, ignore_index=True)
            X_train, X_test, y_train, y_test = train_test_split(df[1], df[0], test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            accuracy = pipeline.score(X_test, y_test)
            print("Accuracy:", accuracy)
        else:
            print("Invalid input. Please enter 'positive', 'negative', or 'neutral'.")
    
    # Step 13: Allow the user to see the current accuracy of the model
    elif isCorrect.lower() == 'accuracy':
        print("Accuracy:", accuracy)
    
    elif isCorrect.lower() == 'quit':
        break
        
    else:
        continue
