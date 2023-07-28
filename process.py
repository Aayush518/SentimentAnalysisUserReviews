import numpy as np
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download the IMDb dataset from the nltk library
nltk.download("movie_reviews")

# Import the movie_reviews dataset from nltk
from nltk.corpus import movie_reviews

# Get the movie reviews data and labels
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to mix positive and negative reviews
np.random.shuffle(documents)

# Separate the text data and labels
X = [" ".join(words) for words, label in documents]
y = [label == 'pos' for words, label in documents]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform the text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Fit the classifier to the training data
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report
report = classification_report(y_test, y_pred)

# Streamlit App
st.title("Movie Review Sentiment Analysis")
st.write("This app uses an SVM model to predict the sentiment (positive/negative) of movie reviews.")

# Text Input for User Review
user_review = st.text_area("Enter your movie review here:")

if user_review:
    # Preprocess the user review and convert it into TF-IDF features
    user_review_tfidf = tfidf_vectorizer.transform([user_review]).toarray()

    # Predict the sentiment of the user review
    sentiment = "Positive" if svm_classifier.predict(user_review_tfidf)[0] else "Negative"

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"The sentiment of the review is {sentiment}.")

# Display Accuracy and Classification Report
st.subheader("Model Performance:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.code(report)
