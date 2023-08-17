# Sentiment Analysis of User Reviews

This project involves using a Support Vector Machine (SVM) classifier to perform sentiment analysis on movie reviews. The goal is to predict whether a given movie review expresses a positive or negative sentiment. The SVM model is trained using the IMDb movie reviews dataset obtained from the nltk library.

## Getting Started

To run this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Aayush518/SentimentAnalysisUserReviews.git
   ```

2. Install the required dependencies:

   ```bash
   pip install numpy nltk streamlit scikit-learn
   ```

3. Download the IMDb movie reviews dataset using nltk:

   ```python
   import nltk
   nltk.download("movie_reviews")
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Enter a movie review in the provided text area and see the sentiment prediction.

## Project Structure

The project consists of the following files:

- `app.py`: This is the main Streamlit app file. It contains the user interface and code to predict sentiment based on user input.

- `README.md`: This file provides information about the project, its purpose, and instructions on how to run it.

## Dependencies

The project uses the following Python libraries:

- numpy
- nltk
- streamlit
- scikit-learn

You can install these dependencies using the following command:

```bash
pip install numpy nltk streamlit scikit-learn
```

## Usage

1. Launch the Streamlit app by running the following command:

   ```bash
   streamlit run app.py
   ```

2. Enter your movie review in the provided text area.

3. The app will predict whether the sentiment of the review is positive or negative and display the result.

## Acknowledgments

- The IMDb movie reviews dataset used in this project is obtained from the nltk library. The dataset contains both positive and negative movie reviews, which are used for training and evaluation.

