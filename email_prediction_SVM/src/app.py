import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import joblib

# Function to preprocess text
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

# Load your trained model
# Replace 'email_prediction_model.pkl' with the actual filename of your trained model
# You need to have joblib installed: pip install joblib
loaded_model = joblib.load('email_prediction_model.pkl')

def main():
    st.title("Email Spam Classification")

    # User input for email
    user_email = st.text_area("Enter an email:", "")

    if st.button("Predict"):
        if user_email:
            # Preprocess the input email
            processed_email = text_process(user_email)
            st.write("Processed Email:", processed_email)

            # Tokenize and transform using CountVectorizer
            bow = loaded_model.named_steps['bow'].transform([processed_email])
            st.write("After CountVectorizer:", bow)

            # Transform using TfidfTransformer
            tfidf = loaded_model.named_steps['tfidf'].transform(bow)
            st.write("After TfidfTransformer:", tfidf)

            # Make a prediction and get probability scores using the classifier
            classifier = loaded_model.named_steps['classifier']
            prediction_value = classifier.predict(tfidf)[0]
            probability_spam = classifier.predict_proba(tfidf)[0, 1]
            st.write("Prediction:", prediction_value)
            st.write("Probability of SPAM:", probability_spam)

            # Display the prediction based on probability threshold
            threshold = 0.5  # You can adjust this threshold as needed
            if probability_spam > threshold:
                st.error("Prediction: SPAM")
            else:
                st.success("Prediction: HAM")
        else:
            st.warning("Please enter an email.")


if __name__ == "__main__":
    main()
