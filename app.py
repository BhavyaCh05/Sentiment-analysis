import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st

nltk.download('vader_lexicon')

st.title("Simple Sentiment Analysis")

@st.cache
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score

user_input = st.text_input("Enter a sentence:")
if user_input:
    score = analyze_sentiment(user_input)
    st.write("Sentiment Analysis Result:")
    st.write(f"Positive: {score['pos']:.2f}")
    st.write(f"Negative: {score['neg']:.2f}")
    st.write(f"Neutral: {score['neu']:.2f}")
    st.write(f"Compound: {score['compound']:.2f}")

