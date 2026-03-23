import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]"," ",text)
    text = re.sub("\s+"," ",text)
    return text

# Title
st.title("Fake News Detection using NLP")

st.write("Enter news text to check if it is Fake or Real")

news = st.text_area("Enter News")

if st.button("Predict"):

    if news == "":
        st.warning("Please enter news text")

    else:
        news = clean_text(news)

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)

        if prediction[0] == 1:
            st.success("Real News")
        else:
            st.error("Fake News")