import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Movie Review Sentiment Analysis")
st.write("Enter your review :")
user_input = st.text_area("Input review here")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        st.success(f"Prediction: {prediction[0]}")