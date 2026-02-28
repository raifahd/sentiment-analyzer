import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

# Page Configuration
st.set_page_config(
    page_title="Advanced Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("🧠 Advanced Sentiment Analyzer")
st.markdown("### Powered by ML + Explainability")

st.markdown("---")
st.subheader("Single Text Analysis")

user_input = st.text_area("Enter text:")

if st.button("Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        # Transform
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        classes = model.classes_

        st.subheader("📊 Probability Distribution")

        prob_df = pd.DataFrame({
            "Sentiment": classes,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Sentiment"))

        if prediction == "positive":
            st.success("Prediction: 😊 Positive")
        else:
            st.error("Prediction: 😠 Negative")

        st.subheader("🔎 Word Importance Highlight")

        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]

        tokens = re.findall(r"\b\w+\b", user_input.lower())

        highlighted_text = ""

        for word in tokens:
            if word in feature_names:
                index = np.where(feature_names == word)[0][0]
                weight = coef[index]

                if weight > 0:
                    highlighted_text += f" <span style='background-color:#90EE90'>{word}</span>"
                else:
                    highlighted_text += f" <span style='background-color:#FFB6C1'>{word}</span>"
            else:
                highlighted_text += " " + word

        st.markdown(highlighted_text, unsafe_allow_html=True)