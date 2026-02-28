import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="General Sentiment Analyzer",
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
st.title("🧠 General Sentiment Analyzer")
st.markdown("### Powered by Machine Learning")

st.markdown("---")

user_input = st.text_area(
    "Enter any text:",
    placeholder="Type your sentence here..."
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vector = vectorizer.transform([user_input])

        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector).max()

        st.markdown("### Prediction Result")

        if prediction == "positive":
            st.success(f"✅ Positive")
        else:
            st.error(f"❌ Negative")
        
        st.write(f"Confidence: **{round(probability*100, 2)}%**")

st.markdown("---")
st.markdown("Built with Logistic Regression + TF-IDF")