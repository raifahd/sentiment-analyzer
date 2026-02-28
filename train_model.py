import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load IMDB
df_imdb = pd.read_csv("/home/fahd/Documents/ML Projects/imdb_sentiment/IMDB_Dataset.csv")
df_imdb = df_imdb.dropna(subset=["review", "sentiment"])
df_imdb = df_imdb.rename(columns={"review": "text"})
df_imdb = df_imdb[["text", "sentiment"]]

# Load Sentiment140
df_twitter = pd.read_csv("/home/fahd/Documents/ML Projects/imdb_sentiment/train_data.csv", encoding="latin-1")
df_twitter.columns = ["text", "sentiment"]

df_twitter["sentiment"] = df_twitter["sentiment"].map({
    0: "negative",
    1: "positive"
})

df_twitter = df_twitter[["text", "sentiment"]]

# Load Sentiment140 Test
df_test = pd.read_csv("/home/fahd/Documents/ML Projects/imdb_sentiment/test_data.csv", encoding="latin-1")
df_test.columns = ["text", "sentiment"]

df_test["sentiment"] = df_test["sentiment"].map({
    0: "negative",
    1: "positive"
})

df_test = df_test[["text", "sentiment"]]

# Combine train Data
df_combined = pd.concat([df_imdb, df_twitter], ignore_index=True)
df_combined = df_combined.dropna(subset=["text", "sentiment"])
print("Training Size: ", df_combined.shape)

# Train Model
X_text = df_combined["text"]
y = df_combined["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Test on Sentiment140 Test Set
X_test_text = df_test["text"].astype(str)
X_test = vectorizer.transform(X_test_text)

y_test = df_test["sentiment"]
test_pred = model.predict(X_test)

print("Test Accuracy (Sentiment140 Test):")
print(accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# Save model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved successfully.")