import pandas as pd

df = pd.read_csv("/home/fahd/Documents/ML Projects/imdb_sentiment/train_data.csv", encoding="latin-1")

count_0 = (df['sentiment'] == 0).sum()
count_1 = (df['sentiment'] == 1).sum()

print(f"Number of 0s: {count_0}")
print(f"Number of 1s: {count_1}")
