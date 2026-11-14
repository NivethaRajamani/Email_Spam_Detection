import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

cleaned_dataset = pd.read_csv("data/processed/cleaned_email_spam_dataset.csv")

cleaned_dataset["cleaned_email"] = cleaned_dataset["cleaned_email"].fillna("")

vectorizer = CountVectorizer(max_features=10000, ngram_range = (1,2))

vectorized_features = vectorizer.fit_transform(cleaned_dataset["cleaned_email"])

joblib.dump(vectorizer, "models/count_vectorizer.pkl")

print(vectorized_features.shape)