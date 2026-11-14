from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from src.models.build_model import build_model

df = pd.read_csv("data/processed/cleaned_email_spam_dataset.csv")
df["cleaned_email"] = df["cleaned_email"].fillna("")


vectorizer = joblib.load("models/count_vectorizer.pkl")

X = vectorizer.transform(df["cleaned_email"])

y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = build_model()

model.fit(X_train, y_train)

joblib.dump(model, "models/email_spam_detection_model.pkl")

print("Training completed")