import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

vectorizer = joblib.load("models/count_vectorizer.pkl")

model = joblib.load("models/email_spam_detection_model.pkl")

cleaned_email_dataset = pd.read_csv("data/processed/cleaned_email_spam_dataset.csv")

x = vectorizer.transform(cleaned_email_dataset["cleaned_email"])

y = cleaned_email_dataset["category"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


prediction = model.predict(x_test)

print(classification_report(y_test, prediction))

labels = ["spam", "not-spam"]

cm = confusion_matrix(y_test, prediction, labels = labels)

cm_table = pd.DataFrame(cm, index = ["actual_spam", "actual_not_spam"], columns = ["predicted_spam", "predicted_not_spam"])

print("\nConfusion Matrix (rows = actual, columns = predicted)")
print(cm_table)









