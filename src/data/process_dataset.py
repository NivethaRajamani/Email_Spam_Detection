import pandas as pd
from src.data.preprocess_data import clean_batch


def process_dataset():

    print("script started")

    df = pd.read_csv("data/raw/email_spam_dataset.csv")

    print("Loading dataset")

    texts = df["email"].astype(str).str.lower()
    texts = texts.str.replace(r"https?\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z\s]", "", regex=True)

    print("Initial text processing done")

    df["cleaned_email"] = clean_batch(texts)

    print("Data cleaning completed")

    print(df["cleaned_email"].head())


    df.to_csv("data/processed/cleaned_email_spam_dataset.csv", index=False)

    print("File saved")

    print(df["category"].value_counts())
    



