import pandas as pd
import nltk
import spacy
import regex as re

nlp = spacy.load("en_core_web_sm")

def preprocess(text):

    text = str(text).lower()
    text = re.sub(r"https?\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)

    words = []

    for token in doc :

        if not token.is_stop :
            words.append(token.lemma_)

    return " ".join(words)


def clean_batch(text) :

    cleaned = []

    docs = nlp.pipe(text,
                    disable = ["parser", "ner"],
                    batch_size = 200)

    for doc in docs :

        words = [token.lemma_ for token in doc if not token.is_stop]

        cleaned.append(" ".join(words))

    return cleaned




