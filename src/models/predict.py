import pandas
import joblib
from sklearn.model_selection import train_test_split
from src.data.preprocess_data import preprocess

model = joblib.load("models/email_spam_detection_model.pkl")

vectorizer = joblib.load("models/count_vectorizer.pkl")

while True :

    email = input("Enter email : ")

    if not email == "" :

        X = vectorizer.transform([preprocess(email)])
        prediction = model.predict(X)
        print(prediction[0])
        

    else :
        break

