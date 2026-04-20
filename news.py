import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

##Load Dataset
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

##Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["text"] = data["text"].apply(clean_text)
##Split Data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Convert Text → Numbers
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
##Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

##Predictions
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

##Test Custom Input
def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    result = model.predict(vector)
    return "Real News" if result[0] == 1 else "Fake News"

print(predict_news("Breaking: Government announces new policy"))

##Visualization
import matplotlib.pyplot as plt

labels = ["Fake", "Real"]
values = data["label"].value_counts()

plt.bar(labels, values)
plt.title("Fake vs Real News Distribution")
plt.show()