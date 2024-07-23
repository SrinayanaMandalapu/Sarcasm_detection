import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

# Load data from JSON file
with open('sarcasm.json', 'r') as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.2, random_state=42)

# Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Linear Support Vector Classifier
clf = LinearSVC()
clf.fit(X_train_vect, y_train)

# Predict on the test set
predictions = clf.predict(X_test_vect)

joblib.dump(clf, 'sarcasm_detection_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')


