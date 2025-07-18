import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("../data/tweets.csv")

df.dropna(subset=["text"], inplace=True)
df = df[df["text"].str.strip() != ""]

X = df["text"]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

os.makedirs("../app", exist_ok=True)
joblib.dump(pipeline, "../app/profanity.joblib")
X_val.to_csv("../app/X_val.csv", index=False)
pd.DataFrame({"label": y_val}).to_csv("../app/y_val.csv", index=False)

print("Model + val data saved.")

# y_pred = pipeline.predict(X_val)
# print("Classification Report:")
# print(classification_report(y_val, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_val, y_pred))
# print("Validation Accuracy:", pipeline.score(X_val, y_val))

# os.makedirs("../app", exist_ok=True)
# joblib.dump(pipeline, "../app/profanity.joblib")
# print("Model saved to ../app/profanity.joblib")