import pandas as pd, joblib
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load("../app/profanity.joblib")
X_val = pd.read_csv("../app/X_val.csv").squeeze("columns")
y_val = pd.read_csv("../app/y_val.csv").squeeze("columns")

#quick sanity check
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

y_pred = model.predict(X_val)

print("Classification Report:")
print(classification_report(y_val, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("Validation Accuracy:", model.score(X_val, y_val))