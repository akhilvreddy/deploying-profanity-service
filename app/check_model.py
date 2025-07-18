import sys
import joblib

if len(sys.argv) < 2:
    print("Usage: python check_model.py \"your sentence here\"")
    sys.exit(1)

text = sys.argv[1]

model = joblib.load("profanity.joblib")

proba = model.predict_proba([text])[0][1]
print(f"\n Text: {text}")
print(f"is_profane: {proba > 0.5} | confidence: {round(proba, 4)*100}%")