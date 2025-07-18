import pandas as pd
import re
from html import unescape

def clean_text(text):
    if pd.isnull(text):
        return ""

    text = unescape(text)
    text = text.strip()

    words = text.split()
    if not words:
        return ""

    if words[0].lower() == "rt":
        words = words[2:]
    else:
        words = words[1:]

    text = " ".join(words)

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = re.sub(r"\brt\b", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")
df = df[["tweet", "class"]].rename(columns={"tweet": "text", "class": "label"})
df["label"] = df["label"].apply(lambda x: 1 if x in [0, 1] else 0)
df["text"] = df["text"].apply(clean_text)
df.to_csv("../data/tweets.csv", index=False)