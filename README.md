[Here](https://akhilvreddy.com/posts/deploying-model/) is the original blog post.

---

# Profanity Detection API (BentoML + Fly.io)

This repository contains a fully automated profanity detection service using

- **scikit-learn** for modeling
- **BentoML** for model serving
- **GitHub Actions** for CI/CD
- **GHCR (GitHub Container Registry)** for Docker image storage
- **Fly.io** for cloud deployment

## Model Pipeline

- Trained on a labeled dataset of offensive tweets
- Uses `TfidfVectorizer` + `LogisticRegression`
- Outputs a probability score and binary classification (`is_profane`)

## Local Development

### Install requirements

```bash
pip install -r /app/requirements.txt
pip install -r /model/requirements.txt
```

### Train and evaluate

```bash
python app/train.py
python app/evaluate.py
```

### Local model tests
```bash
python app/check_model.py "your text here"
```

### Serving locally
```bash
bentoml serve service_bento:ProfanityService
```

## Demo

Please look at [`notebook/modelserve.ipynb`](https://github.com/akhilvreddy/deploying-profanity-service/blob/main/notebook/modelserve.ipynb) for the full demonstration of this working.