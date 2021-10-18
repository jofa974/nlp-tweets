# Natural Language Processing with Disaster Tweets

## Description

This repo is used to practice the developement of ML models for NLP and building good habits with MLOps.

The initial requirements are:

```bash
tensorflow
tensorflow-gpu
dvc
dvc[gdrive]
matplotlib
pandas
jupyter
ipykernel
pytest
black
flake8
joblib
spacy
sklearn
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl
pyspellchecker
kaggle
```

The `requirements.txt` file will be updated along the way to ensure reproducibility.

## Plan

### Minimalist pipeline

- Make a simple pipeline that executes quickly (fetch data, train, evaluate, produce output).
- Establish a github workflow using CML actions.
- Design a basic software architecture.
- The very first model should be very basic and constitute the baseline. For example: random choice between 0 and 1.

### NLP preprocessing

The data preprocessing will constitute an important part of the development. The following approaches will be successively implemented:

- CountVectorizer from sklearn
- TfidfVectorizer from sklearn
- TFTokenizer from tensorflow

### Model

Classical ML and deep learning models will be investigated:

- logistic regression,
- random forest,
- deep neural networks with TensorFlow using LSTM, GRU, Conv1D ...

### More ideas

- learning rate scheduler
- GloVe
- Bert
