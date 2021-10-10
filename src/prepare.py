import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm


def cleanup(text):

    """Remove substrings that will hinder the model:
    - urls
    - ??

    Parameters
    ----------
    text: str
        A string to be processed.

    Returns
    -------
    str:
        The cleaned up string.
    """
    return re.sub(r"https?://\S+", "", text)


def lemmatize_text(nlp, text):
    text = "".join(ch for ch in text if ch.isalnum() or ch == " ")
    text = nlp(text)
    lemma = " ".join([token.lemma_ for token in text if token.text not in STOP_WORDS])
    return lemma


def prepare():

    log.info("I am preparing the data !")

    tqdm.pandas()

    df = pd.read_csv("data/raw/train.csv")
    df["text_clean"] = df["text"].progress_apply(cleanup)

    nlp = spacy.load("en_core_web_sm")
    df["text_clean"] = df["text_clean"].progress_apply(lambda x: lemmatize_text(nlp, x))

    X_train, X_test, Y_train, Y_test = train_test_split(
        df["text_clean"].values, df["target"].values, test_size=0.2, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    log.info("Vectorizing...")
    vectorizer = CountVectorizer(stop_words="english")
    vectorizer.fit(X_train)

    out_path = Path("data/prepared")
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_path / "vectorizer.joblib")

    log.info("Saving texts...")
    texts = {"train": X_train.tolist(), "val": X_val.tolist(), "test": X_test.tolist()}
    with open(out_path / "texts.json", "w") as f:
        json.dump(texts, f)

    log.info("Saving labels...")
    labels = {"train": Y_train.tolist(), "val": Y_val.tolist(), "test": Y_test.tolist()}
    with open(out_path / "labels.json", "w") as f:
        json.dump(labels, f)

    log.info("Done.")


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    prepare()
