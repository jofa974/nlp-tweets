import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


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


def prepare():

    df = pd.read_csv("data/train.csv")

    df["text_clean"] = df["text"].apply(cleanup)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df["text_clean"].values, df["target"].values, test_size=0.2, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_val = vectorizer.transform(X_val).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    out_path = Path("data/prepared")
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Saving train data...")
    np.savetxt(
        out_path / "train.csv", np.column_stack([X_train, Y_train]), delimiter=","
    )

    log.info("Saving validation data...")
    np.savetxt(
        out_path / "validate.csv", np.column_stack([X_val, Y_val]), delimiter=","
    )

    log.info("Saving test data...")
    np.savetxt(out_path / "test.csv", np.column_stack([X_test, Y_test]), delimiter=",")

    log.info("Done.")


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am preparing the data !")
    prepare()
