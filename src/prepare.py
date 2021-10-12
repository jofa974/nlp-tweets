import argparse
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from constants import PARAMS
from logger import logger


class Preprocessor(ABC):
    def __init__(self):
        self.preprocessor = None
        self.out_path = Path("data/prepared") / self.__class__.__name__
        self.out_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def make_preprocessor(self, texts):
        pass

    def load(self):
        self.preprocessor = joblib.load(self.out_path / "preproc.joblib")

    @abstractmethod
    def apply_preprocessor(self, texts):
        pass

    def save(self):
        joblib.dump(self.preprocessor, self.out_path / "preproc.joblib")

    def cleanup(self, text):

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

    def lemmatize_text(self, nlp, text):
        text = "".join(ch for ch in text if ch.isalnum() or ch == " ")
        text = nlp(text)
        lemma = " ".join(
            [token.lemma_ for token in text if token.text not in STOP_WORDS]
        )
        return lemma

    def read_clean(self):
        df = pd.read_csv("data/raw/train.csv")
        df["text_clean"] = df["text"].progress_apply(self.cleanup)

        nlp = spacy.load("en_core_web_sm")
        df["text_clean"] = df["text_clean"].progress_apply(
            lambda x: self.lemmatize_text(nlp, x)
        )
        self.df = df

    def tts(self, save=True):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.df["text_clean"].values,
            self.df["target"].values,
            test_size=0.2,
            random_state=42,
        )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X_train, self.Y_train, test_size=0.2, random_state=42
        )
        if save:
            logger.info("Saving texts...")
            texts = {
                "train": self.X_train.tolist(),
                "val": self.X_val.tolist(),
                "test": self.X_test.tolist(),
            }
            with open(self.out_path / "texts.json", "w") as f:
                json.dump(texts, f)

            logger.info("Saving labels...")
            labels = {
                "train": self.Y_train.tolist(),
                "val": self.Y_val.tolist(),
                "test": self.Y_test.tolist(),
            }
            with open(self.out_path / "labels.json", "w") as f:
                json.dump(labels, f)

    def prepare(self):

        logger.info("I am preparing the data !")

        tqdm.pandas()

        self.read_clean()
        self.tts()
        self.make_preprocessor()
        self.save()

        logger.info("Done.")


class TFTokenizer(Preprocessor):
    def __init__(self):
        super(TFTokenizer, self).__init__()

    def make_preprocessor(self):
        logger.info("Tokenizing...")
        self.preprocessor = tf.keras.preprocessing.text.Tokenizer()
        self.preprocessor.fit_on_texts(self.df["text_clean"])

    def apply_preprocessor(self, texts):
        processed_text = self.preprocessor.texts_to_sequences(texts)
        processed_text = tf.keras.preprocessing.sequence.pad_sequences(
            processed_text, padding="post"
        )
        return processed_text


class SKCountVectorizer(Preprocessor):
    def __init__(self):
        super(SKCountVectorizer, self).__init__()

    def make_preprocessor(self):
        logger.info("Vectorizing...")
        self.preprocessor = CountVectorizer()
        self.preprocessor.fit(self.df["text_clean"])

    def apply_preprocessor(self, texts):
        processed_text = self.preprocessor.transform(texts)
        return processed_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="TFTokenizer",
        help="A preprocessor's name. Must be a sub-class of Preprocessor",
    )
    args = parser.parse_args()
    # TODO: use factory
    constructors = {"SKCountVectorizer": SKCountVectorizer, "TFTokenizer": TFTokenizer}

    preprocessor = constructors[args.preprocessor]()
    preprocessor.prepare()
