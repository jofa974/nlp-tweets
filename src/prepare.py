import argparse
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from constants import PARAMS
from logger import logger


class Preprocessor(ABC):
    def __init__(self):
        self.out_path = Path("data/prepared")
        self.out_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def preprocess(self, texts):
        pass

    @abstractmethod
    def save(self, out_path):
        pass

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
        self.preprocess()
        self.save()

        logger.info("Done.")


class PreprocessorTFTokenizer(Preprocessor):
    def __init__(self):
        super(PreprocessorTFTokenizer, self).__init__()
        self.name = "tokenizer"
        self.tokenizer = None

    def save(self):
        joblib.dump(self.tokenizer, self.out_path / f"{self.name}.joblib")

    def preprocess(self):
        logger.info("Tokenizing...")
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.df["text_clean"])
        self.tokenizer = tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="PreprocessorTFTokenizer",
        help="A model name. Must be a class registered in src/models.py:factory",
    )

    preprocessor = PreprocessorTFTokenizer()
    preprocessor.prepare()
