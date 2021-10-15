import argparse
import json
import re
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import joblib
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from src.dataset import Dataset
from src.logger import logger


class Preprocessor(ABC):
    def __init__(self):
        self.preprocessor = None
        self.out_path = Path("data/prepared") / self.__class__.__name__
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.nlp = spacy.load("en_core_web_sm")

    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def apply(self, texts):
        pass

    def save(self):
        joblib.dump(self.preprocessor, self.out_path / "preproc.joblib")

    def load(self):
        self.preprocessor = joblib.load(self.out_path / "preproc.joblib")

    @staticmethod
    def remove_url(text):
        """Remove url substrings that will hinder the model:

        Parameters
        ----------
        text: str
            A string to be processed.

        Returns
        -------
        str:
            The string without urls.
        """
        return re.sub(r"https?://\S+", "", text)

    def lemmatize(self, text):
        """Transform text string to string of lemmas using spacy.

        Parameters
        ----------
        text: str
            A string to be processed.

        Returns
        -------
        str:
            The string of lemmas.
        """
        text = "".join(ch for ch in text if ch.isalnum() or ch == " ")
        text = self.nlp(text)
        lemma = " ".join(
            [token.lemma_ for token in text if token.text not in STOP_WORDS]
        )
        return lemma

    def prepare(self):

        logger.info("I am preparing the data !")

        # self.clean_text()
        # self.tts()
        # self.make_preprocessor()
        # self.save()

        logger.info("Done.")

    @property
    @abstractmethod
    def vocab_size(self):
        pass


class TFTokenizer(Preprocessor):
    def __init__(self):
        super(TFTokenizer, self).__init__()

    def fit(self):
        logger.info("Tokenizing...")
        self.preprocessor = tf.keras.preprocessing.text.Tokenizer()
        self.preprocessor.fit_on_texts(self.df["text_clean"])

    def apply(self, texts):
        processed_text = self.preprocessor.texts_to_sequences(texts)
        processed_text = tf.keras.preprocessing.sequence.pad_sequences(
            processed_text, padding="post"
        )
        return processed_text

    @property
    def vocab_size(self):
        return len(self.preprocessor.word_index)


class SKCountVectorizer(Preprocessor):
    def __init__(self):
        super(SKCountVectorizer, self).__init__()

    def fit(self, texts):
        logger.info("Vectorizing...")
        self.preprocessor = CountVectorizer()
        self.preprocessor.fit(texts)

    def apply(self, texts):
        processed_text = self.preprocessor.transform(texts)
        return processed_text

    @property
    def vocab_size(self):
        return len(self.preprocessor.vocabulary_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="SKCountVectorizer",
        help="A preprocessor's name. Must be a sub-class of Preprocessor",
    )
    args = parser.parse_args()
    # TODO: use factory
    constructors = {"SKCountVectorizer": SKCountVectorizer, "TFTokenizer": TFTokenizer}

    ds = Dataset()
    ds.load_raw_to_df(raw_file="data/raw/train.csv")

    preprocessor = constructors[args.preprocessor]()
    ds.prepare_features(preprocessor)
    ds.train_test_split(out_path=Path("data/prepared") / args.preprocessor)

    preprocessor.fit(ds._features)
    preprocessor.save()
