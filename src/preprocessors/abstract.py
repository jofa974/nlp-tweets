import re
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


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

    @property
    @abstractmethod
    def vocab_size(self):
        pass