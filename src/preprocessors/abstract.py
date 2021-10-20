import re
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spellchecker import SpellChecker

from .abbreviations import ABBREVIATIONS


class Preprocessor(ABC):
    def __init__(self):
        self.preprocessor = None
        self.out_path = Path("data/prepared") / self.__class__.__name__
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.spell = SpellChecker()

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
    def remove_html(text):
        html = re.compile(r"<.*?>")
        return html.sub(r"", text)

    def correct_spelling(self, text):
        correct_text = []
        misspelled_words = self.spell.unknown(text.split())

        for word in text.split():
            if word in misspelled_words:
                correct_text.append(self.spell.correction(word))
            else:
                correct_text.append(word)

        return " ".join(correct_text)

    @staticmethod
    def remove_mention(text):
        # Remove @ and mention, replace by USER
        at = re.compile(r"@\S+")
        return at.sub(r"USER", text)

    @staticmethod
    def remove_emoji(text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"EMOJI", text)

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

    @staticmethod
    def remove_number(text):
        # Remove numbers, replace it by NUMBER
        num = re.compile(r"[-+]?[.\d]*[\d]+[:,.\d]*")
        return num.sub(r"NUMBER", text)

    @staticmethod
    def transcription_sad(text):
        # Replace some others smileys with SADFACE
        eyes = "[8:=;]"
        nose = "['`\-]"
        smiley = re.compile(r"[8:=;][\'\-]?[(\\/]")
        return smiley.sub(r"SADFACE", text)

    @staticmethod
    def transcription_smile(text):
        # Replace some smileys with SMILE
        smiley = re.compile(r"[8:=;][\'\-]?[)dDp]")
        return smiley.sub(r"SMILE", text)

    @staticmethod
    def transcription_heart(text):
        # Replace <3 with HEART
        heart = re.compile(r"<3")
        return heart.sub(r"HEART", text)

    @staticmethod
    def replace_abbrev(text):
        string = ""
        for word in text.split():
            string += ABBREVIATIONS.get(word.lower(), word) + " "
        return string

    def clean_all(self, text):
        text = self.remove_url(text)
        text = self.remove_emoji(text)
        text = self.replace_abbrev(text)
        text = self.remove_mention(text)
        text = self.remove_number(text)
        text = self.transcription_sad(text)
        text = self.transcription_smile(text)
        text = self.transcription_heart(text)
        # text = self.correct_spelling(text)
        text = self.lemmatize(text)
        return text

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
