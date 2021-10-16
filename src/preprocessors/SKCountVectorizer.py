from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logger
from src.preprocessors.abstract import Preprocessor


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
