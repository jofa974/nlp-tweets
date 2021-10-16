from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logger
from src.preprocessors.abstract import Preprocessor


class SKTfidfVectorizer(Preprocessor):
    def __init__(self):
        super(SKTfidfVectorizer, self).__init__()

    def fit(self, texts):
        logger.info("Vectorizing...")
        self.preprocessor = TfidfVectorizer(use_idf=True)
        self.preprocessor.fit(texts)

    def apply(self, texts):
        processed_text = self.preprocessor.transform(texts)
        return processed_text.toarray()

    @property
    def vocab_size(self):
        return len(self.preprocessor.vocabulary_)
