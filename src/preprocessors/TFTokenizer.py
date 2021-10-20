import tensorflow as tf
from src.logger import logger
from src.preprocessors.abstract import Preprocessor


class TFTokenizer(Preprocessor):
    def __init__(self):
        super(TFTokenizer, self).__init__()

    def fit(self, texts):
        logger.info("Tokenizing...")
        self.preprocessor = tf.keras.preprocessing.text.Tokenizer()
        self.preprocessor.fit_on_texts(texts)

    def apply(self, texts):
        processed_text = self.preprocessor.texts_to_sequences(texts)
        processed_text = tf.keras.preprocessing.sequence.pad_sequences(
            processed_text, padding="post"
        )
        return processed_text

    @property
    def vocab_size(self):
        return len(self.preprocessor.word_index)

    @property
    def word_index(self):
        return self.preprocessor.word_index
