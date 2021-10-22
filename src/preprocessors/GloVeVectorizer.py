import re

import numpy as np
import tensorflow as tf
from src.logger import logger
from src.preprocessors.TFTokenizer import TFTokenizer
from tqdm import tqdm


class GloVeVectorizer(TFTokenizer):
    def __init__(self):
        super().__init__()
        self.glove_file = "ext/glove.twitter.27B.200d.txt"
        self.embedding_dict = {}
        with open(
            self.glove_file,
            "r",
        ) as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], "float32")
                self.embedding_dict[word] = vectors

    def apply(self, texts):
        processed_text = super().apply(texts)
        self.make_embedding_matrix()
        return processed_text

    def make_embedding_matrix(self):
        """This method must be called after self.apply()"""
        logger.info("Making embedding matrix")
        self.embedding_matrix = np.zeros((self.vocab_size + 1, self.output_dim))
        for word, i in tqdm(self.word_index.items()):
            # TODO: check wtf is this "if"
            if i > self.vocab_size + 1:
                continue

            emb_vec = self.embedding_dict.get(word)
            if emb_vec is not None:
                self.embedding_matrix[i] = emb_vec

    @property
    def output_dim(self):
        return int(re.findall("(\d+)d", self.glove_file)[0])
